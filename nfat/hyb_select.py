"""
Hiệu chỉnh chọn tầng trên stack mới (transformers >= 4.57), dùng nfat/hyb.py.

Giao thức đối đầu với NLL-guided (arXiv:2606.27791): cùng mô hình, cùng dữ liệu hiệu chỉnh
(LongAlign-10k + fusang-v1-filtered, 64 mẫu 16k–32k token, đáp án >= 20 token), cùng cửa sổ
và keep-first. Chỉ BỘ ƯỚC LƯỢNG khác nhau: nll_full (của họ), ig, ig_greedy (của ta).

    python nfat/hyb_select.py selftest --model Qwen/Qwen3-4B-Thinking-2507
    python nfat/hyb_select.py select --model Qwen/Qwen3-4B-Thinking-2507 \
        --W 2048 --S 10 --k 9 --scorers nll_full,ig,ig_greedy --tag q4t-w2048
Ghi ra results/layer_importance.csv, cùng định dạng với nfat/layer_importance.py nên
nfat/pick_layers.py đọc được.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat import hyb                                               # noqa: E402
from nfat.csv_append import append_row                             # noqa: E402


# ----------------------------------------------------------------------------- dữ liệu
def _render(tok, prompt):
    """ids của prompt theo chat template + phần mở đầu KHÔNG chấm trước đáp án."""
    if not getattr(tok, 'chat_template', None):     # mô hình base: BOS + văn bản + gợi ý trả lời
        return tok(prompt + '\nAnswer:').input_ids
    s = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                add_generation_prompt=True)
    if s.rstrip().endswith('<think>'):              # Qwen3 *-Thinking: đóng khối suy nghĩ rỗng
        s = s + '\n\n</think>\n\n'
    return tok(s, add_special_tokens=False).input_ids


def calib_items(tok, n, min_len, max_len, min_ans, max_ans, seed, log=print):
    """64 mẫu dài từ LongAlign-10k + fusang-v1-filtered, như NLL-guided."""
    from datasets import load_dataset
    rng = random.Random(seed)
    pools = []
    la = load_dataset('THUDM/LongAlign-10k', split='train')
    for r in la:
        m = r['messages']
        if len(m) >= 2 and m[0]['role'] == 'user' and m[1]['role'] == 'assistant':
            pools.append((m[0]['content'], m[1]['content']))
    try:
        fu = load_dataset('yuyijiong/fusang-v1-filtered', split='train')
        cols = fu.column_names
        for r in fu:
            if 'messages' in cols:
                m = r['messages']
                if len(m) >= 2:
                    pools.append((m[0]['content'], m[1]['content']))
            elif 'prompt' in cols and ('answer' in cols or 'response' in cols):
                pools.append((r['prompt'], r.get('answer') or r.get('response')))
    except Exception as e:                                        # noqa: BLE001
        log(f'!! không nạp được fusang-v1-filtered ({e}); chỉ dùng LongAlign-10k')
    rng.shuffle(pools)
    items = []
    for prompt, ans in pools:
        if len(prompt) < min_len * 2:                 # lọc thô theo ký tự trước khi token hoá
            continue
        ids = _render(tok, prompt)
        if not (min_len <= len(ids) <= max_len):
            continue
        gold = tok(ans, add_special_tokens=False).input_ids[:max_ans]
        if len(gold) < min_ans:
            continue
        items.append(dict(ids=ids, gold=gold))
        if len(items) >= n:
            break
    log(f'-> {len(items)} mẫu hiệu chỉnh, độ dài {np.mean([len(i["ids"]) for i in items]):.0f} '
        f'token (min {min(len(i["ids"]) for i in items)}, max {max(len(i["ids"]) for i in items)})')
    return items


# ----------------------------------------------------------------------------- selftest
@torch.no_grad()
def selftest(a):
    """Kiểm chứng ở FP32 bằng relL2 và độ khớp argmax.

    Không so bằng lệch tuyệt đối lớn nhất trong bf16: hai kernel attention đầy đủ đều ĐÚNG
    (SDPA-GQA của ta và SDPA của HF) đã lệch relL2 ~1.2e-2 trong bf16 qua 36 tầng — cùng cỡ
    với chính hiệu ứng cần đo (SWA vs đầy đủ ~4.6e-2). Ở fp32 hai kernel đúng lệch ~3e-6.
    """
    dev, dt = 'cuda', torch.float32
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    txt = open(os.path.join(ROOT, 'data/haystack_text/rp.txt')).read()[:200000]
    ids = torch.tensor([tok(txt).input_ids[:a.L]], device=dev)

    def rel(x, y):
        x, y = x.float(), y.float()
        return ((x - y).norm() / y.norm()).item(), (x.argmax(-1) == y.argmax(-1)).float().mean().item()

    ref, _ = hyb.load(a.model, dtype=dt, attn='sdpa')
    lg_ref = ref(input_ids=ids, use_cache=False).logits.float()
    del ref; torch.cuda.empty_cache()
    model, _ = hyb.load(a.model, dtype=dt, attn='hyb')
    n = hyb.n_layers(model)
    ok = True

    hyb.set_mode('off')
    lg_off = model(input_ids=ids, use_cache=False).logits.float()
    r, g = rel(lg_off, lg_ref); ok &= r < 1e-4
    print(f'  off (hyb) vs sdpa gốc              : relL2 {r:.2e}, argmax {g*100:.2f}%')
    hyb.set_mode('hard', a.W, a.S, full=())
    lg_swa = model(input_ids=ids, use_cache=False).logits.float()
    real = hyb._swa

    def swa_ref(q, k, v, scale, W, S):
        Q, K = q.shape[-2], k.shape[-2]
        qpos = torch.arange(K - Q, K, device=q.device).view(-1, 1)
        kpos = torch.arange(K, device=q.device).view(1, -1)
        keep = (kpos <= qpos) & (((qpos - kpos) < W) | (kpos < S))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep,
                                                                scale=scale, enable_gqa=True)
    hyb._swa = swa_ref
    lg_swa_ref = model(input_ids=ids, use_cache=False).logits.float()
    hyb._swa = real
    r, g = rel(lg_swa, lg_swa_ref); ok &= r < 1e-4
    print(f'  SWA FlexAttention vs mask tường minh: relL2 {r:.2e}, argmax {g*100:.2f}%')
    r, g = rel(lg_swa, lg_ref); ok &= r > 1e-2
    print(f'  (đối chứng SWA vs gốc — phải LỚN)  : relL2 {r:.2e}, argmax {g*100:.2f}%')

    gt = hyb.Gates(model, a.W, a.S, dev)
    gt.set(1.0); r1, _ = rel(model(input_ids=ids, use_cache=False).logits, lg_off)
    gt.set(0.0); r0, _ = rel(model(input_ids=ids, use_cache=False).logits, lg_swa)
    print(f'  cổng z=1 vs off / z=0 vs SWA        : {r1:.1e} / {r0:.1e}')
    ok &= r1 < 1e-5 and r0 < 1e-5          # z=1: o_swa+(o_full-o_swa) ≠ o_full từng bit ở fp32
    hyb.set_mode('hard', a.W, a.S, full=())
    model(input_ids=ids[:, : a.L - 777], use_cache=False)
    print(f'  shape động (L={a.L} rồi L={a.L - 777})     : chạy được')
    del lg_ref, lg_off, lg_swa, lg_swa_ref; torch.cuda.empty_cache()   # ~3.7 GB/cái ở fp32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    Lg = min(a.L, a.W + 1024)                  # vẫn > W để nhánh SWA khác nhánh đầy đủ
    with torch.enable_grad():
        gt.set(0.5); gt.z.grad = None
        loss = hyb.answer_nll(model, [dict(ids=ids[0, :Lg - 64].tolist(),
                                           gold=ids[0, Lg - 64:Lg].tolist())], dev)
        loss.backward()
    fin = bool(torch.isfinite(gt.z.grad).all()) and float(gt.z.grad.abs().sum()) > 0
    print(f'  gradient theo z (checkpointing)     : hữu hạn & khác 0 = {fin}')
    ok &= fin
    print(f'  {n} tầng ->', 'ĐẠT' if ok else 'HỎNG')
    return ok


# ----------------------------------------------------------------------------- chọn tầng
def select(a):
    dev = 'cuda'
    model, tok = hyb.load(a.model, attn='hyb')
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()                       # checkpoint chỉ bật khi training; dropout = 0 ở Qwen3/Llama
    n = hyb.n_layers(model)
    items = calib_items(tok, a.n, a.min_len, a.max_len, a.min_ans, a.max_ans, a.seed)
    gates = hyb.Gates(model, a.W, a.S, dev)
    hyb.STATE['ans_full'] = a.ans_full
    print(f'ans_full={a.ans_full}', flush=True)
    t0 = time.time()
    log = lambda s: print(f'{s}  [{time.time() - t0:.0f}s]', flush=True)   # noqa: E731

    with torch.no_grad():
        gates.set(1.0); nll1 = float(hyb.answer_nll(model, items, dev))
        gates.set(0.0); nll0 = float(hyb.answer_nll(model, items, dev))
    log(f'NLL đáp án: đầy đủ {nll1:.4f} | SWA thuần {nll0:.4f} | khoảng cách {nll0 - nll1:.4f}')

    want = [s for s in a.scorers.split(',') if s]
    scores = {}
    if 'ig' in want:
        scores['ig'] = hyb.integrated_gradients(model, gates, items, a.ig_steps, dev, log)
        s = scores['ig'].sum()
        log(f'ig xong | completeness: tổng {s:.4f} vs khoảng cách {nll0 - nll1:.4f} '
            f'(sai lệch {abs(s - (nll0 - nll1)) / max(1e-9, abs(nll0 - nll1)) * 100:.1f}%)')
    if 'nll_full' in want:
        scores['nll_full'] = hyb.nll_full(model, gates, items, dev, log)
        log('nll_full xong')
    if 'nll_swa' in want:
        scores['nll_swa'] = hyb.nll_swa(model, gates, items, dev, log)
        log('nll_swa xong')
    if 'ig_greedy' in want:
        order = hyb.ig_greedy(model, gates, items, dev, a.k, a.greedy_steps, log)
        v = np.full(n, -1e9)
        for pos, l in enumerate(order):
            v[l] = -pos
        scores['ig_greedy'] = v
        log(f'ig_greedy xong: thứ tự {order}')

    rows = []
    for name, vec in scores.items():
        order = list(np.argsort(-vec))
        print(f'[{name}] top-{a.k}: {sorted(order[:a.k])}', flush=True)
        for l in range(n):
            rows.append(dict(tag=a.tag, model_config=a.model, scorer=name, layer=l,
                             score=float(vec[l]), rank=int(order.index(l)), window=a.W,
                             n_sink=a.S, variant='longalign', L=a.max_len, n_items=len(items),
                             seed=a.seed, nll_z1=nll1, nll_z0=nll0))
    append_row(pd.DataFrame(rows), a.out)
    log(f'-> ghi {len(rows)} dòng vào {a.out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'select'])
    p.add_argument('--model', required=True)
    p.add_argument('--W', type=int, default=2048)
    p.add_argument('--S', type=int, default=10, help='keep_first / sink')
    p.add_argument('--L', type=int, default=4096, help='độ dài cho selftest')
    p.add_argument('--k', type=int, default=9)
    p.add_argument('--scorers', default='nll_full,ig,ig_greedy')
    p.add_argument('--ig_steps', type=int, default=4)
    p.add_argument('--greedy_steps', type=int, default=2)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--min_len', type=int, default=16000)
    p.add_argument('--max_len', type=int, default=32000)
    p.add_argument('--min_ans', type=int, default=20)
    p.add_argument('--max_ans', type=int, default=256)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--no_grad_ckpt', dest='grad_ckpt', action='store_false')
    p.add_argument('--ans_full', action='store_true',
                   help='token đáp án luôn thấy đủ ngữ cảnh (giao thức chấm của NLL-guided)')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/layer_importance.csv')
    a = p.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest(a) else 1)
    select(a)


if __name__ == '__main__':
    main()
