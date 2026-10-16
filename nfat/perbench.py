"""
PerBench: chọn tầng với dữ liệu hiệu chỉnh THEO BENCHMARK (nfat/perbench_data.py). Chạy lại sáu baseline + ig/ig_greedy
theo cùng giao thức, và các phương pháp mới. Điểm tầng -> results/layer_importance.csv (pick_layers.py đọc được),
thời gian chọn tầng (không tính tải mô hình / dựng dữ liệu) -> results/select_cost.csv.

    python nfat/perbench.py --model meta-llama/Llama-3.1-8B --bench sniah --W 256 --S 4 --k 8 \
        --scorers rethead,cake,bi,moa,duo,ig,ig_greedy,sleb --tag pb-sniah-w256
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat import hyb                                               # noqa: E402
from nfat import baselines as B                                    # noqa: E402
from nfat.csv_append import append_row                             # noqa: E402
from nfat.perbench_data import bench_items                         # noqa: E402


def run_scorer(name, model, tok, gates, items, a, dev, log):
    """Trả (vector điểm theo tầng, số mẫu dùng, ghi chú)."""
    n = len(items)
    if name == 'bi':
        return B.score_bi(model, items, dev), n
    if name == 'moa':
        return B.score_moa(model, gates, items, dev), n
    if name == 'cake':
        return B.score_cake(model, items, dev), n
    if name == 'rethead':
        rh = [it for it in items if it.get('span') is not None]
        return B.score_rethead(model, tok, rh, dev, log), len(rh)
    if name == 'duo':
        return B.score_duo(model, gates, items, dev, a.duo_steps, log=log), a.duo_steps
    if name in ('nll_swa', 'nll_swa_dm'):          # NLL-guided Eq. 1: bật riêng tầng l từ toàn SWA, top-k tĩnh
        # nll_swa = BẢN CÔNG BỐ (§3.2: SWA trên prompt, attention đầy đủ trên token đáp án);
        # nll_swa_dm = biến thể của ta: token đáp án theo chế độ giải mã triển khai (--decode).
        hyb.STATE['ans_full'] = True if name == 'nll_swa' else a.decode == 'full'
        try:
            return hyb.nll_swa(model, gates, items, dev, log), n
        finally:
            hyb.STATE['ans_full'] = False
    if name == 'sleb':
        sub = items[:a.sleb_n]                       # items đã xếp xoay vòng tác vụ/độ sâu
        return B.score_sleb(model, sub, dev, a.W, a.S, a.k, log), len(sub)
    if name == 'ig':
        return hyb.integrated_gradients(model, gates, items, a.ig_steps, dev, log), n
    if name == 'ig_greedy':
        order = hyb.ig_greedy(model, gates, items, dev, a.k, a.greedy_steps, log)
        v = np.full(hyb.n_layers(model), -1e9)
        for pos, l in enumerate(order):
            v[l] = -pos
        return v, n
    from nfat import perbench_methods as PM                       # phương pháp mới
    return PM.run(name, model, tok, gates, items, a, dev, log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--bench', required=True, choices=['sniah', 'ruler', 'babilong', 'babilong2', 'longalign', 'niah'])
    p.add_argument('--W', type=int, required=True)
    p.add_argument('--S', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--decode', choices=['full', 'swa'], default='swa')
    p.add_argument('--scorers', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--duo_steps', type=int, default=256)
    p.add_argument('--sleb_n', type=int, default=16)
    p.add_argument('--ig_steps', type=int, default=8)
    p.add_argument('--greedy_steps', type=int, default=3)
    p.add_argument('--extra', default='', help='tham số cho phương pháp mới, dạng a=1,b=2')
    p.add_argument('--out', default='results/layer_importance.csv')
    p.add_argument('--cost_out', default='results/select_cost.csv')
    a = p.parse_args()
    a.extra = dict(kv.split('=') for kv in a.extra.split(',') if kv)
    dev = 'cuda'
    model, tok = hyb.load(a.model, attn='hyb')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    items = bench_items(tok, 'babilong' if a.bench == 'babilong2' else a.bench, a.W, a.n, a.seed)
    if a.bench == 'babilong2':      # đích = câu trả lời greedy của MÔ HÌNH GỐC (đúng định dạng prompt yêu cầu; không dùng test)
        hyb.set_mode('off'); model.eval()
        with torch.no_grad():
            for it in items:
                out = model.generate(torch.tensor([it['ids']], device=dev), max_new_tokens=20, do_sample=False,
                                     pad_token_id=tok.eos_token_id)[0, len(it['ids']):].tolist()
                txt = tok.decode(out, skip_special_tokens=True).split('\n')[0]
                it['gold'] = tok(txt, add_special_tokens=False).input_ids or out[:1]
        model.train()
        print('-> babilong2: đích = trả lời greedy của mô hình gốc, vd', repr(tok.decode(items[0]['gold'])), flush=True)
    L = hyb.n_layers(model)
    print(f'-> {len(items)} mẫu ({a.bench}, TB {np.mean([len(x["ids"]) for x in items]):.0f} token), {L} tầng, '
          f'W={a.W}, S={a.S}, k={a.k}, giải mã {a.decode}', flush=True)
    gates = hyb.Gates(model, a.W, a.S, dev)
    t0 = time.time()
    log = lambda s: print(f'{s}  [{time.time() - t0:.0f}s]', flush=True)   # noqa: E731
    for name in [s for s in a.scorers.split(',') if s]:
        hyb.set_mode('off'); hyb.STATE['ans_full'] = False
        torch.cuda.synchronize(); t = time.time()
        v, n_used = run_scorer(name, model, tok, gates, items, a, dev, log)
        torch.cuda.synchronize(); sec = time.time() - t
        hyb.set_mode('off')
        order = list(np.argsort(-np.asarray(v)))
        log(f'[{name}] {sec / 60:.1f} phút | top-{a.k}: {sorted(int(x) for x in order[:a.k])}')
        rows = [dict(tag=a.tag, model_config=a.model, scorer=name, layer=l, score=float(v[l]),
                     rank=int(order.index(l)), window=a.W, n_sink=a.S, variant=a.bench,
                     L=int(np.mean([len(x['ids']) for x in items])), n_items=n_used, seed=a.seed)
                for l in range(L)]
        append_row(pd.DataFrame(rows), a.out)
        append_row(pd.DataFrame([dict(tag=a.tag, model=a.model, scorer=name, seconds=sec,
                                      n_items=n_used, gpu=torch.cuda.get_device_name())]), a.cost_out)


if __name__ == '__main__':
    main()
