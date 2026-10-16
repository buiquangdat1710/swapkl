"""
Đánh giá truy xuất theo thang độ khó RULER (S-NIAH và các biến thể nhiều kim).

    # mô hình gốc
    python nfat/eval_niah.py --model_config nf_llama3_2_1b_softmax --tag base

    # SWA thuần
    NFAT_SWA_WINDOW=512 NFAT_SWA_SINK=4 \
      python nfat/eval_niah.py --model_config nf_llama3_2_1b_softmax --tag swa512

    # SWA + 2 tầng attention đầy đủ
    NFAT_SWA_WINDOW=512 NFAT_SWA_SINK=4 NFAT_FULL_LAYERS=6,11 \
      python nfat/eval_niah.py --model_config nf_llama3_2_1b_softmax --tag swa512-k2

    # LoLCATs
    python nfat/eval_niah.py --model_config nf_llama3_2_1b --tag lolcats \
      --distill_ckpt <...> --finetune_ckpt <...> --finetune_config nf_ft_alpaca

Mốc để đối chiếu, từ arXiv:2608.28444 ở ngữ cảnh 4K:
    S-NIAH-3  attention đầy đủ 99,8 | SWA+sink 23,0 | LoLCATs 5,8 | Liger-GLA 0,0
    BABILong  attention đầy đủ 60   | SWA+sink 15   | LoLCATs 3   | Liger-GLA 0

Chấm điểm theo RULER: sinh văn bản rồi kiểm giá trị đúng có nằm trong đầu ra không.
Sinh có dùng cache để chạy nổi ở ngữ cảnh dài. LƯU Ý: với mô hình LoLCATs, đường giải mã
có cache KHÔNG trùng lượt truyền xuôi song song (lệch logit tương đối 0,2392) — đó là một
khác biệt kiến trúc có thật của hiện thực gốc, không phải lỗi đo. Dùng `--no_cache` để so
bằng đường truyền xuôi, chậm hơn nhiều.
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
from nfat.model_io import load_model                                   # noqa: E402
from nfat.csv_append import append_row                                 # noqa: E402
from nfat.swa_sink import make_cache_for                               # noqa: E402
from nfat.niah_utils import build, score, load_haystack_words          # noqa: E402

VARIANTS = {
    's1': 'đống rơm lặp lại, số 7 chữ số',
    's2': 'văn bản thật, số 7 chữ số',
    's3': 'văn bản thật, UUID 32 ký tự',
    'mk': 'nhiều kim khác nhãn, hỏi một nhãn',
    'mv': 'một nhãn, nhiều giá trị, lấy đủ',
}


_CACHE_BROKEN = [False]      # cờ dùng chung: một lần hỏng thì cả lần chạy chuyển sang no_cache


def _eos_ids(model, tok):
    """Mọi token kết thúc: eos của tokenizer + danh sách eos trong generation_config.

    Trước đây vòng sinh KHÔNG dừng ở EOS: mô hình base sinh <|end_of_text|> rồi vẫn chạy
    tiếp và sinh rác ("<|begin_of_text|>Question: Where is Daniel?..."). Với RULER thì vô hại
    (chấm recall, đáp án đứng đầu); với BABILong thì làm hỏng điểm vì hàm chấm đòi đúng một
    nhãn trong đầu ra.
    """
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    ge = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    if isinstance(ge, int):
        ids.add(ge)
    elif ge:
        ids.update(int(x) for x in ge)
    return ids


@torch.no_grad()
def generate(model, tok, ids, max_new: int, use_cache: bool = True):
    """
    Sinh tham lam. Ưu tiên đường có cache; nếu mô hình không hỗ trợ thì TỰ LÙI về lượt
    truyền xuôi và nói rõ một lần.

    Vì sao cần lùi: cấu hình lai `softmax_attentions` của kho gốc trộn tầng softmax với
    tầng tuyến tính, nhưng chỉ có MỘT đối tượng cache cho cả mô hình. Tầng nào chạy trước
    sẽ quyết định kiểu cache, rồi tầng kiểu kia gọi `update` sai chữ ký. Với sm_last8 nó
    ném TypeError; với sm_first8 nó KHÔNG ném gì mà cho kết quả sai — nguy hiểm hơn nhiều.
    Đây là một hạn chế có thật của hiện thực gốc, không phải lỗi của phép đo.
    """
    dev = model.device
    x = torch.tensor([ids], device=dev)
    if use_cache and _CACHE_BROKEN[0]:
        use_cache = False
    if not use_cache:
        seq = list(ids)
        out = []
        for _ in range(max_new):
            lg = model(input_ids=torch.tensor([seq], device=dev), use_cache=False).logits[:, -1]
            nxt = int(lg.argmax(-1))
            if nxt in _eos_ids(model, tok):
                break
            out.append(nxt); seq.append(nxt)
        return tok.decode(out)
    try:
        o = model(input_ids=x, use_cache=True, past_key_values=make_cache_for(model))
    except (TypeError, AttributeError) as e:
        _CACHE_BROKEN[0] = True
        print(f'! đường cache không dùng được ({type(e).__name__}: {e}); '
              f'chuyển sang lượt truyền xuôi cho toàn bộ lần chạy này', flush=True)
        return generate(model, tok, ids, max_new, use_cache=False)
    past = o.past_key_values
    tokid = o.logits[:, -1].argmax(-1, keepdim=True)
    out = [int(tokid)]
    stop = _eos_ids(model, tok)
    for _ in range(max_new - 1):
        if out[-1] in stop:
            break
        o = model(input_ids=tokid, past_key_values=past, use_cache=True)
        past = o.past_key_values
        tokid = o.logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(tokid))
    return tok.decode([t for t in out if t not in stop])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', required=True)
    p.add_argument('--distill_ckpt', default=None)
    p.add_argument('--finetune_ckpt', default=None)
    p.add_argument('--finetune_config', default=None)
    p.add_argument('--variants', default='s1,s2,s3,mk,mv')
    p.add_argument('--lengths', default='2048,4096,8192')
    p.add_argument('--depths', default='0,0.25,0.5,0.75,1.0')
    p.add_argument('--n', type=int, default=10, help='số mẫu mỗi ô (độ dài x độ sâu)')
    p.add_argument('--n_extra', type=int, default=3, help='số kim/giá trị phụ cho mk và mv')
    p.add_argument('--haystack', default=os.environ.get('NFAT_NIAH_TEXT', ''),
                   help='thư mục văn bản thật (.txt/.jsonl/.parquet/.arrow); trống -> câu lặp lại')
    p.add_argument('--seed', type=int, default=1234, help='seed dữ liệu, CỐ ĐỊNH giữa các mô hình')
    p.add_argument('--model_seed', type=int, default=0)
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/niah.csv')
    p.add_argument('--huggingface_token', default=None)
    a = p.parse_args()

    words = load_haystack_words(a.haystack)
    if a.haystack and words is None:
        print(f'! không đọc được văn bản từ {a.haystack} -> dùng đống rơm lặp lại cho mọi '
              f'biến thể (s2/s3/mk/mv sẽ DỄ hơn mức chuẩn)', flush=True)
    print(f'-> đống rơm: {"văn bản thật, " + str(len(words)) + " từ" if words else "câu lặp lại"}',
          flush=True)

    model, tok, _ = load_model(a.model_config, a.distill_ckpt, a.finetune_ckpt,
                               a.finetune_config, seed=a.model_seed,
                               huggingface_token=a.huggingface_token)
    model.config.use_cache = not a.no_cache

    lengths = [int(x) for x in a.lengths.split(',')]
    depths = [float(x) for x in a.depths.split(',')]
    variants = [v for v in a.variants.split(',') if v in VARIANTS]
    rows, details = [], []

    for variant in variants:
        max_new = 48 if variant == 's3' else (24 * (a.n_extra + 1) if variant == 'mv' else 16)
        for L in lengths:
            rng = random.Random(a.seed + L + hash(variant) % 1000)
            accs, t0 = [], time.time()
            for d in depths:
                for _ in range(a.n):
                    try:
                        ids, golds, label = build(tok, variant, L, d, rng, words, a.n_extra)
                        txt = generate(model, tok, ids, max_new, use_cache=not a.no_cache)
                        s = score(txt, golds)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); s, txt, golds = float('nan'), 'OOM', []
                    except ValueError as e:      # target_len quá nhỏ
                        print(f'  bỏ qua {variant} L={L}: {e}', flush=True); s = float('nan')
                        txt, golds = '', []
                    accs.append(s)
                    details.append(dict(variant=variant, length=L, depth=d, score=s,
                                        gold='|'.join(golds), pred=txt[:120]))
            acc = float(np.nanmean(accs))
            se = float(np.sqrt(max(acc * (1 - acc), 1e-12) / max(len(accs), 1)))
            by_depth = {f'd{d}': float(np.nanmean(accs[i * a.n:(i + 1) * a.n]))
                        for i, d in enumerate(depths)}
            rows.append(dict(tag=a.tag, model_config=a.model_config, variant=variant,
                             length=L, acc=acc, stderr=se, n=len(accs),
                             secs=time.time() - t0, haystack='natural' if words else 'filler',
                             **by_depth))
            print(f'[{variant} L={L}] acc={acc:.3f} ± {se:.3f}  ({time.time() - t0:.0f}s)  '
                  f'theo độ sâu={ {k: round(v, 2) for k, v in by_depth.items()} }', flush=True)

    df = pd.DataFrame(rows)
    append_row(df, a.out)
    pd.DataFrame(details).to_csv(a.out.replace('.csv', f'.{a.tag or "run"}.details.csv'),
                                 index=False)
    print('\n' + df.pivot_table(index='length', columns='variant', values='acc')
          .round(3).to_string())


if __name__ == '__main__':
    main()
