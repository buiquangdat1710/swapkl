"""
RULER (NVIDIA) trên mô hình gốc / SWA+sink / lai, dùng bộ sinh ruler_niah.py (bản v2,
khớp key space và template của NVIDIA/RULER). Chấm theo recall chuỗi như RULER.

Mô hình BASE (Llama 3.1 8B không chat template): prompt = BOS + body, trong đó body đã kết
thúc bằng answer prefix của RULER ("... mentioned in the provided text are"), đúng định
dạng RULER dùng cho mô hình base. Mô hình instruct: thêm --chat để dùng chat template.

Cửa sổ / sink / tầng đầy đủ lấy từ biến môi trường NFAT_SWA_WINDOW / NFAT_SWA_SINK /
NFAT_FULL_LAYERS (xem nfat/swa_sink.py); không đặt thì là mô hình gốc.

    NFAT_SWA_WINDOW=256 NFAT_SWA_SINK=4 NFAT_FULL_LAYERS=2,15 \
      python nfat/eval_ruler.py --model_config base_llama3_1_8b_sdpa \
        --tasks S-NIAH-PK,S-NIAH-N,S-NIAH-W --lengths 1024,2048,4096 --n 50 --tag x
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
sys.path.insert(0, os.path.expanduser('~'))                      # ruler_niah.py
from nfat.model_io import load_model                              # noqa: E402
from nfat.csv_append import append_row                            # noqa: E402
from nfat.eval_niah import generate                               # noqa: E402
import ruler_niah as R                                            # noqa: E402

from nfat.ruler_common import RULER_NAME, BaseProcessor, ChatProcessor  # noqa: E402


def _save_details(a, task, L, recs, swa, dhash):
    """Điểm TỪNG MẪU -> results/details/ruler_hf/<tag>.csv (thêm 10-12 cho SwapKL v2: kiểm định ghép cặp theo mẫu).
    Không đổi điểm gộp; lỗi ở đây chỉ cảnh báo, không dừng đánh giá."""
    try:
        rows = pd.DataFrame([dict(tag=a.tag, model_config=a.model_config, W=swa[0], sink=swa[1], layers=swa[2], seed=a.seed,
                                  task=task, length=L, i=j, depth=r['depth'], score=float(r['correct']), data_hash=dhash)
                             for j, r in enumerate(recs)])
        append_row(rows, os.path.join(ROOT, 'results', 'details', 'ruler_hf', f'{a.tag or "run"}.csv'))
    except Exception as e:                                        # noqa: BLE001
        print(f'!! không ghi được điểm từng mẫu: {e}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', required=True)
    p.add_argument('--tasks', default='S-NIAH-PK,S-NIAH-N,S-NIAH-W')
    p.add_argument('--lengths', default='1024,2048,4096')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--seed', type=int, default=777, help='KHÁC seed hiệu chỉnh 1234')
    p.add_argument('--chat', action='store_true', help='dùng chat template (mô hình instruct)')
    p.add_argument('--cache_dir', default='results/ruler_cache')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/ruler.csv')
    a = p.parse_args()

    model, tok, _ = load_model(a.model_config)
    proc = ChatProcessor(tok) if a.chat else BaseProcessor(tok)
    essays = R.load_essay_sentences(split='test')                # tách khỏi phần 'train'
    gen = R.RulerNIAH(proc, essays)
    tk_name = os.path.basename(getattr(tok, 'name_or_path', 'tok')).replace('/', '_')
    cache = os.path.join(a.cache_dir, f"{tk_name}{'-chat' if a.chat else '-base'}")
    swa = (os.environ.get('NFAT_SWA_WINDOW', ''), os.environ.get('NFAT_SWA_SINK', ''),
           os.environ.get('NFAT_FULL_LAYERS', ''))
    print(f'-> {a.model_config} | SWA={swa} | cache {cache}', flush=True)

    for task in a.tasks.split(','):
        for L in [int(x) for x in a.lengths.split(',')]:
            path = R.build_eval_set(gen, task, L, a.n, a.seed, cache_dir=cache)
            data = R.load_eval_set(path)
            t0, recs = time.time(), []
            for s in data:
                ids = tok(s['prompt'], add_special_tokens=False).input_ids
                out = generate(model, tok, ids, R.max_new_tokens(task))
                recs.append(dict(depth=s['depth'], correct=R.score(out, s['answers'])))
            acc = float(np.mean([r['correct'] for r in recs]))
            se = float(np.std([r['correct'] for r in recs], ddof=1) / np.sqrt(len(recs)))
            byd = R.summarize_by_depth(recs)
            print(f'[{task:>9s} L={L:>5d}] acc={acc:.3f} ± {se:.3f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            append_row(pd.DataFrame([dict(
                tag=a.tag, model_config=a.model_config, task=task,
                ruler_task=RULER_NAME.get(task, task), length=L, acc=acc, stderr=se,
                n=len(recs), window=swa[0], n_sink=swa[1], full_layers=swa[2],
                seed=a.seed, data_hash=R.file_hash(path), secs=time.time() - t0,
                **{f'd{int(k)}': v for k, v in byd.items()})]), a.out)
            _save_details(a, task, L, recs, swa, R.file_hash(path))


if __name__ == '__main__':
    main()
