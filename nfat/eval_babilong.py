"""
BABILong (Kuratov et al.) với prompt và hàm chấm CHÍNH THỨC, import thẳng từ repo gốc
booydar/babilong (clone ở ./ext/babilong). Cấu hình theo
scripts/run_model_on_babilong.py: instruction + examples + post_prompt, greedy,
max_new_tokens=20; mô hình base thì không chat template.

    NFAT_SWA_WINDOW=256 NFAT_SWA_SINK=4 \
      python nfat/eval_babilong.py --model_config base_llama3_1_8b_sdpa \
        --tasks qa1,qa2,qa3,qa4,qa5 --lengths 0k,1k,2k,4k --n 100 --tag swa256
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
sys.path.insert(0, os.environ.get('BABILONG_REPO', './ext/babilong'))
from nfat.model_io import load_model                              # noqa: E402
from nfat.csv_append import append_row                            # noqa: E402
from nfat.eval_niah import generate                               # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402
from babilong.metrics import compare_answers, TASK_LABELS         # noqa: E402


def _save_details(a, task, L, hits, swa):
    """Điểm TỪNG MẪU -> results/details/babilong/<tag>.csv (thêm 10-12 cho SwapKL v2: kiểm định ghép cặp theo mẫu).
    Không đổi điểm gộp; lỗi ở đây chỉ cảnh báo, không dừng đánh giá."""
    try:
        rows = pd.DataFrame([dict(tag=a.tag, model_config=a.model_config, W=swa[0], sink=swa[1], layers=swa[2],
                                  dataset=a.dataset, task=task, length=L, i=j, score=float(h)) for j, h in enumerate(hits)])
        append_row(rows, os.path.join(ROOT, 'results', 'details', 'babilong', f'{a.tag or "run"}.csv'))
    except Exception as e:                                        # noqa: BLE001
        print(f'!! không ghi được điểm từng mẫu: {e}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', required=True)
    p.add_argument('--tasks', default='qa1,qa2,qa3,qa4,qa5')
    p.add_argument('--lengths', default='0k,1k,2k,4k')
    p.add_argument('--n', type=int, default=100, help='tối đa 100 mẫu mỗi (task, độ dài)')
    p.add_argument('--dataset', default='RMT-team/babilong')
    p.add_argument('--chat', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/babilong.csv')
    a = p.parse_args()

    from datasets import load_dataset
    model, tok, _ = load_model(a.model_config)
    swa = (os.environ.get('NFAT_SWA_WINDOW', ''), os.environ.get('NFAT_SWA_SINK', ''),
           os.environ.get('NFAT_FULL_LAYERS', ''))
    print(f'-> {a.model_config} | SWA={swa}', flush=True)

    for L in a.lengths.split(','):
        for task in a.tasks.split(','):
            ds = load_dataset(a.dataset, L, split=task)
            ds = ds.select(range(min(a.n, len(ds))))
            pr = DEFAULT_PROMPTS[task]
            t0, hits = time.time(), []
            for s in ds:
                text = get_formatted_input(s['input'], s['question'], pr['examples'],
                                           pr['instruction'], pr['post_prompt'],
                                           template=DEFAULT_TEMPLATE)
                if a.chat:
                    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                                  add_generation_prompt=True)
                else:
                    # Mô hình base cần dấu hiệu đến lượt trả lời. Dùng đúng định dạng của các
                    # ví dụ few-shot trong prompts.py: "Where is Alan?\nAnswer: The most ...".
                    # Không có dòng này, Llama base sinh <|end_of_text|> ngay sau câu hỏi.
                    ids = tok(text + '\nAnswer:').input_ids
                out = generate(model, tok, ids, 20)
                hits.append(float(compare_answers(s['target'], out, s['question'],
                                                  TASK_LABELS[task])))
            acc = float(np.mean(hits))
            print(f'[{task} {L:>4s}] acc={acc:.3f} (n={len(hits)}, '
                  f'{time.time() - t0:.0f}s)', flush=True)
            append_row(pd.DataFrame([dict(tag=a.tag, model_config=a.model_config, task=task,
                                          length=L, acc=acc, n=len(hits), window=swa[0],
                                          n_sink=swa[1], full_layers=swa[2],
                                          secs=time.time() - t0)]), a.out)
            _save_details(a, task, L, hits, swa)


if __name__ == '__main__':
    main()
