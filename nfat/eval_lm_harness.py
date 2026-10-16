"""
LM Evaluation Harness (lm-eval 0.4.x) cho mô hình gốc / tuyến tính hoá.
Metric giống bảng của LoLCATs:
    piqa: acc | arc_easy: acc | arc_challenge: acc_norm | hellaswag: acc_norm
    winogrande: acc | mmlu (5-shot): acc  (trung bình KHÔNG trọng số theo 57 môn, như script LoLCATs;
                                          bản trọng số theo mẫu cũng được lưu)
"Avg. zero-shot" = trung bình 5 tác vụ đầu (như paper).

Ví dụ:
python nfat/eval_lm_harness.py --model_config nf_llama3_2_1b \
    --distill_ckpt <..._distill.pt> --finetune_ckpt <..._ft.pt> --tag lolcats --out results/lm_eval.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfat.model_io import load_model, is_linearized  # noqa: E402

TASKS = [('piqa', 0, 'acc'), ('arc_easy', 0, 'acc'), ('arc_challenge', 0, 'acc_norm'),
         ('hellaswag', 0, 'acc_norm'), ('winogrande', 0, 'acc'), ('mmlu', 5, 'acc')]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', required=True)
    p.add_argument('--distill_ckpt', default=None)
    p.add_argument('--finetune_ckpt', default=None)
    p.add_argument('--finetune_config', default=None)
    p.add_argument('--tasks', default='all', help='comma list or "all"')
    p.add_argument('--batch_size', type=int, default=None,
                   help='default: 1 for linearized models (LoLCATs masks depend on padded length)')
    p.add_argument('--limit', type=float, default=None)
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/lm_eval.csv')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_lk_zero_init', action='store_true')
    p.add_argument('--huggingface_token', default=None)
    return p.parse_args()


def metric(res: dict, name: str):
    for k in (f'{name},none', name):
        if k in res:
            return float(res[k])
    return float('nan')


def main():
    args = get_args()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model, tok, cfg = load_model(args.model_config, args.distill_ckpt, args.finetune_ckpt,
                                 args.finetune_config, lk_zero_init=not args.no_lk_zero_init,
                                 seed=args.seed, huggingface_token=args.huggingface_token)
    bs = args.batch_size or (1 if is_linearized(model) else 8)
    max_len = getattr(model.config, 'max_position_embeddings', 2048)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=min(max_len, 8192))

    wanted = TASKS if args.tasks == 'all' else [t for t in TASKS if t[0] in args.tasks.split(',')]
    row = dict(tag=args.tag, model_config=args.model_config, distill_ckpt=args.distill_ckpt,
               finetune_ckpt=args.finetune_ckpt, time=time.strftime('%Y-%m-%d %H:%M'))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for task, shots, mname in wanted:
        t0 = time.time()
        out = simple_evaluate(model=lm, tasks=[task], num_fewshot=shots, limit=args.limit,
                              log_samples=False, random_seed=args.seed, numpy_random_seed=args.seed,
                              torch_random_seed=args.seed)
        res = out['results']
        if task == 'mmlu':
            subs = [k for k in res if k.startswith('mmlu_') and k not in (
                'mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other')]
            row['mmlu'] = 100 * float(np.mean([metric(res[s], 'acc') for s in subs]))
            row['mmlu_weighted'] = 100 * metric(res['mmlu'], 'acc')
        else:
            row[task] = 100 * metric(res[task], mname)
            row[f'{task}_stderr'] = 100 * metric(res[task], f'{mname}_stderr')
        print(f'[{task}] {row.get(task):.2f}  ({time.time() - t0:.0f}s)', flush=True)
        with open(args.out.replace('.csv', f'.{args.tag or "run"}.{task}.json'), 'w') as f:
            json.dump(res, f, indent=1, default=str)
    zs = [row[t] for t, _, _ in TASKS[:5] if t in row]
    if len(zs) == 5:
        row['avg_zero_shot'] = float(np.mean(zs))
    df = pd.DataFrame([row])
    df.to_csv(args.out, mode='a', header=not os.path.exists(args.out), index=False)
    print(df.T.to_string())


if __name__ == '__main__':
    main()
