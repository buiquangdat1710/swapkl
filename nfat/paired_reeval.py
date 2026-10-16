"""
Đánh giá lại (cùng giao thức quét k, seed test 777) các tập CHƯA có điểm từng mẫu để kiểm định ghép cặp theo câu hỏi
(phản biện OE-Agent 10-16, W3): SwapKL và baseline dẫn đầu (theo TB lưới) ở S-NIAH 512/256, RULER, BABILong.
Ghi chi tiết vào results/details/<kind>/pr-*.csv (qua các script đánh giá); KHÔNG đổi ksweep.csv.
    python nfat/paired_reeval.py --only sniah512,ruler
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from nfat import ksweep as KS                     # noqa: E402
from nfat import swk_v2_stats as V                # noqa: E402
from nfat.swk_ablate_sweep import STD             # noqa: E402

LEAD = {'sniah512': 'rethead', 'sniah256': 'rethead', 'ruler': 'rethead', 'bab': 'cake'}


def run_eval(st, ly):
    c = KS.S[st]; ev, W = c['ev'], c['W']; tag = f'pr-{st}-{KS.lname(ly)}'
    env = dict(KS.ENV, NFAT_SWA_WINDOW=str(W), NFAT_SWA_SINK='4', NFAT_FULL_LAYERS=ly)
    if ev == 'sniah4k':
        KS.sh([KS.PN, 'nfat/eval_ruler.py', '--model_config', KS.M8, '--tasks', 'S-NIAH-PK,S-NIAH-N,S-NIAH-W', '--lengths', '4096',
               '--n', '50', '--tag', tag], env=env)
    elif ev == 'ruler8k':
        KS.sh([KS.PS, 'nfat/run_ruler_vllm.py', '--model', KS.L8B, '--n', '25', '--name', tag, '--W', '512', '--keep_first', '4',
               '--layers', ly, '--lengths', '8192'])
    elif ev == 'bab24':
        KS.sh([KS.PN, 'nfat/eval_babilong.py', '--model_config', KS.M8, '--tasks', 'qa1,qa2,qa3,qa4,qa5', '--lengths', '2k,4k', '--n', '50',
               '--tag', tag], env=env)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--only', default='')
    a = p.parse_args()
    d = pd.read_csv(KS.OUT, dtype={'layers': str})
    for st in [x for x in a.only.split(',') if x]:
        for k in STD[st]:
            for m in ['swapkl', LEAD[st]]:
                r = d[(d.setting == st) & (d.method == m) & (d.k == k)]
                if r.empty:
                    continue
                ly = KS.norm(r.layers.iloc[0])
                V._DET_CACHE.clear()
                if V.per_sample(st, ly) is not None:
                    continue
                run_eval(st, ly)
                V._DET_CACHE.clear()
                print(f"== pr {st} {m} k={k} [{ly}] -> {'OK' if V.per_sample(st, ly) is not None else '!! KHÔNG có chi tiết'}", flush=True)
    print('XONG paired_reeval', flush=True)
