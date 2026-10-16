"""
Driver ablation SwapKL trên LƯỚI CHUẨN (nfat/swk_ablate.py). Mỗi benchmark: chạy MỌI lượt chọn tầng (một lượt / k, mọi biến thể) trước,
rồi đánh giá các tập MỚI theo thứ tự ưu tiên (tầng 1 trước, cùng giao thức quét k; tập trùng tập đã đánh giá -> lấy lại điểm).
    python nfat/swk_ablate_sweep.py run --only niah1b,sniah512 [--tier 2] [--phase select|eval|both]
-> results/swk_ablate/ksweep_ablate.csv (setting, k, variant, layers, score, eval_tag, same_as_v11)
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from nfat import ksweep as KS                     # noqa: E402
from nfat.swk_ablate import TIER1, TIER2, TIER3, TIER4   # noqa: E402

OUT = 'results/swk_ablate/ksweep_ablate.csv'
STD = {'niah1b': [1, 2, 3, 4, 6, 8, 12], 'lme_l8i': [4, 8, 16], 'lme_q4t': [4, 9, 18]}
STD.update({s: [1, 2, 4, 6, 8, 12, 16, 24] for s in ['sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']})


def tag_for(st, k):
    c = KS.S[st]
    return c['ntag'].get(k, f"{c['ntag'][min(c['ntag'])]}-k{k}")


def select(st, tier):
    c = KS.S[st]; nll = min(c['ntag'].items())[1]
    for k in STD[st]:
        tag = tag_for(st, k)
        if tier >= 3:
            if KS.has_scorer(tag, f'swkab{tier}'):
                continue
            KS.sh([KS.PS, 'nfat/perbench.py', '--model', c['model'], '--bench', c['bench'], '--W', str(c['W']), '--S', str(c['Sk']),
                   '--k', str(k), '--decode', c['dec'], '--scorers', f'swkab{tier}', '--tag', tag, '--extra', f'tier={tier},setting={st}'])
            continue
        want = 'swkab_only_rand' if tier >= 2 else 'swkab_r1'
        if KS.has_scorer(tag, 'swkab') and KS.has_scorer(tag, want):
            continue
        KS.sh([KS.PS, 'nfat/perbench.py', '--model', c['model'], '--bench', c['bench'], '--W', str(c['W']), '--S', str(c['Sk']),
               '--k', str(k), '--decode', c['dec'], '--scorers', 'swkab', '--tag', tag, '--extra', f'tier={tier},nllg={nll}'])


def evaluate(st, tier):
    ks = pd.read_csv(KS.OUT, dtype={'layers': str})
    done = pd.read_csv(OUT, dtype={'layers': str}) if os.path.exists(OUT) else pd.DataFrame(columns=['setting', 'k', 'variant'])
    variants = {3: TIER3, 4: TIER4}.get(tier) or TIER1 + (TIER2 if tier >= 2 else [])
    for v in variants:                                       # ưu tiên theo biến thể (tầng 1 trước), mọi k
        sc = 'swkab' if v == 'v11' else f'swkab_{v}'
        for k in STD[st]:
            if ((done.setting == st) & (done.k == k) & (done.variant == v)).any():
                continue
            ly = KS.pick(tag_for(st, k), sc, k)
            if not ly:
                print(f'!! không có tập {st}/{v}/k{k}', flush=True); continue
            ly = KS.norm(ly)
            got = KS.evaluate(st, ly)
            if not got:
                print(f'!! đánh giá lỗi {st}/{v}/k{k} [{ly}]', flush=True); continue
            r11 = ks[(ks.setting == st) & (ks.method == 'swapkl') & (ks.k == k)]
            same = bool(len(r11)) and KS.norm(r11.layers.iloc[0]) == ly
            row = dict(setting=st, k=k, variant=v, layers=ly, score=got[0], eval_tag=got[1], same_as_v11=same)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(OUT, index=False)
            print(f'== ab {st} {v} k={k} [{ly}] -> {got[0]:.4f}{" (= v1.1)" if same else ""}', flush=True)
            if v == 'v11' and not same:
                print(f'!! CẢNH BÁO: v11 khác tập swapkl ở {st} k{k}', flush=True)
        print(f'XONG ab {st} {v}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['run']); p.add_argument('--only', default='')
    p.add_argument('--tier', type=int, default=2); p.add_argument('--phase', default='both')
    a = p.parse_args()
    for st in [x for x in a.only.split(',') if x]:
        if a.phase in ('select', 'both'):
            select(st, a.tier)
        if a.phase in ('eval', 'both'):
            evaluate(st, a.tier)
    print('XONG swk_ablate_sweep', flush=True)
