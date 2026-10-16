"""
Mục tiêu J có xếp đúng các tập theo độ chính xác test không? (phản biện OE-Agent 10-16, W1)
Với mỗi ô (benchmark, k): Spearman giữa J(·;B) (giữ riêng, không dùng khi tìm kiếm) — và NLL đáp án trên B — với điểm test, trên mọi tập
khác nhau đã đánh giá test (6 baseline + NLL-guided + SwapKL + biến thể ablation). Ghi results/paper/jcorr.csv, in tóm tắt theo benchmark.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
norm = lambda x: ','.join(str(v) for v in sorted(int(u) for u in str(x).split(',')))   # noqa: E731


def main():
    j = pd.read_csv('results/swk_ablate/jeval.csv', dtype={'layers': str}).drop_duplicates(['setting', 'k', 'layers'], keep='last')
    ks = pd.read_csv('results/perbench/ksweep.csv', dtype={'layers': str})
    ab = pd.read_csv('results/swk_ablate/ksweep_ablate.csv', dtype={'layers': str})
    sc = pd.concat([ks[['setting', 'k', 'layers', 'score']], ab[['setting', 'k', 'layers', 'score']]])
    sc['layers'] = sc.layers.map(norm); sc = sc.drop_duplicates(['setting', 'k', 'layers'])
    m = j.merge(sc, on=['setting', 'k', 'layers'])
    rows = []
    for (st, k), g in m.groupby(['setting', 'k']):
        if len(g) < 4 or g.score.nunique() < 2:
            continue
        rows.append(dict(setting=st, k=k, n=len(g), rho_JB=spearmanr(-g.J_B, g.score)[0], rho_JA=spearmanr(-g.J_A, g.score)[0],
                         rho_NLLB=spearmanr(-g.NLL_B, g.score)[0],
                         top1_JB=float(g.loc[g.J_B.idxmin(), 'score'] >= g.score.max() - 1e-9)))
    R = pd.DataFrame(rows); os.makedirs('results/paper', exist_ok=True); R.to_csv('results/paper/jcorr.csv', index=False)
    print(R.groupby('setting')[['n', 'rho_JB', 'rho_JA', 'rho_NLLB', 'top1_JB']].agg(['median', 'count']).round(2).to_string())
    print('ALL median rho_JB %.2f rho_NLLB %.2f, cells %d' % (R.rho_JB.median(), R.rho_NLLB.median(), len(R)))
    # gộp: xếp hạng trong ô (khử khác biệt theo k)
    m['r_score'] = m.groupby(['setting', 'k']).score.rank(pct=True); m['r_J'] = m.groupby(['setting', 'k']).J_B.rank(pct=True)
    print('pooled within-cell rank corr (J_B vs acc): %.2f' % spearmanr(-m.r_J, m.r_score)[0])


if __name__ == '__main__':
    main()
