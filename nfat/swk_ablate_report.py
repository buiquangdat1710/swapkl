"""
Bảng ablation (tiếng Anh, cho paper ACML) từ results/swk_ablate/ksweep_ablate.csv + cost.csv.
Chỉ số mỗi (biến thể, benchmark): điểm TB trên lưới k chuẩn (chỉ khi đủ mọi k), Δ so với v11 (= SwapKL v1.1), số ô k tốt hơn / kém hơn
v11, và chi phí chọn tầng nếu chạy riêng (tổng lượt xuôi-mẫu của biến thể / của v11, trên các ô đã chạy).
    python nfat/swk_ablate_report.py   ->  results/swk_ablate/tex/*.tex, results/swk_ablate/summary.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'results/swk_ablate/tex'
ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']
NAME = {'niah1b': 'NIAH-1B', 'sniah512': 'S-NIAH$_{512}$', 'sniah256': 'S-NIAH$_{256}$', 'ruler': 'RULER', 'bab': 'BABILong',
        'niah8b': 'NIAH-8B'}
STD = {'niah1b': [1, 2, 3, 4, 6, 8, 12]}
STD.update({s: [1, 2, 4, 6, 8, 12, 16, 24] for s in ORDER[1:]})
GROUPS = [
    ('Starting sets: leave one out', [('loo_copy', 'w/o copy'), ('loo_cake', 'w/o CAKE'), ('loo_kl', 'w/o KL-grad'),
                                      ('loo_mse', 'w/o MSE-grad'), ('loo_ilv', 'w/o copy$\\cup$CAKE')]),
    ('Starting sets: single start', [('only_copy', 'copy only'), ('only_cake', 'CAKE only'), ('only_kl', 'KL-grad only'),
                                     ('only_mse', 'MSE-grad only'), ('only_ilv', 'copy$\\cup$CAKE only')]),
    ('Starting sets: other sources', [('only_rand', 'random only'), ('only_per', 'periodic only'), ('only_nllg', 'NLL-guided only')]),
    ('Calibration split', [('noB', 'no hold-out B (A only)'), ('merged', 'single 32-sample set')]),
    ('Search', [('nosearch', 'no swaps (best start)'), ('r1', '1 swap round'), ('randrm', 'random swap-out'),
                ('noconfirm', 'no confirmation'), ('r5', 'up to 5 swap rounds')]),
    ('Objective', [('nllobj', 'NLL objective (same search)')]),
]
# bảng gọn cho thân bài (bảng đầy đủ ở phụ lục)
GROUPS_MAIN = [
    ('Search (Part 2)', [('nosearch', 'no swaps (best start)'), ('r1', '1 swap round'), ('randrm', 'random swap-out')]),
    ('Objective', [('nllobj', 'NLL objective (same search)')]),
    ('Starting sets (Part 1)', [('only_cake', 'worst single start (CAKE)'), ('only_rand', 'random start'), ('only_per', 'periodic start'),
                                ('only_nllg', 'NLL-guided start')]),
    ('Calibration split', [('noB', 'no hold-out B (A only)')]),
]
BL6 = ['duo', 'rethead', 'cake', 'moa', 'bi', 'nll_swa']


def load():
    d = pd.read_csv('results/swk_ablate/ksweep_ablate.csv', dtype={'layers': str})
    c = pd.read_csv('results/swk_ablate/cost.csv') if os.path.exists('results/swk_ablate/cost.csv') else pd.DataFrame()
    return d, c


def best_by_j():
    """'bestJ': trong 6 baseline, tập có J (KL khớp giải mã) nhỏ nhất trên A ở mỗi (benchmark, k); điểm test = điểm đã có của tập đó."""
    p = 'results/swk_ablate/jeval.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    j = pd.read_csv(p, dtype={'layers': str}).drop_duplicates(['setting', 'k', 'layers'], keep='last')
    ks = pd.read_csv('results/perbench/ksweep.csv', dtype={'layers': str})
    norm = lambda x: ','.join(str(v) for v in sorted(int(u) for u in str(x).split(',')))   # noqa: E731
    ks['layers'] = ks.layers.map(norm)
    rows = []
    for (st, k), g in j.groupby(['setting', 'k']):
        b = ks[(ks.setting == st) & (ks.k == k) & ks.method.isin(BL6)]
        if b.method.nunique() < len(BL6):
            continue
        g = g[g.layers.isin(set(b.layers))]
        if g.empty:
            continue
        best = g.sort_values('J_A').iloc[0]
        rows.append(dict(setting=st, k=k, variant='bestJ', layers=best.layers, score=float(b[b.layers == best.layers].score.iloc[0])))
    return pd.DataFrame(rows)


def summary(d, c):
    rows = []
    for st in ORDER:
        g = d[d.setting == st]
        base = g[g.variant == 'v11'].set_index('k').score
        for v in g.variant.unique():
            h = g[g.variant == v].set_index('k').score
            ks = [k for k in STD[st] if k in h.index and k in base.index]
            full = len(ks) == len(STD[st])
            diff = (h[ks] - base[ks])
            r = dict(setting=st, variant=v, n_k=len(ks), complete=full, kmean=float(h[ks].mean()) if ks else np.nan,
                     kmean_v11=float(base[ks].mean()) if ks else np.nan, delta=float(diff.mean()) if ks else np.nan,
                     better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum()))
            if len(c):
                cc = c[c.tag.notna()]
                rr = cc[cc.variant == v].merge(cc[cc.variant == 'v11'][['tag', 'k', 'n_req']], on=['tag', 'k'], suffixes=('', '_v11'))
                rr = rr[rr.tag.str.contains({'niah1b': 'niah-1b', 'sniah512': 'sniah-w512', 'sniah256': 'sniah-w256', 'ruler': 'ruler-w512',
                                             'bab': 'bab2-w256', 'niah8b': 'niah-8b'}[st], regex=False)]
                r['cost_ratio'] = float(rr.n_req.sum() / rr.n_req_v11.sum()) if len(rr) and rr.n_req_v11.sum() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def f3(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def fd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    x = round(float(x), 3) + 0.0
    s = f'{x:+.3f}'
    return '$\\mathbf{' + s + '}$' if x < -0.0095 else f'${s}$'


def table(S, groups=GROUPS, fname='t_ablate.tex'):
    os.makedirs(OUT, exist_ok=True)
    have = [st for st in ORDER if ((S.setting == st) & (S.variant == 'v11') & S.complete).any()]
    lines = ['\\begin{tabular}{@{}l' + 'c' * len(have) + '|cccc@{}}', '\\toprule',
             'Variant & ' + ' & '.join(NAME[s] for s in have) + ' & mean $\\Delta$ & worst $\\Delta$ & cells $-$/$+$ & cost \\\\', '\\midrule']
    base = S[S.variant == 'v11'].set_index('setting')
    lines.append('SwapKL (full) & ' + ' & '.join(f3(base.loc[s, 'kmean']) for s in have) + ' & -- & -- & -- & 1.00 \\\\')
    for title, vs in groups:
        if not any((S.variant == v).any() for v, _ in vs):
            continue
        lines.append('\\midrule\\multicolumn{' + str(len(have) + 5) + '}{@{}l}{\\emph{' + title + '}} \\\\')
        for v, lab in vs:
            if not (S.variant == v).any():
                continue
            x = S[S.variant == v].set_index('setting')
            cells, ds = [], []
            for s in have:
                if s in x.index and x.loc[s, 'complete']:
                    cells.append(fd(x.loc[s, 'delta'])); ds.append(x.loc[s, 'delta'])
                else:
                    cells.append('--')
            cr = x.cost_ratio.mean() if 'cost_ratio' in x and len(x) else np.nan
            if v == 'bestJ':
                cr = 6 * 16 / 952                     # 6 tập x 16 mẫu A (chưa kể chạy 6 baseline)
            ok = len(ds) == len(have) and ds
            md = fd(float(np.mean(ds))) if ok else '--'
            wd = fd(float(np.min(ds))) if ok else '--'
            cw = f"{int(x.loc[have, 'worse'].sum())}/{int(x.loc[have, 'better'].sum())}" if ok else '--'
            lines.append(f'{lab} & ' + ' & '.join(cells) + f' & {md} & {wd} & {cw} & ' + ('--' if np.isnan(cr) else f'{cr:.2f}') + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(f'{OUT}/{fname}', 'w').write('\n'.join(lines))
    return have


def main():
    d, c = load()
    S = summary(d, c)
    S.to_csv('results/swk_ablate/summary.csv', index=False)
    have = table(S)
    table(S, GROUPS_MAIN, 't_ablate_main.tex')
    pd.set_option('display.width', 200)
    print(S.round(3).to_string())
    print('benchmarks complete for v11:', have)


if __name__ == '__main__':
    main()
