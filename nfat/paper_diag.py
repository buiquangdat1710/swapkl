"""
Chẩn đoán SwapKL cho paper (phản biện OE-Agent 10-16), chỉ đọc log có sẵn:
  - cuộc đua khởi đầu: số ô mỗi khởi đầu thắng (chia đều khi nhiều khởi đầu trùng tập), số hoán đổi được nhận, B chọn tập nào;
  - tính lồng nhau theo k của tập SwapKL (baseline top-k luôn lồng nhau);
  - hình bản đồ tầng được chọn theo k.
    python nfat/paper_diag.py -> results/paper/t_race.tex, figures/paper_layers.pdf
"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import sys                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
SET = {'niah-1b': 'niah1b', 'sniah-w512': 'sniah512', 'sniah-w256': 'sniah256', 'ruler-w512': 'ruler', 'bab2-w256': 'bab', 'niah-8b': 'niah8b'}
ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']
NAME = {'niah1b': 'NIAH-1B', 'sniah512': 'S-NIAH$_{512}$', 'sniah256': 'S-NIAH$_{256}$', 'ruler': 'RULER', 'bab': 'BABILong',
        'niah8b': 'NIAH-8B', 'lme_l8i': 'LME-Llama', 'lme_q4t': 'LME-Qwen'}
STD = {'niah1b': [1, 2, 3, 4, 6, 8, 12], 'lme_l8i': [4, 8, 16], 'lme_q4t': [4, 9, 18]}
STD.update({s: [1, 2, 4, 6, 8, 12, 16, 24] for s in ['sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']})
STARTS = [('copy', 'copy'), ('cake', 'CAKE'), ('kl', 'KL-grad'), ('mse', 'MSE-grad'), ('copy+cake', 'copy$\\cup$CAKE')]


def race():
    win = collections.defaultdict(collections.Counter); acc = collections.defaultdict(collections.Counter)
    runner = collections.Counter(); ncell = collections.Counter()
    for f in sorted(glob.glob('results/swk_ablate/traces/pb-*.json')):
        if f.endswith('_t3.json'):
            continue
        t = json.load(open(f)); st = [v for kk, v in SET.items() if kk in f][0]; k = t['k']
        if k not in STD[st]:
            continue
        tr = t['variants']['v11']['trace']; ncell[st] += 1
        S = tuple(sorted(tr['start'])); names = [n for n, T in tr['starts'].items() if tuple(sorted(T)) == S]
        for n in names:
            win[st][n] += 1 / len(names)
        a = sum(r['improve'] for r in tr['rounds']); acc[st][a] += 1
        cur = tr['start']
        for r in tr['rounds']:
            if r['improve']:
                cur = r['best']
        runner[st] += sorted(tr['out']) != sorted(cur)
    return win, acc, runner, ncell


def nested():
    d = pd.read_csv('results/perbench/ksweep.csv', dtype={'layers': str})
    out = {}
    for st in STD:
        g = d[(d.setting == st) & (d.method == 'swapkl') & d.k.isin(STD[st])].sort_values('k')
        sets = [set(map(int, x.split(','))) for x in g.layers]
        out[st] = (sum(not a <= b for a, b in zip(sets, sets[1:])), len(sets) - 1)
    return out, d


def fmt(x):
    return ('%.1f' % x).rstrip('0').rstrip('.') if x else '0'


def main():
    os.makedirs('results/paper', exist_ok=True)
    win, acc, runner, ncell = race(); nest, d = nested()
    L = ['\\begin{tabular}{@{}l|ccccc|cccc|c|c@{}}', '\\toprule',
         ' & \\multicolumn{5}{c|}{start-race wins (cells)} & \\multicolumn{4}{c|}{accepted swaps} & B picks & non-nested \\\\',
         'Benchmark & ' + ' & '.join(l for _, l in STARTS) + ' & 0 & 1 & 2 & 3 & runner-up & $k$ pairs \\\\', '\\midrule']
    tot = collections.Counter(); ta = collections.Counter(); tn = [0, 0]
    for st in ORDER:
        L.append(f'{NAME[st]} & ' + ' & '.join(fmt(win[st][n]) for n, _ in STARTS) + ' & ' +
                 ' & '.join(str(acc[st][i]) for i in range(4)) + f' & {runner[st]}/{ncell[st]} & {nest[st][0]}/{nest[st][1]} \\\\')
        tot.update(win[st]); ta.update(acc[st]); tn[0] += nest[st][0]; tn[1] += nest[st][1]
    L += ['\\midrule', 'Total & ' + ' & '.join(fmt(tot[n]) for n, _ in STARTS) + ' & ' + ' & '.join(str(ta[i]) for i in range(4)) +
          f' & {sum(runner.values())}/{sum(ncell.values())} & {tn[0]}/{tn[1]} \\\\', '\\bottomrule', '\\end{tabular}']
    open('results/paper/t_race.tex', 'w').write('\n'.join(L))
    print('\n'.join(L)); print({s: nest[s] for s in STD})
    # bản đồ tầng
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    sts = ORDER + ['lme_l8i', 'lme_q4t']
    fig, axs = plt.subplots(2, 4, figsize=(10.0, 3.6))
    for ax, st in zip(axs.flat, sts):
        g = d[(d.setting == st) & (d.method == 'swapkl') & d.k.isin(STD[st])].sort_values('k')
        Lr = 16 if st == 'niah1b' else (36 if st == 'lme_q4t' else 32)
        M = np.zeros((len(g), Lr))
        for i, ly in enumerate(g.layers):
            M[i, [int(x) for x in ly.split(',')]] = 1
        ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=1.3, interpolation='nearest')
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.k.tolist(), fontsize=6)
        ax.set_xticks(range(0, Lr, 4)); ax.tick_params(axis='x', labelsize=6)
        ax.set_title(NAME[st].replace('$_{512}$', ' W=512').replace('$_{256}$', ' W=256') + f'  ({nest[st][0]}/{nest[st][1]} non-nested)',
                     fontsize=7.5, loc='left')
        ax.set_xlabel('layer', fontsize=7)
    axs[0, 0].set_ylabel('$k$'); axs[1, 0].set_ylabel('$k$')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/paper_layers.{ext}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()


def oracle():
    """Cận trên: mỗi k chọn baseline tốt nhất TRÊN TEST (oracle best-of-6), so với baseline đơn tốt nhất và SwapKL."""
    from nfat.swapkl_report_plot import STD_GRID
    d = pd.read_csv('results/perbench/ksweep.csv'); d = d[[k in STD_GRID[s] for s, k in zip(d.setting, d.k)]]
    BL = ['duo', 'rethead', 'cake', 'moa', 'bi', 'nll_swa']
    L = ['\\begin{tabular}{@{}l|cc|c@{}}', '\\toprule', 'Setting & best baseline & oracle best-of-6 per $k$ & \\swk{} \\\\', '\\midrule']
    for st in ORDER + ['lme_l8i', 'lme_q4t']:
        p = d[d.setting == st].pivot(index='k', columns='method', values='score')
        L.append(f"{NAME[st]} & {p[BL].mean().max():.3f} & {p[BL].max(axis=1).mean():.3f} & {p['swapkl'].mean():.3f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open('results/paper/t_oracle.tex', 'w').write('\n'.join(L)); print('\n'.join(L))


if __name__ == '__main__':
    oracle()
