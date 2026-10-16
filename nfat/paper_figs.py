"""
Hình cho paper ACML (tiếng Anh, 6 baseline: Duo, RH, CAKE, MoA, BI, NLL-guided + SwapKL; lưới k chuẩn).
  figures/paper_kcurves.pdf    — độ chính xác theo k, 8 benchmark (2×4)
  figures/paper_totaltime.pdf  — độ chính xác theo tổng thời gian (chọn tầng + 1000 yêu cầu 32K), 8 benchmark (2×4)
    python nfat/paper_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from nfat.ksweep_plot import REF, costs                                  # noqa: E402
from nfat.swapkl_report_plot import STD_GRID, SPMODEL, speed_table    # noqa: E402
from nfat.paper_time import GEN                                          # noqa: E402

ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b', 'lme_l8i', 'lme_q4t']
TITLE = {'niah1b': 'NIAH, Llama-3.2-1B', 'sniah512': 'S-NIAH 4K, Llama-3.1-8B, W=512', 'sniah256': 'S-NIAH 4K, Llama-3.1-8B, W=256',
         'ruler': 'RULER 8K, Llama-3.1-8B', 'bab': 'BABILong 2K/4K, Llama-3.1-8B', 'niah8b': 'NIAH, Llama-3.1-8B',
         'lme_l8i': 'LongMemEval, Llama-3.1-8B-Instr.', 'lme_q4t': 'LongMemEval, Qwen3-4B-Think.'}
STYLE = {'swapkl': ('SwapKL (ours)', '#2a78d6', 'o'), 'nll_swa': ('NLL-guided', '#e34948', '*'), 'duo': ('DuoAttention', '#1baf7a', '^'),
         'rethead': ('Retrieval Head', '#eda100', 'D'), 'cake': ('CAKE', '#e87ba4', 'v'), 'moa': ('MoA', '#008300', 'P'),
         'bi': ('BI', '#4a3aa7', 'X')}


def style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                         'axes.labelcolor': '#3a3a38', 'xtick.color': '#5a5a57', 'ytick.color': '#5a5a57', 'pdf.fonttype': 42})


def fig(mode, order=ORDER, name=None):
    d = pd.read_csv('results/perbench/ksweep.csv')
    d = d[[k in STD_GRID[s] for s, k in zip(d.setting, d.k)]]
    sp = speed_table(32768)
    one = len(order) == 4
    fig, axs = plt.subplots(1, 4, figsize=(10.0, 2.3)) if one else plt.subplots(2, 4, figsize=(10.0, 4.9))
    axs = np.atleast_2d(axs)
    for ax, st in zip(axs.flat, order):
        g = d[d.setting == st]; L, swa, full = REF[st]; cs = costs(st)
        ax.axhline(full, color='#b9b8b0', lw=0.9, ls='--')
        for m, (lab, col, mk) in STYLE.items():
            h = g[g.method == m].sort_values('k')
            if h.empty:
                continue
            main = m == 'swapkl'
            if mode == 'k':
                x, y = h.k.values, h.score.values
            else:
                G = GEN.get(st, 256); gs = sp[sp.model == SPMODEL[st]].set_index('k')      # độ dài sinh đo thật cho Qwen (paper_time.GEN)
                pts = [(cs.get((m, int(k)), np.nan) + 1000 * (gs.prefill_s[k] + G * gs.tpot_ms[k] / 1000) / 4 / 60 if k in gs.index else np.nan, s)
                       for k, s in zip(h.k, h.score)]
                pts = [(a, b) for a, b in pts if np.isfinite(a)]
                if not pts:
                    continue
                x, y = zip(*pts)
            ax.plot(x, y, color=col, lw=2.0 if main else 1.0, marker=mk, ms=4.5 if main else 3.5, label=lab, zorder=5 if main else 3,
                    alpha=1 if main else .85)
        ax.set_title(TITLE[st], fontsize=9, loc='left')
        ax.grid(axis='y', color='#ececE8', lw=0.6)
        ax.set_xlabel('k (full-attention layers)' if mode == 'k' else 'total time (min)', fontsize=9)
    for r in range(axs.shape[0]):
        axs[r, 0].set_ylabel('accuracy')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.11 if one else 0.05, 1, 1))
    name = name or ('paper_kcurves' if mode == 'k' else 'paper_totaltime')
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    style()
    fig('k'); fig('t')
    fig('k', ['sniah512', 'ruler', 'niah8b', 'lme_q4t'], 'paper_kcurves4')
    print('ok')
