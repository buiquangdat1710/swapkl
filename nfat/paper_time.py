"""
Phân tích thời gian cho paper ACML (phản biện OE-Agent 10-16):
  - k nhỏ nhất trên lưới chuẩn giữ ≥ 90/95/99 % điểm mô hình gốc: SwapKL, baseline tốt nhất (6 baseline không SLEB), SwapKL không hoán đổi
    (biến thể ablation 'nosearch', chỉ 6 benchmark SWA);
  - số yêu cầu hoà vốn N* (32K token + 256 sinh, lô 4, một L40S) của SwapKL so với baseline tốt nhất ở mức 95 % và so với mô hình gốc;
  - hình tổng thời gian theo N (log-log).
    python nfat/paper_time.py  ->  results/paper/t_breakeven.tex, results/paper/breakeven.csv, figures/paper_tn.pdf
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
from nfat.ksweep_plot import REF, costs                                   # noqa: E402
from nfat.swapkl_report_plot import STD_GRID, SPMODEL, speed_table         # noqa: E402

ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b', 'lme_l8i', 'lme_q4t']
NAME = {'niah1b': 'NIAH-1B', 'sniah512': 'S-NIAH$_{512}$', 'sniah256': 'S-NIAH$_{256}$', 'ruler': 'RULER', 'bab': 'BABILong',
        'niah8b': 'NIAH-8B', 'lme_l8i': 'LME-Llama', 'lme_q4t': 'LME-Qwen'}
TITLE = {'niah1b': 'NIAH, Llama-3.2-1B', 'sniah512': 'S-NIAH, W=512', 'sniah256': 'S-NIAH, W=256', 'ruler': 'RULER 8K',
         'bab': 'BABILong 2K/4K', 'niah8b': 'NIAH, Llama-3.1-8B', 'lme_l8i': 'LongMemEval, Llama-3.1-8B-I', 'lme_q4t': 'LongMemEval, Qwen3-4B-T'}
BL = ['duo', 'rethead', 'cake', 'moa', 'bi', 'nll_swa']
LAB = {'duo': 'Duo', 'rethead': 'RH', 'cake': 'CAKE', 'moa': 'MoA', 'bi': 'BI', 'nll_swa': 'NLLG', 'swapkl': 'SwapKL'}
THR = (0.90, 0.95, 0.99)
# số token sinh trung bình đo thật (phản biện vòng 2, M1): Qwen3-4B-Thinking suy luận ~1,26K token/câu (TB 52 lượt chạy LME, gồm phần
# <think>); các thiết lập khác giữ 256 token của phép đo tốc độ (câu trả lời thật ngắn hơn -> bảo thủ).
GEN = {'lme_q4t': 1260}


def kmin(scores, full, thr):
    """scores: {k: acc}; k nhỏ nhất đạt thr·full, None nếu không."""
    return next((k for k in sorted(scores) if scores[k] >= thr * full - 1e-12), None)


def main():
    d = pd.read_csv('results/perbench/ksweep.csv')
    d = d[[k in STD_GRID[s] for s, k in zip(d.setting, d.k)]]
    ab = pd.read_csv('results/swk_ablate/ksweep_ablate.csv')
    sp = speed_table(32768)
    rows, lines = [], []
    for st in ORDER:
        L, swa, full = REF[st]; cs = costs(st); g = sp[sp.model == SPMODEL[st]]
        G = GEN.get(st, 256)                                                          # giây / yêu cầu (lô 4): prefill + G bước giải mã
        treq = {int(k): (float(p) + G * float(t) / 1000) / 4 for k, p, t in zip(g.k, g.prefill_s, g.tpot_ms)}
        tfull = (float(g.full_prefill_s.iloc[0]) + G * float(g.full_tpot_ms.iloc[0]) / 1000) / 4
        sc = {m: dict(zip(h.k, h.score)) for m, h in d[d.setting == st].groupby('method')}
        r = dict(setting=st)
        for thr in THR:
            ks = kmin(sc['swapkl'], full, thr)
            kb = {m: kmin(sc[m], full, thr) for m in BL if m in sc}
            kb = {m: v for m, v in kb.items() if v is not None}
            best = min(kb, key=lambda m: (kb[m], cs.get((m, kb[m]), np.inf))) if kb else None
            ns = dict(zip(ab[(ab.setting == st) & (ab.variant == 'nosearch')].k, ab[(ab.setting == st) & (ab.variant == 'nosearch')].score))
            r[f'swk{int(thr * 100)}'] = ks
            r[f'bl{int(thr * 100)}'] = kb.get(best) if best else None
            r[f'blm{int(thr * 100)}'] = best
            r[f'ns{int(thr * 100)}'] = kmin(ns, full, thr) if len(ns) == len(STD_GRID[st]) else None
        ks, kb, best = r['swk95'], r['bl95'], r['blm95']
        sel_s = cs.get(('swapkl', ks), np.nan) * 60 if ks else np.nan
        sel_b = cs.get((best, kb), np.nan) * 60 if best else np.nan
        # hoà vốn so với baseline tốt nhất
        if ks is None:
            nb = 'never'
        elif best is None:
            nb = 0.0
        elif ks < kb:
            nb = max(0.0, (sel_s - sel_b) / (treq[kb] - treq[ks]))
        elif ks == kb:
            nb = 0.0 if sel_s <= sel_b else 'never'
        else:
            nb = 'never'
        no = sel_s / (tfull - treq[ks]) if ks else np.nan
        r.update(gen_tokens=G, speedup_op=tfull / treq[ks] if ks else np.nan, sel_swk_min=sel_s / 60, sel_bl_min=sel_b / 60, treq_swk=treq.get(ks), treq_bl=treq.get(kb), treq_full=tfull,
                 N_vs_bl=nb, N_vs_orig=no)
        rows.append(r)

        def fk(k, m=None):
            return '\\na' if k is None else (f'{k}' + (f' {LAB[m]}' if m else ''))

        def fn(x):
            if isinstance(x, str):
                return '$\\infty$'
            return '0' if x == 0 else (f'{x:,.0f}'.replace(',', '{,}') if x >= 100 else f'{x:.0f}')
        lines.append(f'{NAME[st]} & ' + ' & '.join(f"{fk(r[f'swk{t}'])} / {fk(r[f'bl{t}'])}" for t in (90, 95, 99)) +
                     f" & {LAB[best] if best else chr(92) + 'na'} & {'n/a' if st.startswith('lme') else fk(r['ns95'])} & {fn(nb)} & {fn(no)} \\\\")
    pd.DataFrame(rows).to_csv('results/paper/breakeven.csv', index=False)
    tex = ['\\begin{tabular}{@{}l|ccc|c|c|cc@{}}', '\\toprule',
           ' & \\multicolumn{3}{c|}{smallest $k$: \\swk{} / best baseline} & best baseline & \\swk{} w/o & \\multicolumn{2}{c}{break-even $N^\\star$ vs.} \\\\',
           'Benchmark & $90\\,\\%$ & $95\\,\\%$ & $99\\,\\%$ & at $95\\,\\%$ & swaps, $95\\,\\%$ & best baseline & original \\\\', '\\midrule'] + lines + \
          ['\\bottomrule', '\\end{tabular}']
    open('results/paper/t_breakeven.tex', 'w').write('\n'.join(tex))
    print(pd.DataFrame(rows).to_string())
    fig_tn(d, sp, rows)


def fig_tn(d, sp, rows):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(2, 4, figsize=(10.0, 4.6))
    N = np.logspace(1, 5, 200)
    for ax, r in zip(axs.flat, rows):
        st = r['setting']
        if r['swk95'] is not None:
            ax.plot(N, r['sel_swk_min'] + N * r['treq_swk'] / 60, color='#2a78d6', lw=2, label=f"SwapKL ($k$@95%)")
        if r['blm95'] is not None:
            ax.plot(N, r['sel_bl_min'] + N * r['treq_bl'] / 60, color='#e34948', lw=1.2, label='best baseline ($k$@95%)')
        ax.plot(N, N * r['treq_full'] / 60, color='#8a8a85', lw=1, ls='--', label='original')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(TITLE[st] + (f"  (k {r['swk95']} vs {r['bl95']})" if r['blm95'] else ''), fontsize=8, loc='left')
        ax.set_xlabel('requests $N$ (32K tokens)', fontsize=8)
    axs[0, 0].set_ylabel('total time (min)'); axs[1, 0].set_ylabel('total time (min)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/paper_tn.{ext}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()


def speed128():
    """Bảng tốc độ ở ~128K (tăng tốc so với mô hình gốc; prefill, giải mã, tổng) -> results/paper/t_speed128.tex."""
    from nfat.swapkl_report_tables import long_speed
    t = long_speed()
    t = t[t.model != 'l8b256']
    L = ['\\begin{tabular}{@{}lr|rrr@{}}', '\\toprule', 'Model (context) & $k$ & prefill $\\times$ & decode $\\times$ & total $\\times$ \\\\', '\\midrule']
    for i, (key, g) in enumerate(t.groupby('model', sort=False)):
        if i:
            L.append('\\midrule')
        lab = g.label.iloc[0] + f" ({int(g.Lctx.iloc[0]) // 1024}K)"
        for j, (_, r) in enumerate(g.sort_values('k').iterrows()):
            L.append(f"{lab if j == 0 else ''} & {int(r.k)}{' (SWA)' if r.k == 0 else ''} & {r.sp_pre:.2f} & {r.sp_dec:.2f} & {r.sp_tot:.2f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open('results/paper/t_speed128.tex', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__' and os.environ.get('SPEED128', '1') == '1':
    speed128()


def op_macros():
    """Tăng tốc tại điểm 95 % (số token sinh như GEN): TB 8 thiết lập ở 32K; TB các thiết lập có đo ~128K. -> results/paper/macros_time.tex"""
    from nfat.swapkl_report_tables import long_speed
    b = pd.read_csv('results/paper/breakeven.csv')
    lt = long_speed()
    sp128 = []
    for _, r in b.iterrows():
        st, k = r.setting, r.swk95
        g = lt[(lt.model == SPMODEL[st]) & (lt.k == k)] if k == k else pd.DataFrame()
        if st in ('sniah256', 'bab') or g.empty:          # W=256 (S-NIAH256, BABILong) và 1B chưa đo ở 128K
            continue
        G = GEN.get(st, 256); g = g.iloc[0]
        sp128.append((st, (g.full_prefill_s + G * g.full_tpot_ms / 1000) / (g.prefill_s + G * g.tpot_ms / 1000)))
    m32 = b.speedup_op.mean(); m128 = np.mean([v for _, v in sp128])
    q = b.set_index('setting')
    L = [f'\\newcommand{{\\PSpTot}}{{{m32:.2f}}}', f'\\newcommand{{\\PSpLong}}{{{m128:.2f}}}', f'\\newcommand{{\\PLongN}}{{{len(sp128)}}}',
         f"\\newcommand{{\\PSpQwen}}{{{q.loc['lme_q4t', 'speedup_op']:.2f}}}",
         f"\\newcommand{{\\PSpQwenLong}}{{{dict(sp128).get('lme_q4t', np.nan):.2f}}}",
         f"\\newcommand{{\\PNQwenBL}}{{{q.loc['lme_q4t', 'N_vs_bl']}}}", f"\\newcommand{{\\PNQwenOrig}}{{{float(q.loc['lme_q4t', 'N_vs_orig']):.0f}}}"]
    open('results/paper/macros_time.tex', 'w').write('\n'.join(L))
    print('\n'.join(L)); print(sp128)


if __name__ == '__main__':
    op_macros()


def total_table(N_list=(1000, 10000)):
    """Tổng thời gian ở điểm 95 % của TỪNG phương pháp (6 baseline + SwapKL; N = 1000), cùng giả định độ dài sinh GEN như Bảng 2.
    Đếm số thiết lập SwapKL có tổng thời gian nhỏ nhất ở N = 1000 / 10000, có và không tính các lần chạy SwapKL ở mọi k lưới nhỏ hơn.
    -> results/paper/t_total.tex, results/paper/macros_total.tex"""
    d = pd.read_csv('results/perbench/ksweep.csv')
    d = d[[k in STD_GRID[s] for s, k in zip(d.setting, d.k)]]
    sp = speed_table(32768)
    cols = BL + ['swapkl']
    lines, wins = [], {(N, sw): 0 for N in N_list for sw in (0, 1)}
    for st in ORDER:
        L, swa, full = REF[st]; cs = costs(st); g = sp[sp.model == SPMODEL[st]]; G = GEN.get(st, 256)
        treq = {int(k): (float(p) + G * float(t) / 1000) / 4 for k, p, t in zip(g.k, g.prefill_s, g.tpot_ms)}
        tfull = (float(g.full_prefill_s.iloc[0]) + G * float(g.full_tpot_ms.iloc[0]) / 1000) / 4
        sc = {m: dict(zip(h.k, h.score)) for m, h in d[d.setting == st].groupby('method')}
        T, cells = {}, []
        for m in cols:
            k95 = kmin(sc.get(m, {}), full, 0.95)
            if k95 is None:
                cells.append('\\na'); continue
            sel = cs.get((m, k95), np.nan)
            sweep = sum(cs.get((m, kk), 0) for kk in STD_GRID[st] if kk < k95) if m == 'swapkl' else 0.0
            T[m] = {(N, sw): sel + sw * sweep + N * treq[k95] / 60 for N in N_list for sw in (0, 1)}
            cells.append(f'{k95} / {T[m][(1000, 0)]:.0f}')
        for key in wins:
            if 'swapkl' in T and all(T['swapkl'][key] <= T[m][key] for m in T):
                wins[key] += 1
        lines.append(f'{NAME[st]} & ' + ' & '.join(cells) + f' & {1000 * tfull / 60:.0f} \\\\')
    hdr = ' & '.join(LAB[c] if c != 'swapkl' else '\\textbf{SwapKL}' for c in cols)
    tex = ['\\begin{tabular}{@{}l|' + 'c' * len(cols) + '|c@{}}', '\\toprule', f'Setting & {hdr} & Orig. \\\\', '\\midrule'] + lines + \
          ['\\bottomrule', '\\end{tabular}']
    open('results/paper/t_total.tex', 'w').write('\n'.join(tex))
    M = [f'\\newcommand{{\\PWinA}}{{{wins[(1000, 0)]}}}', f'\\newcommand{{\\PWinB}}{{{wins[(10000, 0)]}}}',
         f'\\newcommand{{\\PWinASweep}}{{{wins[(1000, 1)]}}}', f'\\newcommand{{\\PWinBSweep}}{{{wins[(10000, 1)]}}}']
    open('results/paper/macros_total.tex', 'w').write('\n'.join(M))
    print('\n'.join(tex)); print(wins)


if __name__ == '__main__':
    total_table()
