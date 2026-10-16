"""
Số liệu + bảng LaTeX cho báo cáo SwapKL (VI + EN). Chỉ SwapKL + 6 baseline (bỏ FRS-4).
    python nfat/swapkl_report_tables.py   ->  results/swk_report/{vi,en}/*.tex, results/swk_report/summary.json
Quy ước:
  - điểm "TB theo k" = trung bình đều trên lưới k của từng benchmark (results/perbench/ksweep.csv);
  - baseline tốt nhất = baseline có điểm TB theo k cao nhất ở benchmark đó;
  - chi phí chọn tầng = phút wall-clock (nfat/ksweep_plot.costs), lấy trung bình trên lưới k;
  - điểm vận hành so với mô hình gốc = k NHỎ NHẤT trên lưới mà SwapKL giữ >= 95 % độ chính xác của mô hình gốc — tức số tầng
    SWA NHIỀU NHẤT còn giữ chất lượng; z (hai tỉ lệ, không ghép cặp, n = số câu test) được in kèm để đọc độ tin cậy.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from nfat.ksweep import S                          # noqa: E402
from nfat.ksweep_plot import REF, costs             # noqa: E402
from nfat.swapkl_report_plot import speed_table, STD_GRID      # noqa: E402

RETAIN = 0.95
ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b', 'lme_l8i', 'lme_q4t']
# Hai bản báo cáo (10-14): SWK_REPORT=orig (mặc định) -> BaoCao_SwapKL.tex, 6 baseline gồm SLEB, ra results/swk_report/;
# SWK_REPORT=nosleb -> BaoCao_SwapKL_KhongSLEB.tex: bỏ SLEB, thêm NLL-guided (nll_swa) khi đã có đủ điểm ở MỌI benchmark và mọi k,
# ra results/swk_report_nosleb/.
VARIANT = os.environ.get('SWK_REPORT', 'orig')
OUTD = 'results/swk_report' if VARIANT == 'orig' else 'results/swk_report_nosleb'
BL = ['duo', 'rethead', 'cake', 'moa', 'bi'] + (['sleb'] if VARIANT == 'orig' else [])
_ks = pd.read_csv('results/perbench/ksweep.csv')
if VARIANT != 'orig' and all(set(g[g.method == 'swapkl'].k) <= set(g[g.method == 'nll_swa'].k) for _, g in _ks.groupby('setting')) and \
        _ks[_ks.method == 'nll_swa'].setting.nunique() == len(ORDER):
    BL = BL + ['nll_swa']
LAB = {'duo': 'Duo', 'rethead': 'RH', 'cake': 'CAKE', 'moa': 'MoA', 'bi': 'BI', 'sleb': 'SLEB', 'nll_swa': 'NLL-g', 'swapkl': 'SwapKL'}
NTEST = {'niah1b': 300, 'sniah512': 150, 'sniah256': 150, 'ruler': 275, 'bab': 500, 'niah8b': 300, 'lme_l8i': 500, 'lme_q4t': 500}
SPMODEL = {'niah1b': 'l1b', 'sniah512': 'l8b', 'sniah256': 'l8b256', 'ruler': 'l8b', 'bab': 'l8b256', 'niah8b': 'l8b',
           'lme_l8i': 'l8i', 'lme_q4t': 'q4t'}
NAME = {
    'vi': {'niah1b': 'NIAH (1B)', 'sniah512': 'S-NIAH $W{=}512$', 'sniah256': 'S-NIAH $W{=}256$', 'ruler': 'RULER 8K',
           'bab': 'BABILong 2K+4K', 'niah8b': 'NIAH (8B)', 'lme_l8i': 'LME Llama-8B-I', 'lme_q4t': 'LME Qwen3-4B'},
    'en': {'niah1b': 'NIAH (1B)', 'sniah512': 'S-NIAH $W{=}512$', 'sniah256': 'S-NIAH $W{=}256$', 'ruler': 'RULER 8K',
           'bab': 'BABILong 2K+4K', 'niah8b': 'NIAH (8B)', 'lme_l8i': 'LME Llama-8B-I', 'lme_q4t': 'LME Qwen3-4B'},
}
TT = {
    'vi': dict(orig='Gốc', swa='SWA', gain='$\\Delta$ so với BL tốt nhất', mean='Trung bình', bench='Benchmark',
               win='thắng', tie='hoà', loss='thua', cells='số ô $k$', k='$k$', best='BL tốt nhất', cost_swk='SwapKL',
               ratio='SwapKL / BL tốt nhất', prefill='Prefill', decode='ms/token', total='Tổng', kop='$k$ chọn',
               acc_swk='SwapKL', acc_orig='Gốc', drop='Giảm', sp_tot='Nhanh hơn (tổng)', sp_pre='Prefill',
               model='Mô hình', note_na='không có'),
    'en': dict(orig='Orig.', swa='SWA', gain='$\\Delta$ vs best BL', mean='Mean', bench='Benchmark',
               win='win', tie='tie', loss='loss', cells='$k$ cells', k='$k$', best='Best BL', cost_swk='SwapKL',
               ratio='SwapKL / best BL', prefill='Prefill', decode='ms/token', total='Total', kop='chosen $k$',
               acc_swk='SwapKL', acc_orig='Orig.', drop='Drop', sp_tot='Speed-up (total)', sp_pre='Prefill',
               model='Model', note_na='n/a'),
}


def dec(x, lang, nd=3):
    s = f'{x:.{nd}f}'
    s = s.replace('.', '{,}') if lang == 'vi' else s
    return '$-$' + s[1:] if s.startswith('-') else s


def pct(x, lang, nd=1, sign=True):
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    s = (s.replace('.', '{,}') if lang == 'vi' else s) + '\\,\\%'
    return '$-$' + s[1:] if s.startswith('-') else s


def data():
    d = pd.read_csv('results/perbench/ksweep.csv')
    d = d[d.method.isin(BL + ['swapkl'])]
    # cả hai bản báo cáo dùng LƯỚI CHUẨN của báo cáo gốc (bảng tab:setup); k dày thêm 10-12 không dùng
    d = d[[k in STD_GRID[st_] for st_, k in zip(d.setting, d.k)]]
    # chỉ giữ các k mà MỌI phương pháp của benchmark đó đều có điểm (lưới đồng nhất -> trung bình so sánh được)
    keep = []
    for st_, g in d.groupby('setting'):
        ms = set(g.method); cnt = g.groupby('k').method.nunique()
        keep.append(g[g.k.isin(cnt[cnt == len(ms)].index)])
    d = pd.concat(keep)
    piv = d.pivot_table(index=['setting', 'k'], columns='method', values='score')
    avg = d.groupby(['setting', 'method']).score.mean().unstack().loc[ORDER]
    return d, piv, avg


def compute():
    d, piv, avg = data()
    out = {'bench': {}}
    for st in ORDER:
        r = avg.loc[st]; b = r[BL].dropna(); lead = b.idxmax()
        g = piv.loc[st]; nk = len(g); n = NTEST[st]
        var = lambda c: float((g[c] * (1 - g[c])).sum() / n / nk ** 2)   # noqa: E731
        z = (r['swapkl'] - b.max()) / np.sqrt(var('swapkl') + var(lead)) if r['swapkl'] != b.max() else 0.0
        cs = costs(st); ks = sorted(int(x) for x in g.index)
        cost = {m: float(np.nanmean([cs.get((m, k), np.nan) for k in ks])) for m in BL + ['swapkl'] if (m, ks[0]) in cs}
        # thắng/hoà/thua theo từng k so với baseline tốt nhất Ở Ô ĐÓ
        w = t = l_ = 0
        for k in ks:
            row = g.loc[k]; bm = row[BL].max()
            if row['swapkl'] > bm + 1e-9:
                w += 1
            elif row['swapkl'] < bm - 1e-9:
                l_ += 1
            else:
                t += 1
        # điểm vận hành so với mô hình gốc
        L, swa, full = REF[st]
        op = None
        for k in ks:
            a = float(g.loc[k, 'swapkl'])
            se = np.sqrt(a * (1 - a) / n + full * (1 - full) / n)
            zz = (a - full) / se if se > 0 else (0.0 if a >= full else -np.inf)
            if a >= RETAIN * full:
                op = dict(k=int(k), acc=a, z=float(zz)); break
        out['bench'][st] = dict(avg={m: float(r[m]) for m in r.index if pd.notna(r[m])}, lead=lead, lead_avg=float(b.max()),
                                gain_abs=float(r['swapkl'] - b.max()), gain_rel=float((r['swapkl'] / b.max() - 1) * 100), z=float(z),
                                wins=w, ties=t, losses=l_, cost=cost, cost_ratio=float(cost['swapkl'] / cost[lead]),
                                L=L, swa=swa, full=full, op=op, n=n, ks=[int(k) for k in ks])
    B = out['bench']
    out['mean_gain_rel'] = float(np.mean([B[s]['gain_rel'] for s in ORDER]))
    out['mean_gain_abs'] = float(np.mean([B[s]['gain_abs'] for s in ORDER]))
    out['mean_cost_ratio'] = float(np.mean([B[s]['cost_ratio'] for s in ORDER]))
    out['sum_cost'] = {'swapkl': float(sum(B[s]['cost']['swapkl'] for s in ORDER)),
                       'lead': float(sum(B[s]['cost'][B[s]['lead']] for s in ORDER))}
    out['strict_wins'] = int(sum(B[s]['gain_abs'] > 1e-9 for s in ORDER))
    out['cells'] = {x: int(sum(B[s][x] for s in ORDER)) for x in ('wins', 'ties', 'losses')}
    # tốc độ ở điểm vận hành (32K và 128K)
    for Lc in (32768, 'long'):
        sp = speed_table(32768) if Lc == 32768 else long_speed()
        key = 'sp32k' if Lc == 32768 else 'splong'
        for st in ORDER:
            op = B[st]['op']
            if op is None or sp.empty:
                continue
            m = sp[(sp.model == SPMODEL[st]) & (sp.k == op['k'])]
            if not m.empty:
                m = m.iloc[0]
                B[st][key] = dict(pre=float(m.sp_pre), dec=float(m.sp_dec), tot=float(m.sp_tot), L=int(m.get('Lctx', 32768)))
    ops = [s for s in ORDER if B[s]['op'] and 'sp32k' in B[s]]
    if ops:
        out['op_mean_drop_rel'] = float(np.mean([(B[s]['op']['acc'] / B[s]['full'] - 1) * 100 for s in ops]))
        out['op_mean_sp_tot'] = float(np.mean([B[s]['sp32k']['tot'] for s in ops]))
        out['op_mean_sp_pre'] = float(np.mean([B[s]['sp32k']['pre'] for s in ops]))
        out['op_mean_sp_dec'] = float(np.mean([B[s]['sp32k']['dec'] for s in ops]))
        lo = [s for s in ops if 'splong' in B[s]]
        if lo:
            out['op_mean_sp_tot_long'] = float(np.mean([B[s]['splong']['tot'] for s in lo]))
            out['op_long_benches'] = lo
        out['op_benches'] = ops
    return out


def long_speed():
    """Tăng tốc ở ngữ cảnh dài nhất đã đo (130K cho Llama 8B, 131K cho Qwen)."""
    rows = []
    for Lc in (130048, 131072):
        t = speed_table(Lc)
        if not t.empty:
            t = t.assign(Lctx=Lc); rows.append(t)
    return pd.concat(rows) if rows else pd.DataFrame()


def write_tables(o, lang):
    t = TT[lang]; nm = NAME[lang]; B = o['bench']
    os.makedirs(f'{OUTD}/{lang}', exist_ok=True)
    W = lambda f, s: open(f'{OUTD}/{lang}/{f}.tex', 'w').write(s)   # noqa: E731
    _, piv, avg = data()

    # Bảng 1: TB theo k
    cols = BL + ['swapkl']
    s = ['\\begin{tabular}{@{}l|cc|' + 'c' * len(BL) + '|c|rr@{}}', '\\toprule',
         f"{t['bench']} & {t['orig']} & {t['swa']} & " + ' & '.join(LAB[c] for c in BL) + f" & \\textbf{{SwapKL}} & {t['gain']} & $z$ \\\\",
         '\\midrule']
    for st in ORDER:
        r = B[st]['avg']; best = max(r[c] for c in cols if c in r)
        cells = []
        for c in cols:
            if c not in r:
                cells.append('\\na'); continue
            v = dec(r[c], lang)
            cells.append(f'\\textbf{{{v}}}' if abs(r[c] - best) < 1e-9 else v)
        g = pct(B[st]['gain_rel'], lang) if abs(B[st]['gain_abs']) > 1e-9 else ('0' if lang == 'en' else '0')
        s.append(f"{nm[st]} & {dec(B[st]['full'], lang)} & {dec(B[st]['swa'], lang)} & " + ' & '.join(cells) + f" & {g} & {dec(B[st]['z'], lang, 1)} \\\\")
    s += ['\\midrule', f"{t['mean']} & & & " + ' & ' * (len(BL)) + f" & {pct(o['mean_gain_rel'], lang)} & \\\\", '\\bottomrule',
          '\\end{tabular}']
    W('t_kavg', '\n'.join(s))

    # Bảng 2: thắng/hoà/thua theo ô k
    s = ['\\begin{tabular}{@{}lccc|c@{}}', '\\toprule',
         f"{t['bench']} & {t['win']} & {t['tie']} & {t['loss']} & {t['cells']} \\\\", '\\midrule']
    for st in ORDER:
        b = B[st]; s.append(f"{nm[st]} & {b['wins']} & {b['ties']} & {b['losses']} & {len(b['ks'])} \\\\")
    c = o['cells']; s += ['\\midrule', f"{t['mean'] if False else ('Tổng' if lang == 'vi' else 'Total')} & {c['wins']} & {c['ties']} & {c['losses']} & {sum(c.values())} \\\\",
                          '\\bottomrule', '\\end{tabular}']
    W('t_wins', '\n'.join(s))

    # Bảng 3: chi phí chọn tầng (phút, TB theo k)
    s = ['\\begin{tabular}{@{}l|' + 'r' * len(BL) + '|r|lr@{}}', '\\toprule',
         f"{t['bench']} & " + ' & '.join(LAB[c] for c in BL) + f" & \\textbf{{SwapKL}} & {t['best']} & {t['ratio']} \\\\", '\\midrule']
    for st in ORDER:
        cs = B[st]['cost']
        cells = [dec(cs[c], lang, 1) if c in cs and np.isfinite(cs[c]) else '\\na' for c in BL]
        s.append(f"{nm[st]} & " + ' & '.join(cells) + f" & \\textbf{{{dec(cs['swapkl'], lang, 1)}}} & {LAB[B[st]['lead']]} & "
                 f"{dec(B[st]['cost_ratio'], lang, 2)} \\\\")
    s += ['\\bottomrule', '\\end{tabular}']
    W('t_cost', '\n'.join(s))

    # Bảng phụ lục: điểm theo từng k
    for st in ORDER:
        g = piv.loc[st]; have = [c for c in BL + ['swapkl'] if c in g.columns and g[c].notna().all()]
        s = ['\\begin{tabular}{@{}r|' + 'c' * len(have) + '@{}}', '\\toprule',
             f"{t['k']} & " + ' & '.join(('\\textbf{SwapKL}' if c == 'swapkl' else LAB[c]) for c in have) + ' \\\\', '\\midrule']
        for k, row in g.iterrows():
            best = max(row[c] for c in have)
            s.append(f'{int(k)} & ' + ' & '.join((f'\\textbf{{{dec(row[c], lang)}}}' if abs(row[c] - best) < 1e-9 else dec(row[c], lang))
                                                 for c in have) + ' \\\\')
        s += ['\\midrule', ('TB' if lang == 'vi' else 'Mean') + ' & ' + ' & '.join(dec(g[c].mean(), lang) for c in have) + ' \\\\',
              '\\bottomrule', '\\end{tabular}']
        W(f't_k_{st}', '\n'.join(s))

    # Bảng tốc độ theo k (32K)
    sp = speed_table(32768)
    sp = sp[sp.model != 'l8b256'] if not sp.empty else sp
    if not sp.empty:
        s = ['\\begin{tabular}{@{}lr|rr|rr|rr@{}}', '\\toprule',
             f"{t['model']} & $k$ & {t['prefill']} (s) & $\\times$ & {t['decode']} & $\\times$ & {t['total']} (s) & $\\times$ \\\\", '\\midrule']
        for key, g in sp.groupby('model', sort=False):
            f = g.iloc[0]
            lab = g.iloc[0]['label'].replace('full decode', 'giải mã đầy đủ') if lang == 'vi' else g.iloc[0]['label']
            s.append(f"{lab} & {g.iloc[0]['L']} ({t['orig']}) & {dec(f.full_prefill_s, lang, 2)} & 1{',' if False else ''}"
                     f"{'{,}00' if lang == 'vi' else '.00'} & {dec(f.full_tpot_ms, lang, 1)} & {'1{,}00' if lang == 'vi' else '1.00'} & "
                     f"{dec(f.full_total_s, lang, 2)} & {'1{,}00' if lang == 'vi' else '1.00'} \\\\")
            for _, r in g.iterrows():
                s.append(f" & {r.k if r.k else '0 (SWA)'} & {dec(r.prefill_s, lang, 2)} & {dec(r.sp_pre, lang, 2)} & {dec(r.tpot_ms, lang, 1)} & "
                         f"{dec(r.sp_dec, lang, 2)} & {dec(r.total_s, lang, 2)} & {dec(r.sp_tot, lang, 2)} \\\\")
            s.append('\\midrule')
        s[-1] = '\\bottomrule'; s.append('\\end{tabular}')
        W('t_speed', '\n'.join(s))

    # Bảng điểm vận hành so với mô hình gốc
    s = ['\\begin{tabular}{@{}l|r|ccrr|rr@{}}', '\\toprule',
         f"{t['bench']} & {t['kop']} & {t['acc_swk']} & {t['acc_orig']} & {t['drop']} & $z$ & {t['sp_pre']} & {t['sp_tot']} \\\\", '\\midrule']
    for st in ORDER:
        b = B[st]; op = b['op']
        if op is None:
            continue
        sp32 = b.get('sp32k')
        zs = dec(op['z'], lang, 1) if np.isfinite(op['z']) else '\\na'
        s.append(f"{nm[st]} & {op['k']}/{b['L']} & {dec(op['acc'], lang)} & {dec(b['full'], lang)} & "
                 f"{pct((op['acc'] / b['full'] - 1) * 100, lang)} & {zs} & "
                 + (f"{dec(sp32['pre'], lang, 2)}$\\times$ & {dec(sp32['tot'], lang, 2)}$\\times$" if sp32 else '\\na & \\na') + ' \\\\')
    if 'op_mean_sp_tot' in o:
        s += ['\\midrule', f"{t['mean']} & & & & {pct(o['op_mean_drop_rel'], lang)} & & {dec(o['op_mean_sp_pre'], lang, 2)}$\\times$ & "
              f"{dec(o['op_mean_sp_tot'], lang, 2)}$\\times$ \\\\"]
    s += ['\\bottomrule', '\\end{tabular}']
    W('t_op', '\n'.join(s))


def write_macros(o, lang):
    """Số liệu tiêu đề cho phần tóm tắt/kết luận (\\newcommand)."""
    f = lambda x, nd=1: dec(x, lang, nd)   # noqa: E731
    m = {'GainRel': f(o['mean_gain_rel']), 'GainAbs': f(o['mean_gain_abs'] * 100), 'StrictWins': str(o['strict_wins']),
         'CostRatioPct': f(o['mean_cost_ratio'] * 100, 0), 'CostSavePct': f((1 - o['mean_cost_ratio']) * 100, 0),
         'CostX': f(1 / o['mean_cost_ratio'], 1), 'SumCostSwk': f(o['sum_cost']['swapkl'], 0), 'SumCostLead': f(o['sum_cost']['lead'], 0),
         'SumCostSavePct': f((1 - o['sum_cost']['swapkl'] / o['sum_cost']['lead']) * 100, 0),
         'CellsW': str(o['cells']['wins']), 'CellsT': str(o['cells']['ties']), 'CellsL': str(o['cells']['losses']),
         'CellsN': str(sum(o['cells'].values())),
         # 10-14: so với DuoAttention (baseline tối ưu hoá duy nhất còn lại) và với các bộ chấm một lượt (RH, CAKE, BI)
         'CostDuoPct': f(100 * np.mean([o['bench'][s_]['cost']['swapkl'] / o['bench'][s_]['cost']['duo'] for s_ in ORDER]), 0),
         'CostDuoMax': f(100 * max(o['bench'][s_]['cost']['swapkl'] / o['bench'][s_]['cost']['duo'] for s_ in ORDER), 0),
         'CostNllgPct': f(100 * np.mean([o['bench'][s_]['cost']['swapkl'] / o['bench'][s_]['cost']['nll_swa'] for s_ in ORDER]), 0)
         if 'nll_swa' in BL else '\\na',
         'CostOnePassX': f(np.median([o['bench'][s_]['cost']['swapkl'] / min(o['bench'][s_]['cost'][m] for m in ('rethead', 'cake', 'bi'))
                                      for s_ in ORDER if not s_.startswith('lme')]), 0)}
    if 'op_mean_sp_tot' in o:
        m.update({'OpDrop': f(-o['op_mean_drop_rel']), 'OpRetain': f(100 + o['op_mean_drop_rel']),
                  'OpSpTot': f(o['op_mean_sp_tot'], 2), 'OpSpTotPct': f((o['op_mean_sp_tot'] - 1) * 100, 0),
                  'OpLatSavePct': f((1 - 1 / o['op_mean_sp_tot']) * 100, 0),
                  'OpSpPre': f(o['op_mean_sp_pre'], 2), 'OpSpPrePct': f((o['op_mean_sp_pre'] - 1) * 100, 0),
                  'OpSpDec': f(o['op_mean_sp_dec'], 2), 'OpN': str(len(o['op_benches']))})
        if 'op_mean_sp_tot_long' in o:
            m.update({'OpSpTotLong': f(o['op_mean_sp_tot_long'], 2), 'OpSpTotLongPct': f((o['op_mean_sp_tot_long'] - 1) * 100, 0),
                      'OpLongN': str(len(o['op_long_benches']))})
    open(f'{OUTD}/{lang}/macros.tex', 'w').write(
        '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.items()) + '\n')


def write_total(lang, N_list=(100, 1000, 10000)):
    """(Bản không SLEB) Tổng thời gian = chọn tầng + suy luận N yêu cầu 32K ở k@95 của TỪNG phương pháp (k nhỏ nhất trên lưới
    chuẩn giữ >= 95 % điểm mô hình gốc). Mô hình gốc: chỉ suy luận (k = L)."""
    from nfat.swapkl_report_plot import total_time, speed_table, SPMODEL as SPM
    nm = NAME[lang]; _, piv, _ = data(); sp = speed_table(32768)
    sp_full = {}
    for key in set(SPM.values()):
        g = sp[sp.model == key]
        if len(g):
            sp_full[key] = float(g.full_total_s.iloc[0])
    cols = BL + ['swapkl']
    hdr = ' & '.join(('\\textbf{SwapKL}' if c == 'swapkl' else LAB[c]) for c in cols)
    s = ['\\begin{tabular}{@{}l|' + 'c' * len(cols) + '|c@{}}', '\\toprule',
         ('Benchmark' if lang == 'en' else 'Benchmark') + f' & {hdr} & ' + ('Gốc' if lang == 'vi' else 'Orig.') + ' \\\\', '\\midrule']
    rows = []
    for st in ORDER:
        full = REF[st][2]; g = piv.loc[st]; cs = costs(st)
        cells = []
        for c in cols:
            k95 = next((int(k) for k in g.index if g.loc[k, c] >= RETAIN * full - 1e-12), None) if c in g else None
            if k95 is None:
                cells.append('\\na'); continue
            tt = [total_time(st, c, k95, cs, sp, N)[0] for N in N_list]
            rows.append(dict(setting=st, method=c, k95=k95, **{f'T{N}': t for N, t in zip(N_list, tt)}))
            cells.append(f'{k95} / {dec(tt[1], lang, 0)}' if np.isfinite(tt[1]) else f'{k95} / \\na')
        fo = sp_full.get(SPM[st])
        s.append(f'{nm[st]} & ' + ' & '.join(cells) + (f' & {dec(N_list[1] / 4 * fo / 60, lang, 0)}' if fo else ' & \\na') + ' \\\\')
        if fo:
            rows.append(dict(setting=st, method='orig', k95=REF[st][0], **{f'T{N}': N / 4 * fo / 60 for N in N_list}))
    s += ['\\bottomrule', '\\end{tabular}']
    open(f'{OUTD}/{lang}/t_total.tex', 'w').write('\n'.join(s))
    pd.DataFrame(rows).to_csv(f'{OUTD}/total_time.csv', index=False)


if __name__ == '__main__':
    o = compute()
    for lang in ('vi', 'en'):
        write_tables(o, lang)
        write_macros(o, lang)
        if VARIANT != 'orig':
            write_total(lang)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(o, open(f'{OUTD}/summary.json', 'w'), indent=1, default=float)
    B = o['bench']
    for st in ORDER:
        b = B[st]
        print(f"{st:9s} swk {b['avg']['swapkl']:.3f} lead {b['lead']:7s} {b['lead_avg']:.3f} gain {b['gain_rel']:+.1f}% z {b['z']:+.2f} "
              f"W/T/L {b['wins']}/{b['ties']}/{b['losses']} cost swk {b['cost']['swapkl']:.1f} lead {b['cost'][b['lead']]:.1f} "
              f"ratio {b['cost_ratio']:.2f} op {b['op']} sp {b.get('sp32k')}")
    print({k: v for k, v in o.items() if k != 'bench'})
