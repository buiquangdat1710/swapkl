"""
Hình cho báo cáo SwapKL (bản VI + EN): quét k, chỉ SwapKL + 6 baseline (bỏ FRS-4).
  trái  — điểm theo k (chấm xám: SWA thuần k=0; đường đứt: mô hình gốc k=L);
  phải  — chi phí bước chọn tầng (phút, log) theo k.
Thêm một hình tốc độ suy luận theo k (results/speed_offline.csv, tên ks_*).
    python nfat/swapkl_report_plot.py  ->  figures/swk_<setting>_<lang>.{png,pdf}, figures/swk_speed_<lang>.{png,pdf}
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
from nfat.ksweep import S                  # noqa: E402
from nfat.ksweep_plot import REF, costs     # noqa: E402

# lưới k CHUẨN của báo cáo gốc (bảng tab:setup); k dày thêm 10-12 không dùng trong cả hai bản báo cáo
STD_GRID = {'niah1b': [1, 2, 3, 4, 6, 8, 12], 'lme_l8i': [4, 8, 16], 'lme_q4t': [4, 9, 18]}
STD_GRID.update({s_: [1, 2, 4, 6, 8, 12, 16, 24] for s_ in ['sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']})
VARIANT = os.environ.get('SWK_REPORT', 'orig')          # orig: BaoCao_SwapKL (có SLEB); nosleb: BaoCao_SwapKL_KhongSLEB
PFX = 'swk' if VARIANT == 'orig' else 'swk_nosleb'
STYLE = {'swapkl': ('SwapKL', '#2a78d6', 'o'), **({'sleb': ('SLEB', '#e34948', '*')} if VARIANT == 'orig' else {'nll_swa': ('NLL-guided', '#e34948', '*')}), 'duo': ('DuoAttention', '#1baf7a', '^'),
         'rethead': ('Retrieval Head', '#eda100', 'D'), 'cake': ('CAKE', '#e87ba4', 'v'), 'moa': ('MoA', '#008300', 'P'),
         'bi': ('BI', '#4a3aa7', 'X')}
TXT = {
    'vi': dict(title={'niah1b': 'NIAH, Llama 3.2 1B (TB 6 ô)', 'sniah512': 'S-NIAH 4K, Llama 3.1 8B, W=512',
                      'sniah256': 'S-NIAH 4K, Llama 3.1 8B, W=256', 'ruler': 'RULER 8K, Llama 3.1 8B, W=512',
                      'bab': 'BABILong 2K+4K, Llama 3.1 8B, W=256', 'niah8b': 'NIAH, Llama 3.1 8B (TB 6 ô)',
                      'lme_l8i': 'LongMemEval, Llama-3.1-8B-Instruct', 'lme_q4t': 'LongMemEval, Qwen3-4B-Thinking'},
               full=' mô hình gốc (k=L)', swa=' SWA (k=0)', xk='k (số tầng attention đầy đủ)', y='độ chính xác',
               ycost='chi phí chọn tầng (phút, log)', tcost='Chi phí bước chọn tầng',
               sp_title='Tốc độ suy luận theo k (ngữ cảnh 32K, 4 prompt)', sp_y='tăng tốc so với mô hình gốc (lần)',
               sp_pre='prefill', sp_dec='giải mã (ms/token)', sp_tot='tổng (prefill + 256 token)',
               ttot='Độ chính xác theo tổng thời gian (1000 yêu cầu 32K)', xtot='tổng thời gian (phút) = chọn tầng + suy luận'),
    'en': dict(title={'niah1b': 'NIAH, Llama 3.2 1B (mean of 6 cells)', 'sniah512': 'S-NIAH 4K, Llama 3.1 8B, W=512',
                      'sniah256': 'S-NIAH 4K, Llama 3.1 8B, W=256', 'ruler': 'RULER 8K, Llama 3.1 8B, W=512',
                      'bab': 'BABILong 2K+4K, Llama 3.1 8B, W=256', 'niah8b': 'NIAH, Llama 3.1 8B (mean of 6 cells)',
                      'lme_l8i': 'LongMemEval, Llama-3.1-8B-Instruct', 'lme_q4t': 'LongMemEval, Qwen3-4B-Thinking'},
               full=' original model (k=L)', swa=' SWA (k=0)', xk='k (number of full-attention layers)', y='accuracy',
               ycost='layer-selection cost (min, log)', tcost='Layer-selection cost',
               sp_title='Inference speed vs k (32K context, 4 prompts)', sp_y='speed-up over the original model (x)',
               sp_pre='prefill', sp_dec='decode (ms/token)', sp_tot='total (prefill + 256 tokens)',
               ttot='Accuracy vs total time (1000 requests at 32K)', xtot='total time (min) = selection + inference'),
}
SPEED = {  # thiết lập tốc độ: (nhãn, tiền tố tên, W, tên mô hình gốc, L)
    'l8b': ('Llama 3.1 8B, W=512', 'ks_l8b_w512', 'l8b_full', 32),
    'l8b256': ('Llama 3.1 8B, W=256', 'ks_l8b_w256', 'l8b_full', 32),   # chỉ dùng cho bảng điểm vận hành (S-NIAH W=256, BABILong)
    'l1b': ('Llama 3.2 1B, W=512', 'ks_l1b_w512', 'l1b_full', 16),
    'l8i': ('Llama-3.1-8B-Instruct, W=2048, full decode', 'ks_l8i_w2048', 'l8i_full', 32),
    'q4t': ('Qwen3-4B-Thinking, W=2048, full decode', 'ks_q4t_w2048', 'full', 36),
}


NREQ = 1000          # khối lượng suy luận cho "tổng thời gian": N yêu cầu 32K token + sinh 256 token (bản không SLEB)
SPMODEL = {'niah1b': 'l1b', 'sniah512': 'l8b', 'ruler': 'l8b', 'niah8b': 'l8b', 'sniah256': 'l8b256', 'bab': 'l8b256',
           'lme_l8i': 'l8i', 'lme_q4t': 'q4t'}


def total_time(st, m, k, cs, sp, N=NREQ):
    """Phút: chi phí chọn tầng (một lần) + suy luận N yêu cầu ở 32K (tốc độ đo thật theo k, lô 4 prompt; mọi tập ở cùng k cùng tốc độ)."""
    sel = cs.get((m, k), np.nan)
    g = sp[(sp.model == SPMODEL[st]) & (sp.k == k)]
    if g.empty or not np.isfinite(sel):
        return np.nan, np.nan, np.nan
    inf = N / 4 * float(g.total_s.iloc[0]) / 60
    return sel + inf, sel, inf


def style():
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                         'axes.labelcolor': '#3a3a38', 'xtick.color': '#5a5a57', 'ytick.color': '#5a5a57'})


def ksweep_figs(lang):
    t = TXT[lang]
    d = pd.read_csv('results/perbench/ksweep.csv')
    d = d[[k in STD_GRID[st_] for st_, k in zip(d.setting, d.k)]]
    global SPT
    SPT = speed_table_all(32768)
    for st, g in d.groupby('setting'):
        L, swa, full = REF[st]; cs = costs(st)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1.35, 1]})
        ax.axhline(full, color='#b9b8b0', lw=1, ls='--'); ax.text(0.2, full, t['full'], va='bottom', fontsize=7, color='#5a5a57')
        ax.scatter([0], [swa], color='#5a5a57', marker='o', s=18, zorder=3); ax.annotate(t['swa'].strip(), (0, swa), xytext=(-2, -11), textcoords='offset points', fontsize=7, color='#5a5a57')
        for m, (lab, col, mk) in STYLE.items():
            h = g[g.method == m].sort_values('k')
            if h.empty:
                continue
            main = m == 'swapkl'
            ax.plot(h.k, h.score, color=col, lw=2.4 if main else 1.3, marker=mk, ms=6 if main else 5, label=lab,
                    zorder=5 if main else 3, alpha=1 if main else .9)
            if VARIANT == 'orig':
                ck = [(k, cs.get((m, k), np.nan)) for k in h.k]
                ck = [(k, v) for k, v in ck if np.isfinite(v) and v > 0]
                if ck:
                    bx.plot([x for x, _ in ck], [y for _, y in ck], color=col, lw=2.4 if main else 1.3, marker=mk, ms=5, label=lab,
                            zorder=5 if main else 3)
            else:
                pts = [(total_time(st, m, int(k), cs, SPT)[0], sc_) for k, sc_ in zip(h.k, h.score)]
                pts = [(x, y) for x, y in pts if np.isfinite(x)]
                if pts:
                    bx.plot([x for x, _ in pts], [y for _, y in pts], color=col, lw=2.4 if main else 1.3, marker=mk, ms=5, label=lab,
                            zorder=5 if main else 3)
        ax.set_xlabel(t['xk']); ax.set_ylabel(t['y']); ax.set_title(t['title'][st], fontsize=9, loc='left')
        ax.grid(axis='y', color='#ececE8', lw=0.6); ax.set_xlim(-0.5, max(g.k) + 1)
        if VARIANT == 'orig':
            bx.set_yscale('log'); bx.set_xlabel('k'); bx.set_ylabel(t['ycost']); bx.set_title(t['tcost'], fontsize=9, loc='left')
            bx.grid(axis='y', color='#ececE8', lw=0.6, which='both')
        else:
            bx.axhline(full, color='#b9b8b0', lw=1, ls='--')
            bx.set_xlabel(t['xtot']); bx.set_ylabel(t['y']); bx.set_title(t['ttot'], fontsize=9, loc='left')
            bx.grid(axis='y', color='#ececE8', lw=0.6)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(f'figures/{PFX}_{st}_{lang}.{ext}', dpi=180)
        plt.close(fig)


def speed_table(L_ctx=32768):
    """Tăng tốc so với mô hình gốc theo k, ở độ dài L_ctx: prefill, giải mã, tổng."""
    sp = pd.read_csv('results/speed_offline.csv').drop_duplicates(['name', 'L'], keep='last')
    rows = []
    for key, (lab, pre, fullname, L) in SPEED.items():
        f = sp[(sp.name == fullname) & (sp.L == L_ctx)]
        if f.empty:
            continue
        f = f.iloc[0]
        for _, r in sp[sp.name.str.startswith(pre + '_k') & (sp.L == L_ctx)].iterrows():
            k = int(r['name'].rsplit('_k', 1)[1])
            rows.append(dict(model=key, label=lab, k=k, L=L, prefill_s=r.prefill_s, tpot_ms=r.tpot_ms, total_s=r.total_s,
                             sp_pre=f.prefill_s / r.prefill_s, sp_dec=f.tpot_ms / r.tpot_ms, sp_tot=f.total_s / r.total_s,
                             full_prefill_s=f.prefill_s, full_tpot_ms=f.tpot_ms, full_total_s=f.total_s))
    return pd.DataFrame(rows).sort_values(['model', 'k']) if rows else pd.DataFrame()


def speed_table_all(L_ctx=32768):
    """Như speed_table nhưng giữ mọi cấu hình (kể cả l8b256) — dùng cho tổng thời gian."""
    return speed_table(L_ctx)


def speed_fig(lang):
    t = TXT[lang]
    tab = speed_table()
    tab = tab[tab.model != 'l8b256'] if not tab.empty else tab
    if tab.empty:
        return
    fig, axs = plt.subplots(1, len(tab.model.unique()), figsize=(3.1 * len(tab.model.unique()), 3.2), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, (key, g) in zip(axs, tab.groupby('model', sort=False)):
        ax.axhline(1, color='#b9b8b0', lw=1, ls='--')
        ax.plot(g.k, g.sp_pre, color='#2a78d6', marker='o', label=t['sp_pre'])
        ax.plot(g.k, g.sp_dec, color='#eb6834', marker='s', label=t['sp_dec'])
        ax.plot(g.k, g.sp_tot, color='#3a3a38', marker='^', label=t['sp_tot'])
        ax.set_title(SPEED[key][0].replace('full decode', 'giải mã đầy đủ') if lang == 'vi' else SPEED[key][0], fontsize=8, loc='left')
        ax.set_xlabel('k')
        ax.grid(axis='y', color='#ececE8', lw=0.6)
    axs[0].set_ylabel(t['sp_y']); axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(t['sp_title'], fontsize=9, x=0.01, ha='left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'figures/{PFX}_speed_{lang}.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    style()
    for lang in ('vi', 'en'):
        ksweep_figs(lang)
        speed_fig(lang)
    print('ok')
