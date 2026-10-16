"""
Vẽ kết quả quét k: mỗi benchmark một hình gồm hai biểu đồ cạnh nhau (không dùng hai trục y):
  trái  — điểm benchmark theo k (kèm mốc k=0: SWA thuần, k=L: mô hình gốc);
  phải  — chi phí BƯỚC CHỌN TẦNG (phút, thang log) theo k.
Bảng màu: palette 8 hạng mục tham chiếu của skill dataviz (đã kiểm tra CVD); mỗi phương pháp có màu + dấu cố định.
    python nfat/ksweep_plot.py  ->  figures/ksweep_<setting>.png (+ .pdf)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STYLE = {  # phương pháp: (nhãn, màu, dấu) — thứ tự cố định, không xoay vòng
    'swapkl': ('SwapKL', '#2a78d6', 'o'), 'frs': ('FRS-4', '#eb6834', 's'), 'duo': ('DuoAttention', '#1baf7a', '^'),
    'rethead': ('Retrieval Head', '#eda100', 'D'), 'cake': ('CAKE', '#e87ba4', 'v'), 'moa': ('MoA', '#008300', 'P'),
    'bi': ('BI', '#4a3aa7', 'X'), 'sleb': ('SLEB', '#e34948', '*'),
    'nll_swa': ('NLL-guided', '#9a6b3c', 'h')}
TITLE = {'niah1b': 'NIAH trong phân phối, Llama 3.2 1B (TB 6 ô)', 'sniah512': 'S-NIAH TB@4K, Llama 3.1 8B, W=512',
         'sniah256': 'S-NIAH TB@4K, Llama 3.1 8B, W=256', 'ruler': 'RULER 8K (11 tác vụ, 25 mẫu), Llama 3.1 8B',
         'bab': 'BABILong 2K+4K (qa1–5), Llama 3.1 8B, W=256', 'niah8b': 'NIAH trong phân phối, Llama 3.1 8B (TB 6 ô)',
         'lme_l8i': 'LongMemEval_24k, Llama-3.1-8B-Instruct', 'lme_q4t': 'LongMemEval_24k, Qwen3-4B-Thinking'}
YLAB = {'lme_l8i': 'độ chính xác', 'lme_q4t': 'độ chính xác'}
# mốc k=0 (SWA+sink) và k=L (gốc) — số đã có trong báo cáo chính / Report.pdf (S-NIAH, RULER 8K: TB 3 tác vụ / 11 tác vụ)
REF = {'niah1b': (16, .20, 1.0), 'sniah512': (32, .16, 1.0), 'sniah256': (32, .08, 1.0), 'ruler': (32, .197, .983),
       'bab': (32, (.300 + .204) / 2, (.636 + .572) / 2), 'niah8b': (32, .20, 1.0), 'lme_l8i': (32, .208, .604),
       'lme_q4t': (36, .528, .788)}
import sys; sys.path.insert(0, ROOT); from nfat.ksweep import S   # noqa: E402


def costs(st):
    """Phút chọn tầng theo (phương pháp, k). Baseline cho điểm: một lần chạy, mọi k. SLEB: thời gian đo tới k_run, quy đổi theo
    số đỉnh phải đánh giá sum_{j=k+1}^{L} j (k < k_run cộng thêm phần nối tiếp đo thật). SwapKL/FRS-4: đo thật cho từng k."""
    c = S[st]; L = c['L']
    sc = pd.read_csv('results/select_cost.csv').drop_duplicates(['tag', 'scorer'], keep='last').set_index(['tag', 'scorer']).seconds / 60
    out = {}
    for m in c['methods']:
        for k in c['ks']:
            if m in ('duo', 'rethead', 'cake', 'moa', 'bi'):
                v = sc.get((c['btag'], m), np.nan)
            elif m == 'nll_swa':                      # NLL-guided: một lần chấm điểm trên tag PerBench -> mọi k
                v = sc.get((min(c['ntag'].items())[1], 'nll_swa'), np.nan)
            elif m == 'sleb':
                full = sc.get((c['btag'], 'sleb'), np.nan); n_run = sum(range(c['krun'] + 1, L + 1))
                if k >= c['krun']:
                    v = full * sum(range(k + 1, L + 1)) / n_run
                else:
                    tail = sc.get((c['btag'] + '-tail', 'sleb_tail'), np.nan)
                    v = full + tail * sum(range(k + 1, c['krun'] + 1)) / sum(range(2, c['krun'] + 1))
            else:
                tag = c['ntag'].get(k, f"{c['ntag'][min(c['ntag'])]}-k{k}")
                v = sc.get((tag, m), np.nan)
            out[(m, k)] = v
    return out


def main():
    d = pd.read_csv('results/perbench/ksweep.csv')
    os.makedirs('figures', exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                         'axes.labelcolor': '#3a3a38', 'xtick.color': '#5a5a57', 'ytick.color': '#5a5a57'})
    for st, g in d.groupby('setting'):
        L, swa, full = REF[st]; cs = costs(st)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1.35, 1]})
        ax.axhline(full, color='#b9b8b0', lw=1, ls='--'); ax.text(0.2, full, ' mô hình gốc (k=L)', va='bottom', fontsize=7, color='#5a5a57')
        ax.scatter([0], [swa], color='#5a5a57', marker='o', s=18, zorder=3); ax.text(0.3, swa, ' SWA (k=0)', va='center', fontsize=7, color='#5a5a57')
        for m, (lab, col, mk) in STYLE.items():
            h = g[g.method == m].sort_values('k')
            if h.empty:
                continue
            lw = 2.2 if m in ('swapkl', 'frs') else 1.4
            ax.plot(h.k, h.score, color=col, lw=lw, marker=mk, ms=6, label=lab, zorder=4 if m in ('swapkl', 'frs') else 3)
            ck = [(k, cs.get((m, k), np.nan)) for k in h.k]
            ck = [(k, v) for k, v in ck if np.isfinite(v) and v > 0]
            if ck:
                bx.plot([x for x, _ in ck], [y for _, y in ck], color=col, lw=lw, marker=mk, ms=5, label=lab)
        ax.set_xlabel('k (số tầng attention đầy đủ)'); ax.set_ylabel(YLAB.get(st, 'điểm')); ax.set_title(TITLE[st], fontsize=9, loc='left')
        ax.grid(axis='y', color='#ececE8', lw=0.6); ax.set_xlim(-0.5, max(g.k) + 1)
        bx.set_yscale('log'); bx.set_xlabel('k'); bx.set_ylabel('chi phí chọn tầng (phút, log)'); bx.set_title('Chi phí bước chọn tầng', fontsize=9, loc='left')
        bx.grid(axis='y', color='#ececE8', lw=0.6, which='both')
        ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(f'figures/ksweep_{st}.{ext}', dpi=180)
        plt.close(fig)
        print('->', f'figures/ksweep_{st}.png')


if __name__ == '__main__':
    main()
