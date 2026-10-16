"""
Kiểm định GHÉP CẶP theo câu hỏi cho Bảng 1 của paper (phản biện OE-Agent 10-16, W3): hiệu TB lưới (SwapKL − baseline dẫn đầu),
bootstrap theo cụm câu hỏi (nfat/swk_v2_stats.paired_cells; mỗi câu hỏi là một cụm chung cho mọi k). Thay cột z không ghép cặp
của tables/t_kavg.tex bằng "Δ (95 % CI)" -> paper_acml/tables/t_kavg.tex (bản sao riêng của paper), results/paper/paired.csv.
    python nfat/paper_stats.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from nfat import swk_v2_stats as V                       # noqa: E402
from nfat.swapkl_report_plot import STD_GRID             # noqa: E402

ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b', 'lme_l8i', 'lme_q4t']


def main():
    o = json.load(open('results/swk_report_nosleb/summary.json'))
    d = pd.read_csv('results/perbench/ksweep.csv', dtype={'layers': str})
    rows = []
    for st in ORDER:
        lead = o['bench'][st]['lead']; cells, miss = [], 0
        for k in STD_GRID[st]:
            a = d[(d.setting == st) & (d.method == 'swapkl') & (d.k == k)].layers
            b = d[(d.setting == st) & (d.method == lead) & (d.k == k)].layers
            xa = V.per_sample(st, a.iloc[0]) if len(a) else None
            xb = V.per_sample(st, b.iloc[0]) if len(b) else None
            if xa is None or xb is None:
                miss += 1; continue
            cells.append((xa, xb))
        r = V.paired_cells(cells) if cells and not miss else None
        rows.append(dict(setting=st, lead=lead, cells_used=len(cells), missing=miss, **(r or {})))
    R = pd.DataFrame(rows); os.makedirs('results/paper', exist_ok=True); R.to_csv('results/paper/paired.csv', index=False)
    print(R.to_string())
    src = open('results/swk_report_nosleb/en/t_kavg.tex').read().split('\n')
    out, i = [], 0
    for line in src:
        if line.startswith('Benchmark &'):
            line = line.replace('& $z$ \\\\', '& $\\Delta$ pts [95\\,\\% CI] \\\\')
        elif line.endswith('\\\\') and '&' in line and not line.startswith('Mean') and i < len(ORDER):
            r = R.iloc[i]; i += 1
            cell = (f"${100 * r['diff']:+.1f}$ [${100 * r['lo']:.1f}$, ${100 * r['hi']:.1f}$]" if 'diff' in R.columns and r['diff'] == r['diff']
                    else '\\na')
            line = line[:line.rstrip('\\ ').rfind('&')] + f'& {cell} \\\\'
        out.append(line)
    open('paper_acml/tables/t_kavg.tex', 'w').write('\n'.join(out).replace('{rr@{}}', '{rc@{}}'))


if __name__ == '__main__':
    main()
