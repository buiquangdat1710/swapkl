"""In các đối số "tag=layers" cho run_niah_selected.sh: mọi baseline (+ ig) ở R10–R12, để đánh giá lại với PYTHONHASHSEED=0
(bộ đánh giá NIAH dùng hash(variant) làm seed -> mỗi tiến trình một tập test khác nếu không cố định).
    python nfat/perbench_niah_sets.py llama3.1-8b 8 8b-w512
"""
import subprocess
import sys

import pandas as pd

model, k, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = pd.read_csv('results/winall/layer_sets.csv')
d = d[(d.model == model) & (d.k == k) & (d.benchmark == 'NIAH_indist')]
out, seen = [], set()
for r in d.itertuples():
    if r.method in ('duo', 'rethead', 'cake', 'moa', 'bi', 'sleb', 'ig_greedy', 'ig') and r.layers not in seen:
        seen.add(r.layers); out.append(f'hs0-{tag}-{r.method}-k{k}={r.layers}')
try:
    ig = subprocess.run([sys.executable, 'nfat/pick_layers.py', '--tag', tag, '--scorer', 'ig', '--k', str(k)],
                        capture_output=True, text=True).stdout.strip().splitlines()[-1]
    if ig and ig not in seen:
        out.append(f'hs0-{tag}-ig-k{k}={ig}')
except Exception:
    pass
print(' '.join(out))
