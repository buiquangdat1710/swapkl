"""
In ra tập tầng top-k của một thước đo, đọc từ results/layer_importance.csv.

    python nfat/pick_layers.py --tag 8b-w256 --scorer ig_greedy --k 8   ->  2,4,5,15,...
Với ig_greedy, điểm = -(thứ tự chọn), nên top-k chính là tập tham lam tại bước k.
Nếu một (tag, thước đo) được ghi nhiều lần thì lấy lần GẦN NHẤT.
"""
import argparse
import sys

import pandas as pd


def pick(path, tag, scorer, k):
    d = pd.read_csv(path)
    d = d[(d.tag == tag) & (d.scorer == scorer)]
    if d.empty:
        sys.exit(f'không có ({tag}, {scorer}) trong {path}')
    d = d.drop_duplicates(subset=['layer'], keep='last')
    order = d.sort_values('score', ascending=False).layer.astype(int).tolist()
    if k > len(order) or (scorer == 'ig_greedy' and d.score.min() <= -1e8
                          and k > int((d.score > -1e8).sum())):
        sys.exit(f'k={k} vượt quá số tầng đã xếp hạng cho ({tag}, {scorer})')
    return sorted(order[:k])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--importance', default='results/layer_importance.csv')
    p.add_argument('--tag', required=True)
    p.add_argument('--scorer', required=True)
    p.add_argument('--k', type=int, required=True)
    a = p.parse_args()
    print(','.join(map(str, pick(a.importance, a.tag, a.scorer, a.k))))
