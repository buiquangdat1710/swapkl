"""
SwapKL v2 — thống kê cho so sánh chọn tầng (định nghĩa: results/swk_v2/LOG.md mục 0, ghi 10-12 16:30 trước mọi phân tích).

  - vùng k QUYẾT ĐỊNH: max_6BL(k) − SWA ≥ 0,05 và Gốc − max_6BL(k) ≥ 0,02 (chỉ từ baseline)
  - BL tốt nhất cho T1: baseline có TB cao nhất trên các ô quyết định
  - điểm TỪNG MẪU của mọi bộ đánh giá: NIAH (results/niah.<tag>.details.csv), LME (results/lme/<name>.judged.parquet),
    S-NIAH/RULER/BABILong (results/details/{ruler_hf,ruler_vllm,babilong}/*.csv, thêm 10-12)
  - kiểm định ghép cặp theo CỤM (cụm = prompt test; cùng prompt ở nhiều k là một cụm): hoán vị dấu theo cụm (p hai phía),
    bootstrap cụm phân tầng theo tác vụ (KTC 95 %); Holm
  - k@q = k nhỏ nhất (trên lưới) có điểm ≥ q × Gốc

    python nfat/swk_v2_stats.py decisive                    # vùng quyết định + BL tốt nhất (chỉ baseline)
    python nfat/swk_v2_stats.py t1 --method swapkl          # T1 (ghép cặp nếu có dữ liệu từng mẫu) cho một phương pháp
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat.ksweep import S, norm                      # noqa: E402
# = nfat/ksweep_plot.REF (L, điểm SWA k=0, điểm mô hình gốc); chép lại để không phải nạp matplotlib trong môi trường GPU (swaa)
REF = {'niah1b': (16, .20, 1.0), 'sniah512': (32, .16, 1.0), 'sniah256': (32, .08, 1.0), 'ruler': (32, .197, .983),
       'bab': (32, (.300 + .204) / 2, (.636 + .572) / 2), 'niah8b': (32, .20, 1.0), 'lme_l8i': (32, .208, .604),
       'lme_q4t': (36, .528, .788)}

BL = ['duo', 'rethead', 'cake', 'moa', 'bi', 'sleb']
ORDER = ['niah1b', 'sniah512', 'sniah256', 'ruler', 'bab', 'niah8b', 'lme_l8i', 'lme_q4t']
T1_SET = [s for s in ORDER if s != 'niah1b']           # NIAH 1B bỏ khỏi T1 (trần, W14)
STD_GRID = {'niah1b': [1, 2, 3, 4, 6, 8, 12], 'lme_l8i': [4, 8, 16], 'lme_q4t': [4, 9, 18]}
STD_GRID.update({s: [1, 2, 4, 6, 8, 12, 16, 24] for s in ['sniah512', 'sniah256', 'ruler', 'bab', 'niah8b']})
KSWEEP = os.path.join(ROOT, 'results/perbench/ksweep.csv')
NIAH_MAP = os.path.join(ROOT, 'results/perbench/ksweep_niah_map.csv')
SNIAH_TASKS = ['S-NIAH-PK', 'S-NIAH-N', 'S-NIAH-W']


def snorm(x):
    """norm() an toàn: NaN/rỗng -> ''."""
    if not isinstance(x, str) or not x.strip():
        return ''
    try:
        return norm(x)
    except ValueError:
        return ''


def load_ksweep(path=KSWEEP):
    d = pd.read_csv(path, dtype={'layers': str}).drop_duplicates(['setting', 'method', 'k'], keep='last')
    d['layers'] = d.layers.map(snorm)
    return d


def ref(st):
    L, swa, full = REF[st]
    return dict(L=L, swa=swa, full=full)


# ----------------------------------------------------------------------------- vùng quyết định / BL tốt nhất
GRID = 'std'   # 10-12 18:58: người dùng DỪNG lưới k dày (ksweep_dense) để dành GPU cho v2 -> vùng quyết định/T1/T3 dùng LƯỚI CHUẨN


def decisive(d=None, a_min=0.05, b_min=0.02, grid=None):
    """DataFrame (setting, k, maxBL, argmaxBL, swa, full, decisive) trên mọi k có đủ 6 baseline (grid='std': chỉ k của lưới chuẩn)."""
    d = load_ksweep() if d is None else d
    grid = grid or GRID
    rows = []
    for st in ORDER:
        g = d[(d.setting == st) & d.method.isin(BL)]
        if grid == 'std':
            g = g[g.k.isin(STD_GRID[st])]
        r = ref(st)
        for k, h in g.groupby('k'):
            if h.method.nunique() < len(BL):
                continue
            i = h.score.idxmax()
            mx = float(h.score.max())
            rows.append(dict(setting=st, k=int(k), maxBL=mx, argmaxBL=h.loc[i, 'method'], swa=r['swa'], full=r['full'],
                             decisive=bool(mx - r['swa'] >= a_min - 1e-12 and r['full'] - mx >= b_min - 1e-12)))
    return pd.DataFrame(rows)


def best_bl(d=None, dec=None):
    """{setting: (baseline, TB trên ô quyết định, [k quyết định])} — CHỈ dùng số baseline."""
    d = load_ksweep() if d is None else d
    dec = decisive(d) if dec is None else dec
    out = {}
    for st in ORDER:
        ks = dec[(dec.setting == st) & dec.decisive].k.tolist()
        if not ks:
            out[st] = (None, np.nan, [])
            continue
        g = d[(d.setting == st) & d.method.isin(BL) & d.k.isin(ks)]
        m = g.groupby('method').score.mean()
        m = m[g.groupby('method').k.nunique() == len(ks)]        # chỉ baseline có đủ mọi ô quyết định
        out[st] = (m.idxmax(), float(m.max()), ks)
    return out


def k_at(scores, full, q):
    """scores: {k: điểm}. k nhỏ nhất có điểm ≥ q·full (None nếu không có)."""
    for k in sorted(scores):
        if scores[k] >= q * full - 1e-12:
            return int(k)
    return None


# ----------------------------------------------------------------------------- điểm từng mẫu
_DET_CACHE = {}


def _details(kind):
    if kind not in _DET_CACHE:
        fs = sorted(glob.glob(os.path.join(ROOT, 'results', 'details', kind, '*.csv')))
        _DET_CACHE[kind] = (pd.concat([pd.read_csv(f, dtype={'layers': str}) for f in fs], ignore_index=True)
                            if fs else pd.DataFrame())
        if len(_DET_CACHE[kind]):
            _DET_CACHE[kind]['layers'] = _DET_CACHE[kind].layers.map(snorm)
    return _DET_CACHE[kind]


def per_sample(st, layers, seed=777, n_ruler=25):
    """Điểm từng mẫu của tập `layers` ở benchmark st (giao thức quét k). DataFrame(cluster, stratum, score) hoặc None."""
    c = S[st]; ev, W = c['ev'], c['W']; ly = snorm(layers)
    if ev in ('niah8b', 'niah1b'):
        if not os.path.exists(NIAH_MAP):
            return None
        mp = pd.read_csv(NIAH_MAP)
        t = mp[(mp.ev == ev) & (mp.layers.map(snorm) == ly)]
        for tag in reversed(t.tag.tolist()):
            f = os.path.join(ROOT, f'results/niah.{tag}.details.csv')
            if os.path.exists(f):
                x = pd.read_csv(f)
                if len(x) == 300:
                    return pd.DataFrame(dict(cluster=np.arange(len(x)).astype(str),
                                             stratum=x.variant.astype(str) + '|' + x.length.astype(str), score=x.score.astype(float)))
        return None
    if ev.startswith('lme:'):
        pre = ev.split(':')[1]
        lm = pd.read_csv(os.path.join(ROOT, 'results/lme.csv'))
        lm = lm[lm.name.str.startswith(pre + '_') & (lm.layers.map(snorm) == ly) & (lm.n == 500)]
        for name in reversed(lm.name.tolist()):
            f = os.path.join(ROOT, f'results/lme/{name}.judged.parquet')
            if os.path.exists(f):
                x = pd.read_parquet(f, columns=['question_type', 'judge'])
                if len(x) == 500:
                    return pd.DataFrame(dict(cluster=np.arange(len(x)).astype(str), stratum=x.question_type.astype(str),
                                             score=x.judge.astype(float)))
        return None
    if ev == 'sniah4k':
        x = _details('ruler_hf')
        if not len(x):
            return None
        x = x[(x.W.astype(str) == str(W)) & (x.layers == ly) & (x.length == 4096) & x.task.isin(SNIAH_TASKS) & (x.seed == seed)]
        x = x.drop_duplicates(['task', 'i'], keep='last')
        if x.groupby('task').size().reindex(SNIAH_TASKS).fillna(0).min() < 50:
            return None
        return pd.DataFrame(dict(cluster=(x.task + '|' + x.i.astype(str)).values, stratum=x.task.values, score=x.score.values))
    if ev == 'ruler8k':
        x = _details('ruler_vllm')
        if not len(x):
            return None
        x = x[(x.W == W) & (x.layers == ly) & (x.length == 8192) & (x.seed == seed)]
        x = x.drop_duplicates(['task', 'i'], keep='last')
        cnt = x.groupby('task').size()
        if len(cnt) < 11 or cnt.min() < n_ruler:
            return None
        x = x[x.i < n_ruler]
        return pd.DataFrame(dict(cluster=(x.task + '|' + x.i.astype(str)).values, stratum=x.task.values, score=x.score.values))
    if ev == 'bab24':
        x = _details('babilong')
        if not len(x):
            return None
        x = x[(x.W.astype(str) == str(W)) & (x.layers == ly) & x.length.isin(['2k', '4k'])]
        x = x.drop_duplicates(['task', 'length', 'i'], keep='last')
        if len(x.groupby(['task', 'length'])) < 10 or x.groupby(['task', 'length']).size().min() < 50:
            return None
        x = x[x.i < 50]
        key = x.task + '|' + x.length
        return pd.DataFrame(dict(cluster=(key + '|' + x.i.astype(str)).values, stratum=key.values, score=x.score.values))
    return None


# ----------------------------------------------------------------------------- kiểm định ghép cặp theo cụm
def paired_cells(cells, n_perm=20000, n_boot=5000, seed=0):
    """cells: list of (xa, xb) = DataFrame(cluster, stratum, score) của hai phương pháp ở cùng ô.
    Thống kê T = TB theo ô của TB hiệu (a − b) trên mẫu chung. Trả dict(diff, p, lo, hi, n_clusters, n_cells, z_eff)."""
    rng = np.random.default_rng(seed)
    per = []
    for xa, xb in cells:
        m = xa.merge(xb, on=['cluster', 'stratum'], suffixes=('_a', '_b'))
        if not len(m):
            continue
        m['w'] = (m.score_a - m.score_b) / len(m)
        per.append(m[['cluster', 'stratum', 'w']])
    if not per:
        return None
    nc = len(per)
    allm = pd.concat(per)
    D = allm.groupby(['cluster', 'stratum']).w.sum() / nc        # đóng góp của mỗi cụm vào T
    T = float(D.sum())
    v = D.values
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(v)))
    Tp = signs @ v
    p = float((1 + np.sum(np.abs(Tp) >= abs(T) - 1e-12)) / (1 + n_perm))
    # bootstrap cụm phân tầng (stratum của cụm)
    strata = D.index.get_level_values('stratum').values
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    bs = np.empty(n_boot)
    for b in range(n_boot):
        tot = 0.0
        for gi in groups:
            tot += v[rng.choice(gi, size=len(gi), replace=True)].sum()
        bs[b] = tot
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sd = float(Tp.std()) if Tp.std() > 0 else np.nan
    return dict(diff=T, p=p, lo=float(lo), hi=float(hi), n_clusters=int(len(v)), n_cells=nc, z_eff=T / sd if sd else np.nan)


def holm(ps):
    """p đã hiệu chỉnh Holm (cùng thứ tự đầu vào; None giữ nguyên)."""
    idx = [i for i, p in enumerate(ps) if p is not None and not np.isnan(p)]
    order = sorted(idx, key=lambda i: ps[i])
    m = len(order); adj = list(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def unpaired_z(a, b, n):
    """z nhị thức không ghép cặp (thô, như báo cáo v1.1) cho hiệu hai tỉ lệ cùng n."""
    se = np.sqrt(a * (1 - a) / n + b * (1 - b) / n)
    return (a - b) / se if se > 0 else 0.0


def t1(method='swapkl', d=None, verbose=True, seed_test=777):
    """T1 cho `method`: theo benchmark, hiệu ghép cặp với BL tốt nhất trên ô quyết định; Holm trên 7 benchmark."""
    d = load_ksweep() if d is None else d
    bb = best_bl(d)
    rows = []
    for st in T1_SET:
        bl, blm, ks = bb[st]
        cells, miss = [], []
        for k in ks:
            ra = d[(d.setting == st) & (d.method == method) & (d.k == k)]
            rb = d[(d.setting == st) & (d.method == bl) & (d.k == k)]
            if not len(ra) or not len(rb):
                miss.append((k, 'thiếu điểm')); continue
            xa, xb = per_sample(st, ra.layers.iloc[0], seed_test), per_sample(st, rb.layers.iloc[0], seed_test)
            if xa is None or xb is None:
                miss.append((k, 'thiếu từng mẫu' + (' (a)' if xa is None else '') + (' (b)' if xb is None else '')))
                continue
            cells.append((xa, xb))
        res = paired_cells(cells) if cells else None
        g = d[(d.setting == st) & d.k.isin(ks)]
        ma = g[g.method == method].score.mean(); mb = g[g.method == bl].score.mean()
        rows.append(dict(setting=st, bl=bl, ks=ks, mean_m=ma, mean_bl=mb, n_cells=len(ks), n_paired=len(cells), missing=miss,
                         **({k: res[k] for k in ('diff', 'p', 'lo', 'hi', 'n_clusters', 'z_eff')} if res else {})))
    R = pd.DataFrame(rows)
    if 'p' in R:
        full = R.n_paired == R.n_cells
        R['p_holm'] = holm([p if f else None for p, f in zip(R.p, full)])
    if verbose:
        pd.set_option('display.width', 250)
        print(R.to_string())
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['decisive', 't1'])
    ap.add_argument('--method', default='swapkl')
    a = ap.parse_args()
    if a.cmd == 'decisive':
        dec = decisive()
        pd.set_option('display.width', 200)
        print(dec[dec.decisive].to_string())
        for st, v in best_bl(dec=dec).items():
            print(st, v)
    else:
        t1(a.method)


# ----------------------------------------------------------------------------- thiết kế test GỘP (đăng ký trước 10-12)
EVALX_NIAH = os.path.join(ROOT, 'results/swk_v2/evalx_niah.csv')


def per_sample_x(st, layers, part='pooled'):
    """Điểm từng mẫu trên thiết kế test gộp: part='orig' (seed 777 / BABILong mẫu 0–49 / LME), 'fresh' (seed 555 / BABILong 50–99),
    'pooled' = orig ∪ fresh (cụm phân biệt theo phần). S-NIAH và RULER dùng 50 mẫu/tác vụ mỗi seed. None nếu thiếu."""
    c = S[st]; ev, W = c['ev'], c['W']; ly = snorm(layers)
    if part == 'confirm':     # phân tích XÁC NHẬN (sửa đổi ghi trước 10-12 21:55 sau tấn công A3): test MỚI; LME không có test mới -> test duy nhất
        part = 'orig' if ev.startswith('lme:') else 'fresh'
    if part == 'confirm2':    # nhân bản (đăng ký 10-13): mới 555 ∪ 556 (S-NIAH, RULER, NIAH 8B); BAB mẫu 50–99; LME test duy nhất
        part = 'orig' if ev.startswith('lme:') else ('fresh' if ev == 'bab24' else 'fresh+rep')
    parts = {'pooled': ['orig', 'fresh'], 'fresh+rep': ['fresh', 'rep']}.get(part, [part])
    outs = []
    for pt in parts:
        x = None
        if ev == 'sniah4k':
            d = _details('ruler_hf')
            if len(d):
                s = {'orig': 777, 'fresh': 555, 'rep': 556}[pt]
                d = d[(d.W.astype(str) == str(W)) & (d.layers == ly) & (d.length == 4096) & d.task.isin(SNIAH_TASKS) & (d.seed == s)]
                d = d.drop_duplicates(['task', 'i'], keep='last'); d = d[d.i < 50]
                if len(d) == 150:
                    x = pd.DataFrame(dict(cluster=(f'{s}|' + d.task + '|' + d.i.astype(str)).values, stratum=d.task.values, score=d.score.values))
        elif ev == 'ruler8k':
            d = _details('ruler_vllm')
            if len(d):
                s = {'orig': 777, 'fresh': 555, 'rep': 556}[pt]
                d = d[(d.W == W) & (d.layers == ly) & (d.length == 8192) & (d.seed == s)]
                d = d.drop_duplicates(['task', 'i'], keep='last'); d = d[d.i < 50]
                if len(d) == 550:
                    x = pd.DataFrame(dict(cluster=(f'{s}|' + d.task + '|' + d.i.astype(str)).values, stratum=d.task.values, score=d.score.values))
        elif ev == 'bab24':
            d = _details('babilong')
            if len(d):
                d = d[(d.W.astype(str) == str(W)) & (d.layers == ly) & d.length.isin(['2k', '4k'])]
                d = d.drop_duplicates(['task', 'length', 'i'], keep='last')
                d = d[(d.i < 50)] if pt == 'orig' else d[(d.i >= 50) & (d.i < 100)]
                if len(d) == 500:
                    key = d.task + '|' + d.length
                    x = pd.DataFrame(dict(cluster=(f'{pt}|' + key + '|' + d.i.astype(str)).values, stratum=key.values, score=d.score.values))
        elif ev in ('niah8b', 'niah1b'):
            if pt == 'orig':
                x = per_sample(st, ly)
                if x is not None:
                    x = x.assign(cluster='777|' + x.cluster)
            else:
                sd = 556 if pt == 'rep' else 555
                f = EVALX_NIAH.replace('.csv', f'.x{sd}-{st}-{ly.replace(",", "_")}.details.csv')
                if os.path.exists(f):
                    y = pd.read_csv(f)
                    if len(y) == 300:
                        x = pd.DataFrame(dict(cluster=f'{sd}|' + pd.Series(range(len(y))).astype(str),
                                              stratum=y.variant.astype(str) + '|' + y.length.astype(str), score=y.score.astype(float)))
        elif ev.startswith('lme:'):
            x = per_sample(st, ly) if pt == 'orig' else None
            if x is None and pt == 'fresh':
                continue                                   # LME không có test mới: phần 'fresh' bỏ qua
        if x is None:
            return None
        outs.append(x)
    return pd.concat(outs, ignore_index=True) if outs else None


def t1x(method='swapkl', part='pooled', d=None, v2=None, verbose=True):
    """T1 trên thiết kế test gộp/mới (ô quyết định, BL tốt nhất chỉ từ baseline). v2: DataFrame ksweep_v2 (cho swapkl2*)."""
    d = load_ksweep() if d is None else d
    bb = best_bl(d)
    rows = []
    for st in T1_SET:
        bl, _, ks = bb[st]
        cells, miss = [], []
        for k in ks:
            src = v2 if (method.startswith(('swapkl2', 'swapkl3')) and v2 is not None) else d
            ra = src[(src.setting == st) & (src.method == method) & (src.k == k)]
            rb = d[(d.setting == st) & (d.method == bl) & (d.k == k)]
            if not len(ra) or not len(rb):
                miss.append((k, 'thiếu tập')); continue
            xa, xb = per_sample_x(st, ra.layers.iloc[0], part), per_sample_x(st, rb.layers.iloc[0], part)
            if xa is None or xb is None:
                miss.append((k, 'thiếu từng mẫu' + (' (a)' if xa is None else '') + (' (b)' if xb is None else ''))); continue
            cells.append((xa, xb))
        res = paired_cells(cells) if cells else None
        rows.append(dict(setting=st, bl=bl, ks=ks, part=part, n_cells=len(ks), n_paired=len(cells), missing=miss,
                         **({k: res[k] for k in ('diff', 'p', 'lo', 'hi', 'n_clusters', 'z_eff')} if res else {})))
    R = pd.DataFrame(rows)
    if 'p' in R:
        full = R.n_paired == R.n_cells
        R['p_holm'] = holm([p if f else None for p, f in zip(R.p, full)])
        R['sig_win'] = (R.p_holm < 0.05) & (R['diff'] > 0)
        R['sig_loss'] = (R.p_holm < 0.05) & (R['diff'] < 0)
    if verbose:
        pd.set_option('display.width', 250)
        print(R.to_string())
    return R
