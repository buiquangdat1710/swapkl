"""
Hướng D — bảo toàn PHẢN HỒI VỚI BẰNG CHỨNG XA (evidence-response preservation), idea-stage/novel/REVIEW_R1.md.

Cặp phản thực tế: từ mỗi mẫu hiệu chỉnh x (đáp án a chứa giá trị v có mặt trong ngữ cảnh), tạo x' bằng cách thay MỌI lần xuất
hiện của v trong ngữ cảnh bằng v' cùng loại (lấy từ mẫu khác cùng tác vụ, không có mặt trong x), đáp án a' = a với v -> v'.
Độ dài/vị trí/định dạng giữ nguyên (chỉ đổi token của v). t = token đầu tiên a và a' khác nhau (tiền tố chung a[:t]).
  m_S(x)  = log p_S(a_t | x, a[:t]) - log p_S(a'_t | x, a[:t])          (log-odds giữa hai đáp án)
  e_S     = m_S(x) - m_S(x')                                             (phản hồi với bằng chứng)
  D(S)    = TB_cặp min(((e_S - e_T) / e_T)^2, 4)  trên các cặp có e_T >= TAU_T (thầy phản hồi tin cậy)
Mọi lượt xuôi chạy token đáp án đúng chế độ giải mã triển khai (như Obj). Một lượt xuôi mỗi prompt cho đồng thời KL (so với
thầy), NLL vàng và m — nên D, KL, NLL trên cùng prompt có CÙNG chi phí (đối chứng công bằng "KL trên dữ liệu tăng cường").

    python nfat/evresp.py pilot --setting ruler       # đo D/KL/NLL tại mọi tập tầng đã có điểm test trong quét k
    python nfat/evresp.py analyze                     # Spearman(điểm test, -thống kê)
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat import hyb                                       # noqa: E402

TAU_T = 2.0        # cặp chỉ dùng khi phản hồi của thầy e_T >= 2 nat (ghi trước khi chạy pilot)
CLIP = 4.0
OUT = 'results/evresp'
BAB_CAT = {  # thực thể bAbI theo loại: chỉ thay bằng giá trị CÙNG loại
    'loc': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office', 'cellar', 'library', 'garage', 'cinema', 'park', 'school'],
    'per': ['Mary', 'John', 'Sandra', 'Daniel', 'Fred', 'Bill', 'Jeff', 'Julie', 'Emma', 'Peter'],
    'obj': ['football', 'apple', 'milk', 'book', 'ball', 'box', 'key', 'pen']}


# ----------------------------------------------------------------------------- cặp phản thực tế
def _find(seq, sub):
    n = len(sub)
    return [i for i in range(len(seq) - n + 1) if seq[i:i + n] == sub] if n else []


def _variants(tok, v):
    return [tok(' ' + v, add_special_tokens=False).input_ids, tok(v, add_special_tokens=False).input_ids]


def _rand_like(v, rng):
    """Giá trị dự phòng cùng 'hình dạng': chữ số -> chữ số, chữ cái -> chữ cái (giữ hoa/thường, độ dài; hex giữ hex)."""
    hexy = all(ch in '0123456789abcdef-' for ch in v.lower())
    out = []
    for ch in v:
        if ch.isdigit():
            out.append(rng.choice('0123456789'))
        elif ch.isalpha():
            c = rng.choice('abcdef' if hexy else 'abcdefghijklmnopqrstuvwxyz')
            out.append(c.upper() if ch.isupper() else c)
        else:
            out.append(ch)
    return ''.join(out)


def make_pair(tok, it, pool, rng):
    """Trả dict cặp hoặc None. pool: các giá trị cùng tác vụ từ mẫu khác."""
    v = it.get('value')
    if not v:
        return None
    ids, gold = it['ids'], it['gold']
    vs = _variants(tok, v)
    which = next((j for j, s in enumerate(vs) if _find(ids, s)), None)
    gw = next((j for j, s in enumerate(vs) if _find(gold, s)), None)
    if which is None or gw is None:
        return None
    cands = [c for c in pool if c != v]
    rng.shuffle(cands)
    cat = next((ws for ws in BAB_CAT.values() if v in ws), None)
    if cat is not None:                  # thực thể bAbI: chỉ cùng loại, không sinh ngẫu nhiên
        cands = [c for c in cands if c in cat] + [c for c in cat if c not in cands]
    else:
        cands += [_rand_like(v, rng) for _ in range(8)]
    for v2 in cands:
        v2s = _variants(tok, v2)
        if any(_find(ids, s) for s in v2s) or v2.lower() == v.lower():
            continue
        # thay mọi lần xuất hiện của v trong ngữ cảnh (cả hai biến thể token hoá)
        new = list(ids)
        for j in (0, 1):
            src, dst = vs[j], v2s[j]
            pos = _find(new, src)
            for p in reversed(pos):
                new[p:p + len(src)] = dst
        if any(_find(new, s) for s in vs):
            continue
        g2 = list(gold)
        p = _find(g2, vs[gw])[0]
        g2[p:p + len(vs[gw])] = v2s[gw]
        t = next((i for i, (a, b) in enumerate(zip(gold, g2)) if a != b), None)
        if t is None:
            continue
        return dict(ids=ids, gold=gold, ids2=new, gold2=g2, t=t, task=it.get('task', ''), v=v, v2=v2,
                    n_occ=len(_find(ids, vs[which])))
    return None


def build_pairs(tok, items, seed=1234):
    rng = random.Random(seed + 7)
    by = {}
    for it in items:
        by.setdefault(it.get('task', ''), []).append(it.get('value'))
    pairs = []
    for it in items:
        pool = [x for x in by.get(it.get('task', ''), []) if x]
        pairs.append(make_pair(tok, it, pool, rng))
    return pairs


# ----------------------------------------------------------------------------- đo
class EvObj:
    def __init__(self, model, items, pairs, dev, W, S, decode, log=print):
        self.model, self.dev, self.W, self.S, self.decode, self.log = model, dev, W, S, decode, log
        self.items, self.pairs = items, pairs
        self.n_eval = 0
        hyb.set_mode('off')
        self.ref, self.eT = [], []
        with torch.no_grad():
            for i, pr in enumerate(pairs):
                if pr is None:
                    self.ref.append(None); self.eT.append(np.nan); continue
                l1 = self._lp(pr['ids'], pr['gold']); l2 = self._lp(pr['ids2'], pr['gold2'])
                self.ref.append((l1.cpu(), l2.cpu()))
                self.eT.append(self._e(l1, l2, pr))
        self.eT = np.array(self.eT)
        self.use = [i for i in range(len(pairs)) if pairs[i] is not None and self.eT[i] >= TAU_T]
        log(f'-> cặp: {sum(p is not None for p in pairs)}/{len(pairs)} dựng được, {len(self.use)} có e_T >= {TAU_T} '
            f'(e_T trung vị {np.nanmedian(self.eT):.2f})')

    def _lp(self, ids, gold):
        x = torch.tensor([ids + gold], device=self.dev)
        P, m = len(ids), len(gold)
        hyb.STATE['ans_full'] = self.decode == 'full'; hyb.STATE['P'] = P
        try:
            h = self.model.model(input_ids=x, use_cache=False).last_hidden_state[0, P - 1:P - 1 + m]
        finally:
            hyb.STATE['ans_full'] = False; hyb.STATE['P'] = None
        return torch.log_softmax(self.model.lm_head(h).float(), -1)

    @staticmethod
    def _e(l1, l2, pr):
        t, a, b = pr['t'], pr['gold'][pr['t']], pr['gold2'][pr['t']]
        return float((l1[t, a] - l1[t, b]) - (l2[t, a] - l2[t, b]))

    @torch.no_grad()
    def measure(self, S, idx=None):
        """Thống kê của tập tầng S trên các cặp idx (mặc định mọi cặp dùng được)."""
        idx = self.use if idx is None else idx
        hyb.set_mode('hard', self.W, self.S, full=S)
        rows = []
        try:
            for i in idx:
                pr = self.pairs[i]; r1, r2 = (z.to(self.dev) for z in self.ref[i])
                l1 = self._lp(pr['ids'], pr['gold']); l2 = self._lp(pr['ids2'], pr['gold2'])
                self.n_eval += 2
                e = self._e(l1, l2, pr); eT = self.eT[i]
                kl1 = float((r1.exp() * (r1 - l1)).sum(-1).mean()); kl2 = float((r2.exp() * (r2 - l2)).sum(-1).mean())
                g1 = torch.tensor(pr['gold'], device=self.dev); g2 = torch.tensor(pr['gold2'], device=self.dev)
                n1 = float(-l1.gather(-1, g1[:, None]).mean()); n2 = float(-l2.gather(-1, g2[:, None]).mean())
                t, a, b = pr['t'], pr['gold'][pr['t']], pr['gold2'][pr['t']]
                flip = float((l1[t, a] - l1[t, b]) <= 0) + float((l2[t, b] - l2[t, a]) <= 0)
                rows.append(dict(D=min(((e - eT) / eT) ** 2, CLIP), short=max(0.0, 1 - e / eT), e=e, flip=flip / 2,
                                 KL=kl1, KLaug=(kl1 + kl2) / 2, NLL=n1, NLLaug=(n1 + n2) / 2))
        finally:
            hyb.set_mode('off')
        return pd.DataFrame(rows).mean().to_dict() if rows else {}


# ----------------------------------------------------------------------------- phương pháp chọn tầng
def swapd(model, tok, gates, items, a, dev, log, mult=0.5):
    """SwapD: khung SwapKL (cùng 5 tập khởi đầu rẻ, cùng vòng hoán đổi, cùng chọn cuối trên B) nhưng mọi phép đo dùng D thay KL.
    Mỗi lần đo D tốn 2 lượt xuôi/cặp -> mult=0.5 giảm nửa số cặp ở mọi bước để GIỮ NGÂN SÁCH lượt xuôi như SwapKL
    (swapd); mult=1.0 (swapd1) dùng đủ số mẫu như SwapKL (gấp đôi chi phí, ablation). Không có cặp dùng được (LME) -> SwapKL."""
    from nfat import perbench_methods as PM
    HP = PM.HP
    k, L = a.k, hyb.n_layers(model)
    obj, A, B, ranks = PM.setup(model, gates, items, a, dev, log)
    pairs = build_pairs(tok, items)
    ev = EvObj(model, items, pairs, dev, a.W, a.S, a.decode, log)
    use = set(ev.use)
    Au, Bu = [i for i in A if i in use], [i for i in B if i in use]
    if len(Au) < 6 or len(Bu) < 4:
        log(f'   swapd: chỉ {len(Au)}/{len(Bu)} cặp A/B dùng được -> quay về SwapKL')
        return PM.swapkl(model, tok, gates, items, a, dev, log)
    n = lambda x: max(2, int(round(x * mult)))                                # noqa: E731
    J = lambda S, idx: ev.measure(list(S), idx)['D']                          # noqa: E731
    rot = lambda r, m: [Au[(r * m + j) % len(Au)] for j in range(m)]          # noqa: E731
    Aconf, Bconf = Au[:n(16)], Bu[:n(16)]
    inits = {nm: sorted(r[:k]) for nm, r in ranks.items()}
    inits['copy+cake'] = PM._interleave_sets([ranks['copy'], ranks['cake']], k)
    seen = {}
    for nm, S in inits.items():
        t = tuple(S)
        if t not in seen:
            seen[t] = J(S, Au[:n(HP['init_items'])])
        log(f'   swapd khởi đầu {nm} {S}: D {seen[t]:.4f}')
    S = list(min(seen, key=seen.get))
    full = {tuple(S): J(S, Aconf)}
    log(f'   -> khởi đầu {S}: D_A {full[tuple(S)]:.4f}')
    for r in range(HP['rounds']):
        sub = rot(r, n(HP['rm_items']))
        rm = {l: J([x for x in S if x != l], sub) for l in S}
        keep = sorted(rm, key=rm.get)[:HP['rm_keep']]
        sw = rot(r + 7, n(HP['swap_items']))
        cand = {}
        for l in keep:
            for o in range(L):
                if o not in S:
                    T = tuple(sorted([x for x in S if x != l] + [o]))
                    cand[T] = J(T, sw)
        top = sorted(cand, key=cand.get)[:HP['top_full']]
        for T in top:
            if T not in full:
                full[T] = J(T, Aconf)
        best = min(top, key=lambda T: full[T])
        log(f'   swapd vòng {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} D_A {full[best]:.4f} (hiện tại {full[tuple(S)]:.4f}) '
            f'| {ev.n_eval + obj.n_eval} lượt-mẫu')
        if full[best] >= full[tuple(S)]:
            break
        S = list(best)
    order = sorted(full, key=full.get)
    fin = [tuple(S)] + [T for T in order if T != tuple(S)][:1]
    jb = {T: J(T, Bconf) for T in fin}
    win = min(jb, key=jb.get)
    log('   swapd chọn cuối trên B: ' + '; '.join(f'{list(T)} {v:.4f}' for T, v in jb.items()) + f' -> {list(win)}')
    log(f'   swapd tổng {ev.n_eval + obj.n_eval} lượt xuôi-mẫu tại đỉnh (+{len(items)} + {2 * len(pairs)} tham chiếu, '
        f'+{2 * HP["n_grad"]} xuôi+ngược)')
    v = np.zeros(L)
    for pos, l in enumerate(ranks['kl']):
        v[l] = -pos * 1e-3
    v[list(win)] += 10.0
    return v, len(items)


def run(name, model, tok, gates, items, a, dev, log):
    if name == 'swapd':
        return swapd(model, tok, gates, items, a, dev, log, 0.5)
    if name == 'swapd1':
        return swapd(model, tok, gates, items, a, dev, log, 1.0)
    raise SystemExit(f'không biết {name}')


# ----------------------------------------------------------------------------- pilot
def load_setting(st):
    from nfat.ksweep import S as KS
    from nfat.perbench_data import bench_items
    c = KS[st]
    model, tok = hyb.load(c['model'], attn='hyb')
    model.eval()
    items = bench_items(tok, 'babilong' if c['bench'] == 'babilong2' else c['bench'], c['W'], 32, 1234)
    if c['bench'] == 'babilong2':           # đích = trả lời greedy của mô hình gốc (như perbench.py --bench babilong2)
        hyb.set_mode('off')
        with torch.no_grad():
            for it in items:
                out = model.generate(torch.tensor([it['ids']], device='cuda'), max_new_tokens=20, do_sample=False,
                                     pad_token_id=tok.eos_token_id)[0, len(it['ids']):].tolist()
                txt = tok.decode(out, skip_special_tokens=True).split('\n')[0]
                it['gold'] = tok(txt, add_special_tokens=False).input_ids or out[:1]
    return c, model, tok, items


def pilot(st):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f'{s}  [{time.time() - t0:.0f}s]', flush=True)   # noqa: E731
    c, model, tok, items = load_setting(st)
    pairs = build_pairs(tok, items)
    for pr in pairs[:3]:
        if pr:
            log(f'   ví dụ cặp [{pr["task"]}] {pr["v"]!r} -> {pr["v2"]!r} ({pr["n_occ"]} lần), t={pr["t"]}, '
                f'a={tok.decode(pr["gold"])!r} a\'={tok.decode(pr["gold2"])!r}')
    ev = EvObj(model, items, pairs, 'cuda', c['W'], c['Sk'], c['dec'], log)
    ks = pd.read_csv('results/perbench/ksweep.csv').drop_duplicates(['setting', 'method', 'k'], keep='last')
    ks = ks[ks.setting == st]
    out = f'{OUT}/pilot_{st}.csv'
    done = set(pd.read_csv(out, dtype={'layers': str}).layers) if os.path.exists(out) else set()
    sets = ks.drop_duplicates('layers')
    for r in sets.itertuples():
        if r.layers in done:
            continue
        S = [int(x) for x in str(r.layers).split(',')]
        m = ev.measure(S)
        row = dict(setting=st, layers=r.layers, k=len(S), n_pairs=len(ev.use), **m)
        pd.DataFrame([row]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        log(f'   k={len(S)} {S}: D {m["D"]:.3f} | KL {m["KL"]:.3f} | KLaug {m["KLaug"]:.3f} | lật {m["flip"]:.3f}')
    log(f'XONG pilot {st}: {len(sets)} tập, {ev.n_eval} lượt xuôi-mẫu')


def analyze():
    from scipy.stats import spearmanr
    ks = pd.read_csv('results/perbench/ksweep.csv', dtype={'layers': str}).drop_duplicates(['setting', 'method', 'k'], keep='last')
    sc = ks.groupby(['setting', 'layers']).score.mean().reset_index()
    rows = []
    for f in sorted(os.listdir(OUT)):
        if not f.startswith('pilot_') or f == 'pilot_spearman.csv':
            continue
        d = pd.read_csv(f'{OUT}/{f}', dtype={'layers': str}).merge(sc, on=['setting', 'layers'])
        st = d.setting.iloc[0]
        for stat in ['D', 'short', 'flip', 'KL', 'KLaug', 'NLL', 'NLLaug']:
            # gộp mọi k, và trong từng k (bỏ k có điểm hằng) rồi lấy TB
            rho_all = spearmanr(d.score, -d[stat]).correlation
            per = []
            for k, g in d.groupby('k'):
                if len(g) >= 4 and g.score.nunique() > 1 and g[stat].nunique() > 1:
                    per.append(spearmanr(g.score, -g[stat]).correlation)
            rows.append(dict(setting=st, stat=stat, rho_all=rho_all, rho_within_k=np.mean(per) if per else np.nan,
                             n_k=len(per), n_sets=len(d)))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R.pivot_table(index='stat', columns='setting', values='rho_within_k').round(3).to_string())
    print(R.pivot_table(index='stat', columns='setting', values='rho_all').round(3).to_string())
    R.to_csv(f'{OUT}/pilot_spearman.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['pilot', 'analyze', 'pairs'])
    p.add_argument('--setting', default='')
    a = p.parse_args()
    if a.cmd == 'pilot':
        for st in a.setting.split(','):
            pilot(st)
    elif a.cmd == 'analyze':
        analyze()
    else:                                    # kiểm tra dựng cặp (CPU, chỉ tokenizer)
        from transformers import AutoTokenizer
        from nfat.ksweep import S as KS
        from nfat.perbench_data import bench_items
        for st in a.setting.split(','):
            c = KS[st]
            tok = AutoTokenizer.from_pretrained(c['model'])
            items = bench_items(tok, 'babilong' if c['bench'] == 'babilong2' else c['bench'], c['W'], 32, 1234)
            pairs = build_pairs(tok, items)
            ok = [p for p in pairs if p]
            print(f'{st}: {len(ok)}/{len(pairs)} cặp')
            import collections
            print('  theo tác vụ:', dict(collections.Counter(p['task'] for p in ok)),
                  '| thiếu:', dict(collections.Counter(it.get('task', '') for it, p in zip(items, pairs) if not p)))
            for pr in ok[:4]:
                print(f'  [{pr["task"]}] {pr["v"]!r} -> {pr["v2"]!r} ×{pr["n_occ"]}, t={pr["t"]}: '
                      f'{tok.decode(pr["gold"])!r} | {tok.decode(pr["gold2"])!r}')
