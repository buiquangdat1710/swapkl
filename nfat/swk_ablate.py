"""
Ablation SwapKL v1.1 (yêu cầu 10-14: chứng minh từng thành phần cần thiết). MỘT lượt chạy cho mỗi (benchmark, k) chạy NHIỀU biến thể
trên CÙNG lượt thầy, cùng dữ liệu hiệu chỉnh và cùng chia A/B; mọi phép đo J(tập, mẫu) được lưu đệm và dùng chung giữa các biến thể
(tái dùng tiền tố, chính xác). Biến thể nào ra cùng quyết định với v1.1 thì gần như không tốn thêm.

Hai phần của SwapKL:
  Phần 1 — tập khởi đầu rẻ: copy, CAKE, grad-KL, grad-MSE (+ xen kẽ copy∪CAKE).
  Phần 2 — tìm kiếm: đua khởi đầu (J8) -> xác nhận J16 -> ≤3 vòng hoán đổi (đo hại khi bỏ, sàng, xác nhận) -> chọn cuối trên B.

Biến thể (tên scorer 'swkab_<tên>'; 'swkab' = v11, phải trùng tập của `swapkl`):
  TẦNG 1  v11 | loo_<x> (bỏ một khởi đầu x ∈ copy, cake, kl, mse, ilv) | only_<x> (chỉ một khởi đầu x ∈ copy, cake, kl, mse)
          noB (không giữ B: chọn cuối = tập hiện tại) | merged (một tập 32 mẫu cho mọi bước, không có B)
          nosearch (khởi đầu tốt nhất, không hoán đổi) | r1 (tối đa 1 vòng)
  TẦNG 2  only_rand (k tầng ngẫu nhiên) | only_per (tuần hoàn mỗi L/k tầng) | only_nllg (top-k NLL-guided) | only_ilv
          randrm (tầng đem đổi chọn NGẪU NHIÊN thay vì đo hại) | noconfirm (nhận ứng viên sàng tốt nhất, không xác nhận 16 mẫu)
--extra tier=1|2 (2 = tầng 1 + tầng 2), nllg=<tag có điểm nll_swa>.
"""
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER1 = ['v11', 'loo_copy', 'loo_cake', 'loo_kl', 'loo_mse', 'loo_ilv', 'only_copy', 'only_cake', 'only_kl', 'only_mse',
         'noB', 'merged', 'nosearch', 'r1']
TIER2 = ['only_rand', 'only_per', 'only_nllg', 'only_ilv', 'randrm', 'noconfirm']
TIER4 = ['r5']              # 10-16 (phản biện vòng 2, M9): tối đa 5 vòng hoán đổi (S-NIAH256 chạm giới hạn 3 vòng ở 4/8 ô)
TIER3 = ['nllobj']          # 10-16 (phản biện OE-Agent): cùng tìm kiếm, mục tiêu NLL đáp án thay KL; kèm đo J(A), J(B) của mọi tập đã đánh giá
START_NAMES = ['copy', 'cake', 'kl', 'mse', 'copy+cake']       # thứ tự chèn của v1.1 (quan trọng khi hoà J8)
ALIAS = {'ilv': 'copy+cake'}


def search(M, starts, A, B, k, L, rounds=3, remove='measured', confirm=True, rng=None, hp=None, key='kl'):
    """Đúng SwapKL v1.1 (perbench_methods.swapkl) với các công tắc ablation. M: Meter (đệm theo (tập, mẫu)).
    Trả (tập cuối, vết, số cặp (tập, mẫu) biến thể này tự yêu cầu = chi phí nếu chạy riêng, theo lượt xuôi-mẫu)."""
    hp = hp or dict(init_items=8, rm_items=8, rm_keep=2, swap_items=4, top_full=2)
    req = set()

    def J(base, cands, idx):
        cands = [tuple(sorted(T)) for T in cands]
        req.update((T, i) for T in cands for i in idx)
        return M.mean(base, cands, idx, key=key)

    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731
    uniq = list(dict.fromkeys(tuple(sorted(S)) for S in starts.values()))
    j8 = J(uniq[0], uniq, A[:hp['init_items']])
    S = min(uniq, key=j8.get)
    full = {S: J(S, [S], A)[S]}
    tr = dict(starts={n: list(T) for n, T in starts.items()}, J8={str(list(T)): j8[T] for T in uniq}, start=list(S), rounds=[])
    keep_n = min(hp['rm_keep'], k)
    for r in range(rounds):
        if k > 1:
            if remove == 'random':
                keep = sorted(int(x) for x in rng.choice(list(S), keep_n, replace=False))
            else:
                rmc = [tuple(x for x in S if x != l) for l in S]
                rm = J(S, rmc, rot(r, hp['rm_items']))
                dmg = {l: rm[tuple(x for x in S if x != l)] for l in S}
                keep = sorted(dmg, key=dmg.get)[:keep_n]
        else:
            keep = list(S)
        cands = list(dict.fromkeys(tuple(sorted([x for x in S if x != l] + [o])) for l in keep for o in range(L) if o not in S))
        sw = rot(r + 7, hp['swap_items'])
        sc = J(S, cands, sw)
        top = sorted(cands, key=sc.get)[:hp['top_full']]
        if confirm:
            new = [T for T in top if T not in full]
            if new:
                full.update(J(S, new, A))
            best = min(top, key=lambda T: full[T])
            improve = full[best] < full[S]
        else:
            best = top[0]
            cur = J(S, [S], sw)[S]
            improve = sc[best] < cur
            if improve:
                full[best] = J(best, [best], A)[best]
        tr['rounds'].append(dict(round=r + 1, keep=[int(x) for x in keep], best=list(best), improve=bool(improve)))
        if not improve:
            break
        S = best
    if B is not None:
        order = sorted(full, key=full.get)
        fin = [S] + [T for T in order if T != S][:1]
        jb = J(S, fin, B)
        win = min(fin, key=jb.get)
        tr['final'] = {str(list(T)): jb[T] for T in fin}
    else:
        win = S
    tr['out'] = list(win)
    return list(win), tr, len(req)


def run(name, model, tok, gates, items, a, dev, log):
    import torch
    from nfat import hyb
    from nfat import perbench_methods as PM
    from nfat.csv_append import append_row
    from nfat.swapkl_v2 import Meter, PrefixEval
    from nfat.win80_methods import PrefixX
    t0 = time.time()
    L, k = hyb.n_layers(model), a.k
    tier = int(a.extra.get('tier', 1))
    variants = {3: TIER3, 4: TIER4}.get(tier) or TIER1 + (TIER2 if tier >= 2 else [])
    obj, A, B, ranks = PM.setup(model, gates, items, a, dev, log)
    M = Meter(PrefixEval(PrefixX(obj)))
    inits = {n: tuple(sorted(r[:k])) for n, r in ranks.items()}           # copy, cake, kl, mse (thứ tự của setup)
    inits = {n: inits[n] for n in ['kl', 'mse', 'copy', 'cake']}           # = thứ tự dict ranks trong setup (kl, mse, copy, cake)
    inits['copy+cake'] = tuple(PM._interleave_sets([ranks['copy'], ranks['cake']], k))
    rng = np.random.RandomState(1000 + k)
    extra = {'rand': tuple(sorted(int(x) for x in rng.choice(L, k, replace=False))), 'per': tuple(sorted(hyb.periodic(L, k)))}
    if 'only_nllg' in variants:
        li = pd.read_csv(os.path.join(ROOT, 'results/layer_importance.csv'), usecols=['tag', 'scorer', 'layer', 'score'])
        g = li[(li.tag == a.extra.get('nllg', '')) & (li.scorer == 'nll_swa')].drop_duplicates('layer', keep='last')
        if len(g) == L:
            extra['nllg'] = tuple(sorted(int(x) for x in g.sort_values('score', ascending=False).layer[:k]))
        else:
            log(f'   !! không có điểm nll_swa cho {a.extra.get("nllg")} -> bỏ only_nllg')
            variants = [v for v in variants if v != 'only_nllg']
    merged = sorted(A + B)
    res, cost = {}, {}
    for v in variants:
        kw = dict(rounds=3)
        St, Av, Bv = dict(inits), A, B
        if v.startswith('loo_'):
            St.pop(ALIAS.get(v[4:], v[4:]))
        elif v.startswith('only_'):
            x = v[5:]
            St = {x: extra[x]} if x in extra else {ALIAS.get(x, x): inits[ALIAS.get(x, x)]}
        elif v == 'noB':
            Bv = None
        elif v == 'merged':
            Av, Bv = merged, None
        elif v == 'nosearch':
            kw['rounds'] = 0
        elif v == 'r1':
            kw['rounds'] = 1
        elif v == 'r5':
            kw['rounds'] = 5
        elif v == 'randrm':
            kw.update(remove='random', rng=np.random.RandomState(2000 + k))
        elif v == 'noconfirm':
            kw['confirm'] = False
        elif v == 'nllobj':
            kw['key'] = 'nll'
        fe0 = M.fe
        out, tr, nreq = search(M, St, Av, Bv, k, L, **kw)
        res[v] = dict(out=out, trace=tr, n_req=nreq, fe_marginal=M.fe - fe0)
        log(f'   swkab {v:10s}: {out} | tự chạy ≈ {nreq} lượt-mẫu | thêm {M.fe - fe0:.0f} lượt quy đổi (tổng {M.fe:.0f})')
    torch.cuda.synchronize()
    secs = time.time() - t0
    d = os.path.join(ROOT, 'results', 'swk_ablate', 'traces'); os.makedirs(d, exist_ok=True)
    json.dump(dict(tag=a.tag, k=k, L=L, W=a.W, decode=a.decode, A=A, B=B, secs=secs, fe_total=M.fe, variants=res,
                   ranks={n: r[:2 * k] for n, r in ranks.items()}), open(os.path.join(d, f'{a.tag}_k{k}' + (f'_t{tier}' if tier >= 3 else '') + '.json'), 'w'), indent=1, default=float)
    append_row(pd.DataFrame([dict(tag=a.tag, k=k, variant=v, n_req=r['n_req'], fe_marginal=r['fe_marginal'], secs_run=secs, fe_run=M.fe)
                             for v, r in res.items()]), os.path.join(ROOT, 'results', 'swk_ablate', 'cost.csv'))

    def vec(S):
        x = np.zeros(L)
        for pos, l in enumerate(ranks['kl']):
            x[l] = -pos * 1e-3
        x[list(S)] += 10.0
        return x
    for v, r in res.items():
        n = 'swkab' if v == 'v11' else f'swkab_{v}'
        if n == name:
            continue
        x = vec(r['out']); order = list(np.argsort(-x))
        append_row(pd.DataFrame([dict(tag=a.tag, model_config=a.model, scorer=n, layer=l, score=float(x[l]), rank=int(order.index(l)),
                                      window=a.W, n_sink=a.S, variant=a.bench, L=int(np.mean([len(it['ids']) for it in items])),
                                      n_items=len(items), seed=a.seed) for l in range(L)]), a.out)
    if tier >= 3:
        if tier == 3:
            jeval(M, a.extra.get('setting', ''), k, A, B, a.tag, log, extra_sets={f'swkab_{v}': r['out'] for v, r in res.items()})
        return vec(res[variants[0]]['out']), len(items)
    return vec(res['v11']['out']), len(items)


def jeval(M, st, k, A, B, tag, log, extra_sets=None):
    """Đo J (KL khớp giải mã) và NLL đáp án trên A và B cho MỌI tập đã đánh giá test ở (st, k): 6 baseline + NLL-guided + SwapKL
    (results/perbench/ksweep.csv) và mọi biến thể ablation (ksweep_ablate.csv). Ghi results/swk_ablate/jeval.csv (một dòng / tập)."""
    from nfat.csv_append import append_row
    norm = lambda ly: tuple(sorted(int(x) for x in str(ly).split(',')))          # noqa: E731
    src = {}
    ks = pd.read_csv(os.path.join(ROOT, 'results/perbench/ksweep.csv'), dtype={'layers': str})
    for _, r in ks[(ks.setting == st) & (ks.k == k)].iterrows():
        src.setdefault(norm(r.layers), []).append(r.method)
    p = os.path.join(ROOT, 'results/swk_ablate/ksweep_ablate.csv')
    if os.path.exists(p):
        ab = pd.read_csv(p, dtype={'layers': str})
        for _, r in ab[(ab.setting == st) & (ab.k == k)].iterrows():
            src.setdefault(norm(r.layers), []).append('swkab_' + r.variant)
    for n, T in (extra_sets or {}).items():
        src.setdefault(tuple(sorted(T)), []).append(n)
    if not src:
        log(f'   !! jeval: không có tập nào cho {st} k{k}'); return
    sets = list(src)
    g = M.get(sets[0], sets, list(A) + list(B))
    rows = []
    for T in sets:
        v = g[T]; ka = [x['kl'] for x in v[:len(A)]]; kb = [x['kl'] for x in v[len(A):]]
        na = [x['nll'] for x in v[:len(A)]]; nb = [x['nll'] for x in v[len(A):]]
        rows.append(dict(setting=st, k=k, tag=tag, layers=','.join(map(str, T)), sources=';'.join(dict.fromkeys(src[T])),
                         J_A=float(np.mean(ka)), J_B=float(np.mean(kb)), NLL_A=float(np.mean(na)), NLL_B=float(np.mean(nb)),
                         J_B_items=json.dumps([round(float(x), 5) for x in kb])))
    append_row(pd.DataFrame(rows), os.path.join(ROOT, 'results', 'swk_ablate', 'jeval.csv'))
    log(f'   jeval {st} k{k}: {len(sets)} tập, fe tổng {M.fe:.0f}')
