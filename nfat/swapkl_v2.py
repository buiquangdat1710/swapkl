"""
SwapKL v2 (đặc tả: idea-stage/swk_v2/BUNDLE.md §5 V2-A′/V2-B′; nhật ký: results/swk_v2/LOG.md). KHÔNG đổi hành vi của `swapkl`.

Chạy theo TỪNG k (chi phí "mỗi k" = chi phí sinh tập ở k từ đầu — đính chính T5 sau phản biện A1):
  0. Như v1.1: chia A/B xen kẽ trong tầng lớp, lượt thầy fp32 + thăm dò chép/CAKE trên A, gradient KL/MSE tại z = k/L trên A[:8],
     J = KL(gốc‖lai) khớp chế độ giải mã (perbench_methods.setup / Obj).
  1. Khởi đầu KIẾN TẠO (mới): k ≤ L/2 -> khôi phục tiến từ ∅ tới k; k > L/2 -> loại bỏ lùi từ đủ L tầng xuống k. Mỗi bước chấm MỌI
     ứng viên (thêm/bỏ một tầng) bằng NLL đáp án khớp giải mã (tín hiệu của FRS-4), ĐUA MẪU trên A xoay vòng: 2 mẫu -> giữ 6 -> +2 mẫu
     -> giữ 2 -> +4 mẫu -> argmin TB 8 mẫu. Tái dùng tiền tố (chính xác).
  2. Khởi đầu = argmin J_A trong {tập kiến tạo, top-k chép, CAKE, grad-KL, grad-MSE, xen kẽ chép+CAKE}: J trên 8 mẫu A, 2 tập tốt
     nhất đo trên đủ A.
  3. ≤ 2 vòng hoán đổi đúng luật v1.1 (bỏ-một trên 8 mẫu A xoay vòng, giữ 2 tầng ít hại nhất, sàng mọi 2(L−k) hoán đổi trên 4 mẫu,
     xác nhận 2 ứng viên trên đủ A, nhận khi J_A giảm, không thì dừng), mọi phép đo tái dùng tiền tố.
  4. Chọn cuối — ba luật trên CÙNG một quỹ đạo (ablation miễn phí):
       swapkl2  (V2-A′): argmin J_{A∪B} (32 mẫu) trong {tập hiện tại, tập tốt nhì theo J_A};
       swapkl2b (luật v1.1): argmin J_B trong cùng hai tập;
       swapkl2c (V2-B′): pool = 3 tập J_A nhỏ nhất; "hoà" = hiệu J ghép cặp theo mẫu so với tập J_{A∪B} nhỏ nhất ≤ 1 SE; trong các tập
                 hoà chọn tập có tổng điểm chép (thăm dò trên mô hình GỐC) lớn nhất. Chỉ khi giải mã SWA; giải mã đầy đủ -> = swapkl2.
  Chạy scorer 'swapkl2' ghi cả ba (swapkl2b, swapkl2c ghi thêm vào layer_importance/select_cost với cùng thời gian của quỹ đạo).
"""
import json
import os
import time

import numpy as np
import pandas as pd

HP2 = dict(n_grad=8, init_items=8, top_starts=2, beam=2, rounds=2, rm_items=8, rm_keep=2, swap_items=4, top_full=2,
           path_frac=3 / 8, path_min_k=2, path_sched=(2, 2, 4), path_keep=(6, 2, 1), path_stride=2, tie_se=1.0, tie_pool=3)
# V2-A″ (10-12, sau phản biện nội bộ): khởi đầu kiến tạo CHỈ tiến và chỉ khi 2 ≤ k ≤ ⌊3L/8⌋ (k lớn: đường dài nhất, thua cuộc đua
# khởi đầu ở FRS-4 5/7 lần ở k = L/2, vượt trần T5; lùi = kiểu RaceElim/BRS đã thất bại); xoay mẫu bước 2 (cắt đầu tiên đổi độ sâu);
# cắt đầu giữ max(6, ⌈c/3⌉); vòng hoán đổi 1 chạy từ CẢ 2 khởi đầu tốt nhất (beam) rồi đi tiếp nhánh có J_A nhỏ hơn (sửa W8: ở NIAH 8B
# k6 khởi đầu chép + 1 hoán đổi đạt J_A 0,555 < mọi tập trên quỹ đạo khởi đầu tiến 0,572).
# V2-A‴ (10-12 20:20, người dùng chọn sau khi V2-A″ trượt cổng (b) ở sàng lọc): như V2-A″ nhưng (1) giai đoạn TIẾN giống hệt FRS-4
# (sàng MỌI ứng viên trên 4 mẫu A xoay vòng rot(step, 4), giữ 4, xác nhận thêm 4 mẫu khác của rot(step+3, 8), argmin TB 8 mẫu);
# (2) vòng hoán đổi 2 LUÔN được thử từ nhánh thắng (luật FRS-4; nguồn của lợi thế FRS-4 ở BABILong k12). Mọi thứ khác = V2-A″.
HP3 = dict(HP2, path_mode='frs', force_r2=1)
FAMILIES = {'swapkl2': HP2, 'swapkl3': HP3}
VARIANTS = ('swapkl2', 'swapkl2b', 'swapkl2c')          # tên luật cuối của họ swapkl2 (họ swapkl3 dùng swapkl3, swapkl3b, swapkl3c)


# ----------------------------------------------------------------------------- bộ nhớ đệm phép đo theo (tập, mẫu)
class Meter:
    """Bọc một bộ đo `ev.measure(base, cands, idx) -> ({T: [rec theo idx]}, fe)` (rec có 'kl', 'nll'); đệm theo (T, i) để không
    đo lại; cộng dồn chi phí fe (lượt xuôi-mẫu quy đổi, có tái dùng tiền tố)."""

    def __init__(self, ev):
        self.ev, self.cache, self.fe = ev, {}, 0.0

    def get(self, base, cands, idx):
        cands = [tuple(sorted(T)) for T in cands]
        for i in idx:
            need = [T for T in dict.fromkeys(cands) if (T, i) not in self.cache]
            if need:
                r, f = self.ev.measure(tuple(sorted(base)), need, [i])
                self.fe += f
                for T in need:
                    self.cache[(T, i)] = r[T][0]
        return {T: [self.cache[(T, i)] for i in idx] for T in cands}

    def mean(self, base, cands, idx, key='kl'):
        g = self.get(base, cands, idx)
        return {T: float(np.mean([x[key] for x in v])) for T, v in g.items()}

    def per_item(self, T, idx, key='kl'):
        return np.array([self.cache[(tuple(sorted(T)), i)][key] for i in idx])


def construct(L, k, A, M, hp, log=print):
    """Khởi đầu kiến tạo: khôi phục TIẾN từ ∅ tới k, chấm NLL đáp án khớp giải mã.
    path_mode='race' (V2-A″): mọi ứng viên trên 2 mẫu -> giữ max(6, ⌈c/3⌉) -> +2 mẫu -> giữ 2 -> +4 mẫu -> argmin TB 8 mẫu;
                              mẫu của bước s: A[(stride·s + j) mod |A|].
    path_mode='frs'  (V2-A‴): đúng FRS-4: mọi ứng viên trên rot(s, 4), giữ 4, +4 mẫu khác của rot(s+3, 8), argmin TB 8 mẫu."""
    S, step, trace = [], 0, []
    n_tot = sum(hp['path_sched'])
    frs = hp.get('path_mode', 'race') == 'frs'
    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731
    while len(S) < k:
        cands = [tuple(sorted(S + [l])) for l in range(L) if l not in S]
        if frs:
            scr = rot(step, 4)
            m1 = M.mean(S, cands, scr, key='nll')
            top = sorted(cands, key=lambda T: m1[T])[:4]
            ext = [x for x in rot(step + 3, 8) if x not in scr][:4]
            m = M.mean(S, top, scr + ext, key='nll')
            alive = sorted(top, key=lambda T: m[T])
        else:
            order = [A[(hp['path_stride'] * step + j) % len(A)] for j in range(n_tot)]
            alive, pos = list(cands), 0
            for st_i, (n_i, keep) in enumerate(zip(hp['path_sched'], hp['path_keep'])):
                idx = order[:pos + n_i]; pos += n_i
                m = M.mean(S, alive, idx, key='nll')
                kk = max(keep, -(-len(alive) // 3)) if st_i == 0 else keep
                alive = sorted(alive, key=lambda T: m[T])[:kk]
        new = list(alive[0])
        trace.append(dict(step=step, add=sorted(set(new) - set(S)), nll8=round(m[alive[0]], 4)))
        S = new; step += 1
    log(f'   v2 kiến tạo (tiến, {"FRS" if frs else "đua"}) k={k}: {S} | {M.fe:.0f} lượt quy đổi')
    return S, trace


def select(L, k, A, B, M, ranks, copy_scores, decode, hp=None, log=print, interleave=None):
    """Thuật toán chính (độc lập với mô hình: M đo J/NLL). Trả dict kết quả + vết để ghi log/ablation."""
    hp = hp or HP2
    t0 = time.time()
    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731  (giống v1.1)
    use_path = hp['path_min_k'] <= k <= int(hp['path_frac'] * L)
    Sc, ptrace = construct(L, k, A, M, hp, log) if use_path else (None, [])
    starts = {'kiến tạo': tuple(sorted(Sc))} if use_path else {}
    for n in ('copy', 'cake', 'kl', 'mse'):
        if n in ranks:
            starts[n] = tuple(sorted(ranks[n][:k]))
    if interleave is not None:
        starts['copy+cake'] = tuple(sorted(interleave))
    uniq = list(dict.fromkeys(starts.values()))
    j8 = M.mean(uniq[0], uniq, A[:hp['init_items']])
    for n, T in starts.items():
        log(f'   v2 khởi đầu {n} {list(T)}: J8 {j8[T]:.4f}')
    top = sorted(uniq, key=j8.get)[:hp['top_starts']]
    JA = M.mean(top[0], top, A)
    top = sorted(top, key=JA.get)
    log('   v2 khởi đầu tốt nhất: ' + '; '.join(f'{list(T)} J_A {JA[T]:.4f}' for T in top) + f' | {M.fe:.0f} lượt quy đổi')

    def one_round(cur, r):
        kk = min(hp['rm_keep'], k)
        rmc = [tuple(x for x in cur if x != l) for l in cur]
        rm = M.mean(cur, rmc, rot(r, hp['rm_items'])) if k > 1 else {(): 0.0}
        dmg = {l: rm[tuple(x for x in cur if x != l)] for l in cur}
        keep = sorted(dmg, key=dmg.get)[:kk]
        sw = sorted({tuple(sorted([x for x in cur if x != l] + [o])) for l in keep for o in range(L) if o not in cur})
        sc = M.mean(cur, sw, rot(r + 7, hp['swap_items']))
        cand = sorted(sw, key=sc.get)[:hp['top_full']]
        JA.update(M.mean(cur, cand, A))
        best = min(cand, key=JA.get)
        return keep, best

    rounds = []
    # vòng 1: beam trên các khởi đầu tốt nhất (cùng bộ mẫu xoay vòng), đi tiếp nhánh có J_A nhỏ nhất sau vòng 1
    branch = []
    for b, T in enumerate(top[:hp['beam']]):
        keep, best = one_round(T, 0)
        state = best if JA[best] < JA[T] else T
        branch.append(dict(start=T, best=best, state=state, improved=state != T))
        rounds.append(dict(round=1, branch=b, start=list(T), keep=keep, best=list(best), J_best=JA[best], J_start=JA[T]))
        log(f'   v2 vòng 1 nhánh {b}: từ {list(T)} (J_A {JA[T]:.4f}); bỏ-ít-hại {keep}; tốt nhất {list(best)} J_A {JA[best]:.4f} '
            f'| {M.fe:.0f} lượt quy đổi')
    win = min(branch, key=lambda x: JA[x['state']])
    start, cur = win['start'], win['state']
    if win['improved'] or hp.get('force_r2', 0):          # V2-A‴: vòng 2 luôn được thử (luật FRS-4)
        for r in range(1, hp['rounds']):
            keep, best = one_round(cur, r)
            rounds.append(dict(round=r + 1, branch=-1, keep=keep, best=list(best), J_best=JA[best], J_cur=JA[cur]))
            log(f'   v2 vòng {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} J_A {JA[best]:.4f} (hiện tại {JA[cur]:.4f}) | {M.fe:.0f} lượt quy đổi')
            if JA[best] < JA[cur]:
                cur = best
            elif not hp.get('force_r2', 0):
                break
    # mọi tập đã đo trên ĐỦ A ở k này
    full = {T: v for T, v in JA.items() if all((T, i) in M.cache for i in A)}
    order = sorted(full, key=full.get)
    runner = next((T for T in order if T != cur), None)
    pool2 = [cur] + ([runner] if runner is not None else [])
    poolc = order[:hp['tie_pool']]
    allp = list(dict.fromkeys(pool2 + poolc))
    JB = M.mean(cur, allp, B)
    AB = list(A) + list(B)
    JAB = {T: float(np.mean(M.per_item(T, AB))) for T in allp}
    out = {}
    out['swapkl2'] = min(pool2, key=JAB.get)
    out['swapkl2b'] = min(pool2, key=JB.get)
    tie = None
    if decode == 'swa':
        best = min(poolc, key=JAB.get)
        db = M.per_item(best, AB)
        tie = {}
        for T in poolc:
            d = M.per_item(T, AB) - db
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            tie[T] = dict(diff=float(d.mean()), se=se, tied=bool(d.mean() <= hp['tie_se'] * se + 1e-12),
                          copy=float(sum(copy_scores[l] for l in T)))
        tied = [T for T in poolc if tie[T]['tied']]
        out['swapkl2c'] = max(tied, key=lambda T: (tie[T]['copy'], -JAB[T]))
    else:
        out['swapkl2c'] = out['swapkl2']
    log('   v2 chọn cuối: ' + '; '.join(f'{list(T)} J_A {full[T]:.4f} J_B {JB[T]:.4f} J_AB {JAB[T]:.4f}'
                                       + (f' (hiệu {tie[T]["diff"]:+.4f}±{tie[T]["se"]:.4f}, chép {tie[T]["copy"]:.2f})' if tie and T in tie else '')
                                       for T in allp))
    log('   v2 kết quả: ' + ', '.join(f'{n} {list(T)}' for n, T in out.items()) + f' | tổng {M.fe:.0f} lượt quy đổi [{time.time() - t0:.0f}s]')
    return dict(out={n: [int(x) for x in T] for n, T in out.items()}, construct=[int(x) for x in Sc] if Sc else None, use_path=use_path,
                path=ptrace, starts={n: [int(x) for x in T] for n, T in starts.items()}, J8={str(list(T)): v for T, v in j8.items()},
                start=[int(x) for x in start], rounds=rounds,
                finals={str(list(T)): dict(J_A=full[T], J_B=JB[T], J_AB=JAB[T], **({k_: v for k_, v in tie[T].items()} if tie and T in tie else {}))
                        for T in allp}, fe=M.fe, secs=time.time() - t0)


# ----------------------------------------------------------------------------- bộ đo thật (mô hình) dựa trên PrefixX của Win80
class PrefixEval:
    def __init__(self, px):
        self.px = px

    def measure(self, base, cands, idx):
        acc, fe = self.px.cand_eval(list(base), [tuple(T) for T in cands], idx)
        return {tuple(T): [dict(kl=x['kl'], nll=x['nll']) for x in acc[tuple(T)]] for T in cands}, fe


def run(name, model, tok, gates, items, a, dev, log):
    from nfat import hyb
    from nfat import perbench_methods as PM
    from nfat.win80_methods import PrefixX
    from nfat.csv_append import append_row
    import torch
    fam = name[:7]                                         # 'swapkl2' (V2-A″) | 'swapkl3' (V2-A‴)
    hp = dict(FAMILIES[fam])
    names = {'swapkl2': fam, 'swapkl2b': fam + 'b', 'swapkl2c': fam + 'c'}   # luật cuối -> tên scorer của họ
    L, k = hyb.n_layers(model), a.k
    t0 = time.time()
    obj, A, B, ranks = PM.setup(model, gates, items, a, dev, log, {**PM.HP, 'n_grad': hp['n_grad']})
    ilv = PM._interleave_sets([ranks['copy'], ranks['cake']], k)
    M = Meter(PrefixEval(PrefixX(obj)))
    res = select(L, k, A, B, M, ranks, obj.copy, a.decode, hp, log, interleave=ilv)
    torch.cuda.synchronize()
    secs = time.time() - t0
    fe_total = res['fe'] + len(items) + 3 * 2 * hp['n_grad']            # + lượt thầy + gradient (xuôi+ngược ≈ 3 lượt)
    log(f'   v2 chi phí: {fe_total:.0f} lượt xuôi-mẫu quy đổi (gồm thầy {len(items)}, gradient ≈ {6 * hp["n_grad"]}), {secs / 60:.1f} phút')
    tr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'swk_v2', 'traces')
    os.makedirs(tr_dir, exist_ok=True)
    res.update(tag=a.tag, k=k, L=L, W=a.W, decode=a.decode, fe_total=fe_total, secs_total=secs, A=A, B=B,
               ranks={n: r[:2 * k] for n, r in ranks.items()})
    res['family'] = fam
    with open(os.path.join(tr_dir, f'{a.tag}_k{k}.json' if fam == 'swapkl2' else f'{a.tag}_{fam}_k{k}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)

    def vec(S):
        v = np.zeros(L)
        for pos, l in enumerate(ranks['kl']):
            v[l] = -pos * 1e-3
        v[list(S)] += 10.0
        return v
    # các biến thể khác của CÙNG quỹ đạo -> ghi thêm (cùng thời gian, vì muốn có chúng phải chạy cả quỹ đạo)
    for rule, n in names.items():
        if n == name:
            continue
        v = vec(res['out'][rule]); order = list(np.argsort(-v))
        append_row(pd.DataFrame([dict(tag=a.tag, model_config=a.model, scorer=n, layer=l, score=float(v[l]), rank=int(order.index(l)),
                                      window=a.W, n_sink=a.S, variant=a.bench, L=int(np.mean([len(x['ids']) for x in items])),
                                      n_items=len(items), seed=a.seed) for l in range(L)]), a.out)
        append_row(pd.DataFrame([dict(tag=a.tag, model=a.model, scorer=n, seconds=secs, n_items=len(items),
                                      gpu=torch.cuda.get_device_name())]), a.cost_out)
    rule = {v: r for r, v in names.items()}[name]
    return vec(res['out'][rule]), len(items)
