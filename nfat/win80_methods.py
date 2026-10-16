"""
Win80 — phương pháp chọn tầng mới cho giao thức PerBench (đặc tả: idea-stage/win80/METHODS.md; nhật ký: results/win80/LOG.md).

  brs   Backward-Racing elimination + Swaps.
        (1) Loại bỏ lùi CHÍNH XÁC từ z=1 (mọi tầng đầy đủ) với mục tiêu J = KL(gốc‖lai) khớp chế độ giải mã, đo tại đỉnh, tái dùng
            tiền tố (ứng viên S\{l} chỉ tính lại từ tầng l). Thay sàng cố định bằng ĐUA MẪU successive-halving trên các mẫu A xoay
            vòng (2 → 4 → 8 [→ 16] mẫu; giữ ⌈m/3⌉, tối thiểu 3). Khi |S| > 2k, xen kẽ một bước đầy đủ với một bước "lười" (đo lại
            top-3 của bảng xếp hạng cũ trên 8 mẫu) để giảm chi phí mà vẫn tránh hạ cùng lúc hai tầng thay thế được cho nhau.
        (2) Khởi đầu hoán đổi = J_A thấp nhất trong {tập loại bỏ, top-k gradient KL tại z=k/L, top-k chép} (đa khởi đầu, chọn chính xác).
        (3) ≤ 2 vòng hoán đổi: 3 tầng ít hại nhất (4 mẫu) × mọi tầng ngoài, đua mẫu tới 16 mẫu, nhận nếu J giảm trên cùng 16 mẫu.
        (4) Chọn cuối trong ≤ 3 ứng viên (tập hiện tại, tập loại bỏ, khởi đầu tốt nhì) theo J trên TOÀN BỘ 32 mẫu A∪B.
        Một bộ siêu tham số (BRS_HP) cho mọi mô hình / benchmark / k. Không nhìn test.
"""
import time

import numpy as np
import torch

from nfat import hyb
from nfat import perbench_methods as PM

BRS_HP = dict(obj='ghinge', n_grad=4, sched_early=(2, 2, 4), sched_late=(2, 2, 4, 8), late_margin=3, keep_frac=3, keep_min=2,
              lazy_top=3, lazy_items=6, swap_rounds=2, swap_rm=2, rm_items=8, n_final=2)   # cắt chi phí 10-08 06:35 (cổng 42 s ở 1B)
OBJ = BRS_HP['obj']
SOFT_TAU = 1.0                           # thang biên (nat) của mục tiêu 'soft'                      # thống kê theo mẫu được tối ưu: 'kl' (KL khớp giải mã) hoặc 'nll' (NLL vàng khớp giải mã)


HINGE_TAU = 2.0                          # biên đích (nat) của hinge theo vàng: max(0, tau - [logp(vàng) - max logp khác])


def _agg(name):
    """Hàm gộp danh sách thống kê-theo-mẫu (dict) -> tuple so sánh được (từ điển). 'kl'/'nll': TB; 'ghinge': (TB hinge vàng, TB NLL)
    — hinge quyết định, NLL phá hoà (ở z≈1 hinge ≈ 0 cho mọi ứng viên)."""
    if name == 'kl':
        return lambda xs: (float(np.mean([x['kl'] for x in xs])),)
    if name == 'nll':
        return lambda xs: (float(np.mean([x['nll'] for x in xs])),)
    if name == 'ghinge':
        return lambda xs: (float(np.mean([x['ghinge'] for x in xs])), float(np.mean([x['nll'] for x in xs])))
    if name == 'soft':
        return lambda xs: (float(np.mean([x['gsoft'] for x in xs])), float(np.mean([x['nll'] for x in xs])))
    raise ValueError(name)


def _f(t):
    return '/'.join(f'{v:.4f}' for v in t)


class PrefixX(PM.Prefix):
    """Như perbench_methods.Prefix (tái dùng tiền tố, phép đo CHÍNH XÁC) nhưng trả KL THEO MẪU + lật token đầu + NLL vàng."""

    @torch.no_grad()
    def run(self, i, S, start=0, h=None, store=()):
        o, it = self.o, self.o.items[i]
        ids = torch.tensor([it['ids'] + it['gold']], device=o.dev)
        P, m = len(it['ids']), len(it['gold'])
        pos = torch.arange(ids.shape[1], device=o.dev)[None]
        if h is None:
            h = self.m.embed_tokens(ids)
        pe = self.m.rotary_emb(h, pos)
        hyb.set_mode('hard', o.W, o.S, full=S)
        hyb.STATE['ans_full'] = o.decode == 'full'; hyb.STATE['P'] = P
        saved = {}
        try:
            for li in range(start, self.L):
                if li in store:
                    saved[li] = h
                h = self.m.layers[li](h, attention_mask=None, position_ids=pos, position_embeddings=pe)
                h = h[0] if isinstance(h, tuple) else h
            hn = self.m.norm(h)[0, P - 1:P - 1 + m]
            lg = torch.log_softmax(o.model.lm_head(hn).float(), -1)
            lr = o.ref[i].float()
            klt = (lr.exp() * (lr - lg)).sum(-1)
            gold = torch.tensor(it['gold'], device=o.dev)
            tam = lr.argmax(-1)                                   # argmax của teacher tại từng token đáp án
            top = lg.gather(-1, tam[:, None]).squeeze(-1)
            lg2 = lg.clone(); lg2.scatter_(-1, tam[:, None], -1e9)
            margin = top - lg2.max(-1).values                     # biên quyết định (teacher-forced) theo token
            gtop = lg.gather(-1, gold[:, None]).squeeze(-1)
            lg3 = lg.clone(); lg3.scatter_(-1, gold[:, None], -1e9)
            gmargin = gtop - lg3.max(-1).values                   # biên theo vàng
            self.last = dict(kl=float(klt.mean()), kl_first=float(klt[0]), flip=int(lg[0].argmax() != tam[0]),
                             nll=float(-gtop.mean()),
                             soft=float(torch.sigmoid(-margin / SOFT_TAU).mean()),     # tỉ lệ lỗi token mượt (so với teacher)
                             gsoft=float(torch.sigmoid(-gmargin / SOFT_TAU).mean()),   # tỉ lệ lỗi token mượt (so với vàng)
                             nflip=int((margin < 0).sum()), gnflip=int((gmargin < 0).sum()),
                             ghinge=float(torch.clamp(HINGE_TAU - gmargin, min=0).mean()))
            self.last_nll = self.last['nll']
            kl = self.last['kl']
        finally:
            hyb.STATE['ans_full'] = False; hyb.STATE['P'] = None; hyb.set_mode('off')
        return kl, saved

    def cand_eval(self, S, cands, idx):
        """{T: [dict(kl, kl_first, flip, nll) theo mẫu idx]}, số lượt xuôi quy đổi. T == S thì dùng ngay lượt tiền tố."""
        Sset = set(S); acc = {T: [] for T in cands}; fe = 0.0
        first = {T: (min(set(T) ^ Sset) if set(T) ^ Sset else self.L) for T in cands}
        store = set(v for v in first.values() if v < self.L)
        for i in idx:
            _, saved = self.run(i, sorted(S), store=store); fe += 1
            base = dict(self.last)
            for T in cands:
                st = first[T]
                if st >= self.L:
                    acc[T].append(base)
                else:
                    self.run(i, sorted(T), start=st, h=saved[st]); fe += (self.L - st) / self.L
                    acc[T].append(dict(self.last))
            del saved
        return acc, fe


def race(px, S, cands, order, sched, agg, keep_frac=3, keep_min=3, always=()):
    """Successive halving: vòng r đo các ứng viên còn sống trên sched[r] mẫu tiếp theo của order (tích luỹ), giữ ⌈m/keep_frac⌉ (≥ keep_min).
    `always`: ứng viên KHÔNG BAO GIỜ bị loại (tập đương nhiệm) để mọi so sánh cuối cùng cùng một bộ mẫu (sửa theo phản biện GPT 10-08).
    Trả (xếp hạng ứng viên sống theo J tích luỹ, {T: (J, số mẫu)}, lượt quy đổi)."""
    seen = {T: [] for T in cands}; alive = list(cands); pos = 0; fe = 0.0
    for r, n in enumerate(sched):
        idx = order[pos:pos + n]; pos += n
        if not idx or not alive:
            break
        v, f = px.cand_eval(S, alive, idx); fe += f
        for T in alive:
            seen[T] += list(v[T])
        if r < len(sched) - 1 and len(alive) > keep_min:
            kept = sorted(alive, key=lambda T: agg(seen[T]))[:max(keep_min, int(np.ceil(len(alive) / keep_frac)))]
            alive = kept + [T for T in alive if T in always and T not in kept]
    info = {T: (agg(seen[T]), len(seen[T])) for T in cands if seen[T]}
    rank = sorted(alive, key=lambda T: info[T][0])
    return rank, info, fe


def brs(model, tok, gates, items, a, dev, log):
    H = BRS_HP; agg = _agg(H['obj']); k, L = a.k, hyb.n_layers(model); t0 = time.time()
    log(f'   brs HP: {H} tau={HINGE_TAU} soft_tau={SOFT_TAU}')
    A, B, seen = [], [], {}
    for i, it in enumerate(items):          # chia A/B xen kẽ trong từng tầng lớp (như SwapKL v1.1 / FRS-4)
        t = it.get('task', '')
        (A if seen.get(t, 0) % 2 == 0 else B).append(i)
        seen[t] = seen.get(t, 0) + 1
    obj = PM.Obj(model, items, dev, a.W, a.S, a.decode, log, probe_idx=A)
    px = PrefixX(obj); fe = 0.0
    rot = lambda t: [A[(2 * t + j) % len(A)] for j in range(len(A))]       # noqa: E731  hoán vị xoay vòng của A theo bước
    # ---- (1) loại bỏ lùi có đua mẫu, tái dùng tiền tố
    S, removed, stale, t = list(range(L)), [], None, 0
    while len(S) > k:
        m = len(S); order = rot(t)
        lazy = m > 2 * k and stale is not None and t % 2 == 1
        if lazy:
            ls = [l for l in stale if l in S][:H['lazy_top']]
            cands = [tuple(x for x in S if x != l) for l in ls]
            v, f = px.cand_eval(S, cands, order[:H['lazy_items']]); fe += f
            J = {T: agg(v[T]) for T in cands}
            best = min(J, key=J.get); jb, nb = J[best], H['lazy_items']; tag = 'lười'
        else:
            cands = [tuple(x for x in S if x != l) for l in S]
            sched = H['sched_late'] if m <= k + H['late_margin'] else H['sched_early']
            rank, info, f = race(px, S, cands, order, sched, agg, H['keep_frac'], H['keep_min']); fe += f
            best = rank[0]; jb, nb = info[best]
            full = sorted(info, key=lambda T: (-info[T][1], info[T][0]))
            stale = [next(iter(set(S) - set(T))) for T in full if T != best]; tag = f'đua {sched}'
        l = next(iter(set(S) - set(best))); S.remove(l); removed.append(l); t += 1
        log(f'   brs hạ {l:2d} ({tag}, J{nb} {_f(jb)}) -> còn {len(S)} | {fe:.0f} lượt quy đổi [{time.time() - t0:.0f}s]')
    S_elim = tuple(sorted(S))
    # ---- (2) khởi đầu hoán đổi: chọn chính xác trong {loại bỏ, grad-KL, chép} theo J trên A
    g = obj.grads(gates, k, A[:H['n_grad']], names=('kl',))['kl']
    starts = {'elim': S_elim, 'kl': tuple(sorted(PM._rank(g)[:k])), 'copy': tuple(sorted(PM._rank(obj.copy)[:k]))}
    uniq = list(dict.fromkeys(starts.values()))
    v, f = px.cand_eval(list(S_elim), uniq, A); fe += f
    JA = {T: agg(v[T]) for T in uniq}
    for n, T in starts.items():
        log(f'   brs khởi đầu {n} {list(T)}: J_A {_f(JA[T])}')
    cur = min(uniq, key=JA.get)
    # ---- (3) hoán đổi có đua mẫu; nhận nếu J giảm trên cùng 16 mẫu (tập hiện tại đua cùng)
    for r in range(H['swap_rounds']):
        order = rot(100 + r)
        rmc = [tuple(x for x in cur if x != l) for l in cur]
        v, f = px.cand_eval(list(cur), rmc, order[:H['rm_items']]); fe += f
        dmg = {l: agg(v[tuple(x for x in cur if x != l)]) for l in cur}
        keep = sorted(dmg, key=dmg.get)[:H['swap_rm']]
        sw = sorted({tuple(sorted([x for x in cur if x != l] + [o])) for l in keep for o in range(L) if o not in cur})
        rank, info, f = race(px, cur, [cur] + sw, order, H['sched_late'], agg, H['keep_frac'], H['keep_min'], always=(cur,)); fe += f
        best = rank[0]
        log(f'   brs hoán đổi {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} J{info[best][1]} {_f(info[best][0])} '
            f'(hiện tại {_f(info[cur][0])}) | {fe:.0f} lượt quy đổi [{time.time() - t0:.0f}s]')
        assert info[best][1] == info[cur][1], 'so sánh phải cùng số mẫu'
        if best == cur or info[best][0] >= info[cur][0]:
            break
        cur = best
    # ---- (4) chọn cuối theo J trên toàn bộ A∪B
    fin = list(dict.fromkeys([cur] + [T for T in sorted(uniq, key=JA.get) if T != cur][:H['n_final'] - 1]))
    v, f = px.cand_eval(list(cur), fin, list(range(len(items)))); fe += f
    J32 = {T: agg(v[T]) for T in fin}
    win = min(J32, key=J32.get)
    log('   brs chọn cuối trên A∪B: ' + '; '.join(f'{list(T)} {_f(J32[T])}' for T in fin) + f' -> {list(win)} | tổng {fe:.0f} lượt quy đổi '
        f'(+{len(items)} tham chiếu, +{H["n_grad"]} xuôi+ngược) [{time.time() - t0:.0f}s]')
    out = np.zeros(L)
    for pos, l in enumerate(removed):      # thứ tự phụ: tầng bị hạ càng muộn càng cao
        out[l] = pos * 1e-3
    out[list(win)] += 10.0
    return out, len(items)


def brs_selftest(model, tok, gates, items, a, dev, log):
    """PrefixX.cand_eval (tái dùng tiền tố, KL theo mẫu) == Obj.J đủ lượt, cả bỏ / thêm / hoán đổi, theo a.decode."""
    obj = PM.Obj(model, items[:2], dev, a.W, a.S, a.decode, log, probe_idx=[])
    px = PrefixX(obj); L = obj.L; S = [1, 5, 9, 12]
    cands = [tuple(S), tuple(sorted([1, 5, 12])), tuple(sorted(S + [3])), tuple(sorted([1, 5, 9, L - 2])), tuple(sorted([0, 5, 9, 12]))]
    v, fe = px.cand_eval(S, cands, [0, 1])
    for T in cands:
        ref = obj.J(list(T), [0, 1]); got = float(np.mean([x['kl'] for x in v[T]]))
        log(f'   selftest brs ({a.decode}): {list(T)}: tiền tố {got:.6f} | đủ lượt {ref:.6f} | lệch {abs(got - ref) / max(ref, 1e-9):.2e}')
    return np.zeros(L), 2


def run(name, model, tok, gates, items, a, dev, log):
    if name == 'brs':
        return brs(model, tok, gates, items, a, dev, log)
    if name == 'brs_wide':                 # ablation hậu kiểm (LOG 10-08): sàng rộng, bỏ trần chi phí
        old = dict(BRS_HP)
        BRS_HP.update(sched_early=(4, 4, 8), sched_late=(4, 4, 8, 16), keep_min=3, lazy_items=8)
        try:
            return brs(model, tok, gates, items, a, dev, log)
        finally:
            BRS_HP.clear(); BRS_HP.update(old)
    if name == 'brs_selftest':
        return brs_selftest(model, tok, gates, items, a, dev, log)
    raise SystemExit(f'không biết {name}')
