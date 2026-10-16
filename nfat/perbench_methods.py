"""
Phương pháp mới cho PerBench (đặc tả: idea-stage/METHODS.md; quyết định chạy: idea-stage/kill/KILL_ARGUMENT.md).

  swapkl    M2: khởi đầu từ các bảng xếp hạng rẻ (chép, CAKE, gradient KL/MSE tại z=k/L), rồi tìm kiếm HOÁN ĐỔI CHÍNH XÁC
            tại đỉnh với mục tiêu KL(gốc‖lai) khớp chế độ giải mã; chọn cuối trên tập mẫu B rời.
  poolelim  M1: loại bỏ lùi chính xác bắt đầu từ pool 2k tầng (xoay vòng 4 bảng xếp hạng rẻ).

Một bộ siêu tham số cho mọi mô hình/benchmark (hằng số HP dưới đây). Chỉ đầu vào giao thức thay đổi: mô hình, L, k, W, S,
chế độ giải mã (--decode), dữ liệu hiệu chỉnh của benchmark.
"""
import numpy as np
import torch

from nfat import hyb
from nfat.baselines import _attn_rows

HP = dict(n_grad=8, init_items=8, rounds=3, rm_items=8, rm_keep=2, swap_items=4, top_full=2,
          pool_mult=2, elim_screen=4, elim_top=2)

# Ablation SwapKL (10-09): mỗi biến thể đổi ĐÚNG MỘT yếu tố so với SwapKL v1.1; 4 bảng xếp hạng rẻ (khởi tạo) giữ nguyên
# (gradient KL để xếp hạng luôn là KL thuận). div: fwd = KL(gốc‖lai), rev = KL(lai‖gốc), sym = fwd + rev.
# nostop=1: không dừng khi một vòng không cải thiện (chạy đủ số vòng, mẫu xoay vòng khác nhau).
SWK_VARIANTS = {
    'swk_rev': dict(div='rev'), 'swk_sym': dict(div='sym'),
    'swk_rk4': dict(rm_keep=4), 'swk_rkall': dict(rm_keep=99),            # 99 -> k: lân cận 1-hoán-đổi đầy đủ k(L-k)
    'swk_r6': dict(rounds=6), 'swk_r6ns': dict(rounds=6, nostop=1),
    'swk_sw8': dict(swap_items=8), 'swk_top4': dict(top_full=4), 'swk_rm16': dict(rm_items=16),
    'swk_init16': dict(init_items=16), 'swk_ng16': dict(n_grad=16),
    'swk_ms2': dict(n_starts=2),                                          # hoán đổi từ 2 khởi đầu tốt nhất, B chọn
    'swk_owm': dict(owm=1),        # 10-14 (yêu cầu người dùng): + tập khởi đầu thứ 6 = top-k theo khối lượng attention NGOÀI cửa sổ
}
# Ablation bước khởi đầu (10-10): giới hạn tập khởi đầu được phép (5 tập: copy, cake, kl, mse, copy+cake).
# swk_i_no<X>: bỏ một tập; swk_i_<X>: chỉ dùng một tập. Mọi bảng xếp hạng vẫn được tính (gradient KL còn dùng cho thứ tự phụ).
_INITS = {'copy': 'copy', 'cake': 'cake', 'kl': 'kl', 'mse': 'mse', 'ilv': 'copy+cake'}
for _x, _n in _INITS.items():
    SWK_VARIANTS[f'swk_i_no{_x}'] = dict(inits=tuple(n for n in _INITS.values() if n != _n))
    if _x != 'ilv':
        SWK_VARIANTS[f'swk_i_{_x}'] = dict(inits=(_n,))


class Obj:
    """J(S) = TB KL(gốc‖lai) tại vị trí dự đoán token đáp án; token đáp án chạy theo chế độ giải mã triển khai.
    Lượt xuôi tham chiếu (mô hình gốc) đồng thời thu thập hai thăm dò rẻ: điểm chép theo tầng và CAKE theo tầng."""

    def __init__(self, model, items, dev, W, S, decode, log=print, Sw=32, probe_idx=None, div='fwd'):
        self.model, self.items, self.dev, self.W, self.S, self.decode = model, items, dev, W, S, decode
        self.div = div                       # chiều KL của mục tiêu tìm kiếm J (gradient xếp hạng luôn dùng 'fwd')
        self.L = hyb.n_layers(model)
        self.ref, self.href = [], []
        self.copy = np.zeros(self.L); self.cake = np.zeros(self.L)
        self.owm = np.zeros(self.L)          # khối lượng attention ngoài cửa sổ (10-14; chỉ dùng bởi biến thể swk_owm)
        pr = dict(fn=self._probe, Sw=Sw)
        hyb.set_mode('off')
        with torch.no_grad():
            probe_idx = set(range(len(items))) if probe_idx is None else set(probe_idx)
            for i, it in enumerate(items):
                pr['P'], pr['m'] = len(it['ids']), len(it['gold'])
                pr['ids'] = torch.tensor(it['ids'] + it['gold'], device=dev)
                hyb.STATE['probe'] = pr if i in probe_idx else None
                try:
                    h = self._hidden(it)
                finally:
                    hyb.STATE['probe'] = None
                self.href.append(h.to(torch.bfloat16).clone())   # clone: không giữ cả trạng thái ẩn dài
                self.ref.append(torch.log_softmax(model.lm_head(h).float(), -1))      # teacher fp32 (v1.1)
        self.copy /= max(1, len(probe_idx)); self.cake /= max(1, len(probe_idx)); self.owm /= max(1, len(probe_idx))
        self.n_eval = 0                      # số lượt xuôi-mẫu tại đỉnh (để báo chi phí)

    def _probe(self, l, q, k, scale, pr):
        P, m = pr['P'], pr['m']
        ids = pr['ids']
        # chép: argmax attention của token dự đoán đáp án rơi đúng token ngữ cảnh trùng token đích
        A = _attn_rows(q, k, scale, list(range(P - 1, P - 1 + m)))[:, :, :P]
        j = A.argmax(-1)
        hit = ids[j] == ids[P:P + m].view(1, -1)
        self.copy[l] += float(hit.float().mean(-1).sum())
        # CAKE trên prompt: Sw truy vấn cuối của prompt, khoá trước chúng
        Sw = min(pr['Sw'], P // 2)
        A = _attn_rows(q, k, scale, list(range(P - Sw, P)))[:, :, :P - Sw]
        Hs = -(A * torch.log(A.clamp_min(1e-30))).sum(-1).sum(-1)
        Vs = A.var(dim=1, unbiased=False).sum(-1)
        self.cake[l] += float(Hs.mean()) * float(Vs.mean())
        # khối lượng attention NGOÀI cửa sổ (OWM): tỉ lệ attention của mô hình gốc ở truy vấn t rơi vào khoá j với S <= j <= t-W
        # (ngoài cửa sổ {t-W+1..t} và ngoài S sink) = đúng phần bị cắt khi tầng này chuyển sang SWA. Truy vấn: các vị trí dự đoán
        # đáp án (bỏ nếu giải mã đầy đủ: đáp án vẫn attention đầy đủ khi triển khai) và 64 vị trí prompt cách đều trong [W, P).
        W, S0 = self.W, self.S
        def om(rows):
            A = _attn_rows(q, k, scale, rows)                                   # (H, n, K)
            pos = torch.arange(A.shape[-1], device=A.device).view(1, 1, -1)
            t = torch.as_tensor(rows, device=A.device).view(1, -1, 1)
            return float(((pos >= S0) & (pos <= t - W)).float().mul(A).sum(-1).mean())
        parts = []
        if self.decode != 'full':
            parts.append(om(list(range(P - 1, P - 1 + m))))
        if P - 1 >= W:
            parts.append(om(sorted(set(np.linspace(W, P - 1, 64).astype(int).tolist()))))
        if parts:
            self.owm[l] += float(np.mean(parts))

    def _hidden(self, it, keep=False):
        """keep=True: KHÔNG xoá trạng thái giải mã (ans_full, P) — bắt buộc khi sẽ backward qua gradient checkpointing,
        vì lượt xuôi được tính lại trong backward phải thấy đúng mặt nạ (lỗi của v1, phát hiện bởi phân xử kill-argument)."""
        ids = torch.tensor([it['ids'] + it['gold']], device=self.dev)
        P, m = len(it['ids']), len(it['gold'])
        hyb.STATE['ans_full'] = self.decode == 'full'; hyb.STATE['P'] = P
        try:
            return self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0, P - 1:P - 1 + m]
        finally:
            if not keep:
                hyb.STATE['ans_full'] = False; hyb.STATE['P'] = None

    def _kl(self, i, h, div=None):
        lr = self.ref[i].float()
        lg = torch.log_softmax(self.model.lm_head(h).float(), -1)
        div = div or self.div
        fwd = lambda: (lr.exp() * (lr - lg)).sum(-1)          # noqa: E731  KL(gốc‖lai)
        rev = lambda: (lg.exp() * (lg - lr)).sum(-1)          # noqa: E731  KL(lai‖gốc)
        return {'fwd': fwd, 'rev': rev, 'sym': lambda: fwd() + rev()}[div]().mean()

    @torch.no_grad()
    def J(self, S, idx):
        hyb.set_mode('hard', self.W, self.S, full=S)
        v = [float(self._kl(i, self._hidden(self.items[i]))) for i in idx]
        self.n_eval += len(idx)
        hyb.set_mode('off')
        return float(np.mean(v))

    def grads(self, gates, k, idx, names=('kl', 'mse')):
        """-dJ/dz và -dMSE/dz tại z = k/L (điểm vận hành)."""
        out = {}
        for name in names:
            gates.set(k / self.L); gates.z.grad = None
            for i in idx:
                try:
                    h = self._hidden(self.items[i], keep=True)
                    loss = self._kl(i, h, 'fwd') if name == 'kl' else (h.float() - self.href[i].float()).pow(2).mean()
                    (loss / len(idx)).backward()
                finally:
                    hyb.STATE['ans_full'] = False; hyb.STATE['P'] = None
            out[name] = -gates.z.grad.detach().float().cpu().numpy().copy()
        hyb.set_mode('off')
        return out


def _rank(v):
    return [int(x) for x in np.argsort(-np.asarray(v))]


def _interleave_sets(ranks, k):
    """Giao của các bảng xếp hạng (top-k) rồi điền xen kẽ theo thứ tự hạng."""
    tops = [set(r[:k]) for r in ranks]
    S = list(set.intersection(*tops)) if tops else []
    p = 0
    while len(S) < k:
        for r in ranks:
            if p < len(r) and r[p] not in S and len(S) < k:
                S.append(r[p])
        p += 1
    return sorted(S)


def setup(model, gates, items, a, dev, log, hp=None, div='fwd'):
    hp = hp or HP
    A, B, seen = [], [], {}
    for i, it in enumerate(items):           # chia A/B xen kẽ TRONG từng tầng lớp (tác vụ×độ dài / độ sâu), giữ thứ tự gốc
        t = it.get('task', '')
        (A if seen.get(t, 0) % 2 == 0 else B).append(i)
        seen[t] = seen.get(t, 0) + 1
    obj = Obj(model, items, dev, a.W, a.S, a.decode, log, probe_idx=A, div=div)   # thăm dò chỉ dùng A (B là xác nhận độc lập)
    g = obj.grads(gates, a.k, A[:hp['n_grad']])
    ranks = {'kl': _rank(g['kl']), 'mse': _rank(g['mse']), 'copy': _rank(obj.copy), 'cake': _rank(obj.cake)}
    for n, r in ranks.items():
        log(f'   xếp hạng {n}: top-{2 * a.k} {r[:2 * a.k]}')
    return obj, A, B, ranks


def swapkl(model, tok, gates, items, a, dev, log, var=None):
    var = dict(var or {})
    div, nostop = var.pop('div', 'fwd'), bool(var.pop('nostop', 0))
    use_owm = bool(var.pop('owm', 0))
    hp = {**HP, **var}
    k, L = a.k, hyb.n_layers(model)
    hp['rm_keep'] = min(hp['rm_keep'], k)
    if var or div != 'fwd' or nostop:
        log(f'   biến thể: div={div} nostop={nostop} hp={ {x: hp[x] for x in var} }')
    obj, A, B, ranks = setup(model, gates, items, a, dev, log, hp, div)
    if use_owm:                                     # swk_owm: thêm bảng xếp hạng OWM -> tập khởi đầu thứ 6
        ranks['owm'] = _rank(obj.owm)
        log(f'   xếp hạng owm: top-{2 * k} {ranks["owm"][:2 * k]} | OWM theo tầng {np.round(obj.owm, 3).tolist()}')
    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731  tập con xoay vòng của A
    inits = {n: sorted(r[:k]) for n, r in ranks.items()}
    inits['copy+cake'] = _interleave_sets([ranks['copy'], ranks['cake']], k)
    if 'inits' in hp:
        inits = {n: S for n, S in inits.items() if n in hp['inits']}
    seen = {}
    for n, S in inits.items():
        t = tuple(S)
        if t not in seen:
            seen[t] = obj.J(S, A[:hp['init_items']])
        log(f'   khởi đầu {n} {S}: J8 {seen[t]:.4f}')
    full16, finals = {}, []
    for S0 in sorted(seen, key=seen.get)[:hp.get('n_starts', 1)]:     # n_starts=1: đúng SwapKL v1.1
        S = list(S0)
        if tuple(S) not in full16:
            full16[tuple(S)] = obj.J(S, A)
        log(f'   -> chọn khởi đầu {S}: J16 {full16[tuple(S)]:.4f}')
        for r in range(hp['rounds']):
            sub = rot(r, hp['rm_items'])
            rm = {l: obj.J([x for x in S if x != l], sub) for l in S}
            keep = sorted(rm, key=rm.get)[:hp['rm_keep']]
            sw = rot(r + 7, hp['swap_items'])
            cand = {}
            for l in keep:
                for o in range(L):
                    if o in S:
                        continue
                    T = tuple(sorted([x for x in S if x != l] + [o]))
                    cand[T] = obj.J(list(T), sw)
            top = sorted(cand, key=cand.get)[:hp['top_full']]
            for T in top:
                if T not in full16:
                    full16[T] = obj.J(list(T), A)
            best = min(top, key=lambda T: full16[T])
            log(f'   vòng {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} J16 {full16[best]:.4f} (hiện tại {full16[tuple(S)]:.4f}) '
                f'| {obj.n_eval} lượt-mẫu')
            if full16[best] >= full16[tuple(S)]:
                if nostop:
                    continue
                break
            S = list(best)
        finals.append(tuple(S))
    # chọn cuối trên B rời: kết quả của mỗi khởi đầu (v1.1: tập hiện tại vs tập tốt thứ hai đã đo trên 16 mẫu A)
    fin = list(dict.fromkeys(finals))
    if len(fin) == 1:
        order = sorted(full16, key=full16.get)
        fin += [T for T in order if T != fin[0]][:1]
    jb = {T: obj.J(list(T), B) for T in fin}
    win = min(jb, key=jb.get)
    log(f'   chọn cuối trên B: ' + '; '.join(f'{list(T)} {v:.4f}' for T, v in jb.items()) + f' -> {list(win)}')
    log(f'   tổng {obj.n_eval} lượt xuôi-mẫu tại đỉnh (+{len(items)} tham chiếu, +{2 * hp["n_grad"]} xuôi+ngược)')
    v = np.zeros(L)
    for pos, l in enumerate(ranks['kl']):          # thứ tự phụ (ngoài tập) theo gradient KL
        v[l] = -pos * 1e-3
    v[list(win)] += 10.0
    return v, len(items)


def poolelim(model, tok, gates, items, a, dev, log):
    k, L = a.k, hyb.n_layers(model)
    obj, A, B, ranks = setup(model, gates, items, a, dev, log)
    pool, p, order = [], 0, ['kl', 'mse', 'copy', 'cake']
    while len(pool) < HP['pool_mult'] * k:
        for n in order:
            l = ranks[n][p]
            if l not in pool and len(pool) < HP['pool_mult'] * k:
                pool.append(l)
        p += 1
    S = sorted(pool)
    log(f'   pool {S}')
    step = 0
    while len(S) > k:
        sub = [A[(step * HP['elim_screen'] + j) % len(A)] for j in range(HP['elim_screen'])]
        c = {l: obj.J([x for x in S if x != l], sub) for l in S}
        top = sorted(c, key=c.get)[:HP['elim_top']]
        f = {l: obj.J([x for x in S if x != l], A) for l in top}
        l = min(f, key=f.get)
        S.remove(l); step += 1
        log(f'   hạ {l} (J16 {f[l]:.4f}) -> {S} | {obj.n_eval} lượt-mẫu')
    v = np.zeros(L); v[S] = 10.0
    return v, len(items)


# ----------------------------------------------------------------------------- M3 RaceElim
class Prefix:
    """Lượt xuôi thủ công theo tầng để TÁI DÙNG TIỀN TỐ: với tập S và ứng viên S\\{l}, các tầng < l giống hệt nhau nên chỉ
    tính lại từ tầng l (phép đo vẫn CHÍNH XÁC, chỉ là tối ưu cài đặt). Dùng cho Llama/Qwen3 (embed_tokens, rotary_emb, layers, norm)."""

    def __init__(self, obj):
        self.o = obj; self.m = obj.model.model; self.L = obj.L

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
            self.last_nll = float(-lg.gather(-1, torch.tensor(it['gold'], device=o.dev)[:, None]).mean())
            kl = float((lr.exp() * (lr - lg)).sum(-1).mean())
        finally:
            hyb.STATE['ans_full'] = False; hyb.STATE['P'] = None; hyb.set_mode('off')
        return kl, saved

    def removal_J(self, S, cands, idx):
        """J(S\\{l}) cho mọi l trong cands, trung bình trên idx; trả (dict, số lượt xuôi quy đổi)."""
        acc = {l: 0.0 for l in cands}; fe = 0.0
        for i in idx:
            _, saved = self.run(i, S, store=set(cands)); fe += 1
            for l in cands:
                v, _ = self.run(i, [x for x in S if x != l], start=l, h=saved[l]); acc[l] += v / len(idx)
                fe += (self.L - l) / self.L
            del saved
        return acc, fe

    def cand_J(self, S, cands, idx):
        """J cho các tập ứng viên bất kỳ (tuple tầng), dùng lại tiền tố của tập hiện tại S: ứng viên T chỉ tính lại từ tầng
        đầu tiên mà T và S khác nhau. Trả ({T: (KL TB, NLL TB)}, số lượt xuôi quy đổi)."""
        Sset = set(S); acc = {T: [0.0, 0.0] for T in cands}; fe = 0.0
        first = {T: min(set(T) ^ Sset) if set(T) ^ Sset else self.L for T in cands}
        for i in idx:
            _, saved = self.run(i, sorted(S), store=set(first.values())); fe += 1
            for T in cands:
                st = first[T]
                if st >= self.L:
                    v, _ = self.run(i, sorted(T)); nl = self.last_nll; fe += 1
                else:
                    v, _ = self.run(i, sorted(T), start=st, h=saved[st]); nl = self.last_nll; fe += (self.L - st) / self.L
                acc[T][0] += v / len(idx); acc[T][1] += nl / len(idx)
            del saved
        return {T: tuple(x) for T, x in acc.items()}, fe


def raceelim(model, tok, gates, items, a, dev, log):
    k, L = a.k, hyb.n_layers(model)
    A, seen = [], {}
    for i, it in enumerate(items):
        t = it.get('task', '')
        if seen.get(t, 0) % 2 == 0:
            A.append(i)
        seen[t] = seen.get(t, 0) + 1
    obj = Obj(model, items, dev, a.W, a.S, a.decode, log, probe_idx=[])
    px = Prefix(obj)
    S, step, fe = list(range(L)), 0, 0.0
    while len(S) > k:
        scr = [A[(2 * step + j) % len(A)] for j in range(2)]
        c, f1 = px.removal_J(S, S, scr)
        top = sorted(c, key=c.get)[:2]
        conf = [A[(8 * step + j) % len(A)] for j in range(8)]
        d, f2 = px.removal_J(S, top, conf)
        l = min(d, key=d.get); S.remove(l); step += 1; fe += f1 + f2
        log(f'   raceelim: hạ {l} (J8 {d[l]:.4f}; ứng viên {top}) -> còn {len(S)} | {fe:.0f} lượt-xuôi quy đổi')
    v = np.zeros(L); v[S] = 10.0
    return v, len(items)


def raceelim_selftest(model, tok, gates, items, a, dev, log):
    """So J tính theo hậu tố (tái dùng tiền tố) với J tính đủ lượt (Obj.J) cho vài ứng viên."""
    obj = Obj(model, items[:2], dev, a.W, a.S, a.decode, log, probe_idx=[])
    px = Prefix(obj)
    L = obj.L; S = list(range(0, L, 2)) + [L - 1]
    c, _ = px.removal_J(S, S[:4], [0, 1])
    for l in S[:4]:
        ref = obj.J([x for x in S if x != l], [0, 1])
        log(f'   selftest raceelim: bỏ {l}: hậu tố {c[l]:.6f} | đủ lượt {ref:.6f} | lệch tương đối {abs(c[l] - ref) / max(ref, 1e-9):.2e}')
    return np.zeros(L), 2


# ----------------------------------------------------------------------------- NLL-guided tham lam (yêu cầu người dùng)
@torch.no_grad()
def nllg_greedy(model, tok, gates, items, a, dev, log):
    """NLL-guided dạng THAM LAM (khôi phục tiến, có tính lại): bắt đầu toàn SWA (S = rỗng); mỗi vòng thử bật từng tầng l ngoài S
    lên đầy đủ, đo NLL đáp án tại đỉnh S ∪ {l} (cùng phép đo của nll_swa / NLL-guided Eq. 1), thêm tầng làm NLL thấp nhất;
    lặp k vòng. Vòng đầu = NLL-guided gốc. Dùng 16 mẫu đầu (xếp xoay vòng tầng lớp) như SLEB. Chi phí sum_{j<k} (L - j) đỉnh."""
    k, L = a.k, hyb.n_layers(model)
    sub = items[:a.sleb_n]
    hyb.STATE['ans_full'] = False

    def nll(S):
        hyb.set_mode('hard', a.W, a.S, full=S)
        return float(hyb.answer_nll(model, sub, dev))
    S, order = [], []
    base = nll([])
    for step in range(k):
        cand = {l: nll(S + [l]) for l in range(L) if l not in S}
        l = min(cand, key=cand.get); S.append(l); order.append(l)
        log(f'   nllg_greedy vòng {step + 1}/{k}: +{l} (NLL {cand[l]:.4f}, toàn SWA {base:.4f}) -> {sorted(S)}')
    hyb.set_mode('off')
    v = np.full(L, -1e9)
    for pos, l in enumerate(order):
        v[l] = -pos
    return v, len(sub)


# ----------------------------------------------------------------------------- FRS (SwapKL v2): forward restore + swaps
FRS_HP = dict(n_grad=8, screen=4, confirm=8, keep_fwd=4, init_items=16, swap_rounds=2, rm_items=4, rm_keep=2, swap_items=4, top_full=2)  # FRS-4


def frs(model, tok, gates, items, a, dev, log):
    """refine-logs/round-1-refinement.md. Một trục kiến tạo (khôi phục tiến từ toàn SWA, xếp hạng bằng NLL khớp giải mã, có
    đua mẫu) + một toán tử hoán đổi (KL khớp giải mã) + các tập rẻ dự phòng; tập B dùng đúng một lần."""
    import time
    H = FRS_HP; k, L = a.k, hyb.n_layers(model); t0 = time.time()
    A, B, seen = [], [], {}
    for i, it in enumerate(items):
        t = it.get('task', '')
        (A if seen.get(t, 0) % 2 == 0 else B).append(i)
        seen[t] = seen.get(t, 0) + 1
    obj = Obj(model, items, dev, a.W, a.S, a.decode, log, probe_idx=A)
    px = Prefix(obj); fe = 0.0
    g = obj.grads(gates, k, A[:H['n_grad']], names=('kl',))['kl']
    rk = {'kl': _rank(g), 'copy': _rank(obj.copy), 'cake': _rank(obj.cake)}
    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731
    # 2. khôi phục tiến bằng NLL, đua mẫu
    S = []
    for step in range(k):
        cands = [tuple(sorted(S + [l])) for l in range(L) if l not in S]
        scr = rot(step, H['screen'])
        v, f1 = px.cand_J(S, cands, scr)
        top = sorted(cands, key=lambda T: v[T][1])[:H['keep_fwd']]
        ext = [x for x in rot(step + 3, H['confirm']) if x not in scr][:H['confirm'] - H['screen']]
        w, f2 = px.cand_J(S, top, ext)
        tot = {T: (v[T][1] * len(scr) + w[T][1] * len(ext)) / (len(scr) + len(ext)) for T in top}
        S = list(min(tot, key=tot.get)); fe += f1 + f2
        log(f'   frs tiến {step + 1}/{k}: {S} (NLL8 {tot[tuple(S)]:.4f}) | {fe:.0f} lượt quy đổi  [{time.time() - t0:.0f}s]')
    # 3. khởi đầu hoán đổi = J_KL thấp nhất trên 16 A
    starts = {'forward': tuple(S)}
    for n, r in rk.items():
        starts[n] = tuple(sorted(r[:k]))
    uniq = sorted(set(starts.values()))
    v, f = px.cand_J(list(uniq[0]), uniq, A); fe += f
    A16 = {T: v[T][0] for T in uniq}
    for n, T in starts.items():
        log(f'   frs khởi đầu {n} {list(T)}: KL16 {A16[T]:.4f}')
    cur = min(A16, key=A16.get)
    # 4. hoán đổi
    for r in range(H['swap_rounds']):
        rmc = [tuple(x for x in cur if x != l) for l in cur]
        v, f = px.cand_J(list(cur), [tuple(sorted(T)) for T in rmc], rot(r + 5, H['rm_items'])); fe += f
        dmg = {l: v[tuple(sorted(x for x in cur if x != l))][0] for l in cur}
        keep = sorted(dmg, key=dmg.get)[:H['rm_keep']]
        sw = sorted({tuple(sorted([x for x in cur if x != l] + [o])) for l in keep for o in range(L) if o not in cur})
        v, f = px.cand_J(list(cur), sw, rot(r + 9, H['swap_items'])); fe += f
        top = sorted(sw, key=lambda T: v[T][0])[:H['top_full']]
        new = [T for T in top if T not in A16]
        if new:
            w, f = px.cand_J(list(cur), new, A); fe += f
            A16.update({T: w[T][0] for T in new})
        best = min(top, key=lambda T: A16[T])
        log(f'   frs hoán đổi {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} KL16 {A16[best]:.4f} (hiện tại {A16[cur]:.4f}) '
            f'| {fe:.0f} lượt quy đổi  [{time.time() - t0:.0f}s]')
        if A16[best] < A16[cur]:              # FRS-4: vòng sau vẫn thử (bảng mẫu xoay vòng) kể cả khi vòng này không cải thiện
            cur = best
    # 5. B dùng một lần cho 2 tập tốt nhất theo A
    fin = sorted(A16, key=A16.get)[:2]
    v, f = px.cand_J(list(fin[0]), fin, B); fe += f
    win = min(fin, key=lambda T: v[T][0])
    log(f'   frs chọn cuối trên B: ' + '; '.join(f'{list(T)} {v[T][0]:.4f}' for T in fin) + f' -> {list(win)} | tổng {fe:.0f} lượt quy đổi')
    out = np.zeros(L)
    for pos, l in enumerate(rk['kl']):
        out[l] = -pos * 1e-3
    out[list(win)] += 10.0
    return out, len(items)


def frs_selftest(model, tok, gates, items, a, dev, log):
    """cand_J (tái dùng tiền tố) == Obj.J đủ lượt cho phép THÊM và HOÁN ĐỔI, theo chế độ giải mã của a.decode."""
    obj = Obj(model, items[:2], dev, a.W, a.S, a.decode, log, probe_idx=[])
    px = Prefix(obj); L = obj.L; S = [1, 5, 9]
    cands = [tuple(sorted(S + [3])), tuple(sorted(S + [L - 2])), tuple(sorted([1, 9, 12])), tuple(sorted([5, 9, 0]))]
    v, _ = px.cand_J(S, cands, [0, 1])
    for T in cands:
        ref = obj.J(list(T), [0, 1])
        log(f'   selftest frs ({a.decode}): {list(T)}: hậu tố {v[T][0]:.6f} | đủ lượt {ref:.6f} | lệch {abs(v[T][0] - ref) / max(ref, 1e-9):.2e}')
    return np.zeros(L), 2


# ----------------------------------------------------------------------------- SLEB nối tiếp (quét k nhỏ hơn k đã chạy)
@torch.no_grad()
def sleb_tail(model, tok, gates, items, a, dev, log):
    """Tiếp tục ĐÚNG thuật toán SLEB (loại bỏ tham lam, NLL đáp án, cùng 16 mẫu đầu) từ tập start (= tập SLEB tại k đã chạy)
    xuống 1 tầng. Điểm: thứ tự loại (tầng bị loại muộn hơn = điểm cao hơn) -> top-k cho mọi k < |start|."""
    start = [int(x) for x in a.extra['start'].split('_')]
    L = hyb.n_layers(model); sub = items[:a.sleb_n]

    def nll(fs):
        hyb.set_mode('hard', a.W, a.S, full=fs)
        return float(hyb.answer_nll(model, sub, dev))
    full, removed = set(start), []
    while len(full) > 1:
        cand = {l: nll(full - {l}) for l in sorted(full)}
        l = min(cand, key=cand.get); full.remove(l); removed.append(l)
        log(f'   sleb_tail: hạ {l} -> còn {sorted(full)}')
    hyb.set_mode('off')
    v = np.full(L, -1e9)
    for pos, l in enumerate(removed):
        v[l] = pos
    v[list(full)] = len(removed) + 1
    return v, len(sub)


def _measure(px, obj, S, cands, idx):
    """J (TB KL trên idx) cho các tập ứng viên, chọn cách rẻ hơn giữa (a) tái dùng tiền tố của S (ứng viên T chỉ tính lại từ tầng
    đầu tiên khác S; T == S dùng luôn lượt tiền tố) và (b) chạy đủ lượt từng tập. Cả hai đều là phép đo CHÍNH XÁC. Trả (dict, lượt quy đổi)."""
    Sset = set(S); L = px.L
    first = {T: (min(set(T) ^ Sset) if set(T) ^ Sset else L) for T in cands}
    pref = 1.0 + sum((L - f) / L for f in first.values() if f < L)
    if pref >= len(cands):
        return {T: obj.J(list(T), idx) for T in cands}, float(len(cands) * len(idx))
    acc = {T: 0.0 for T in cands}; fe = 0.0
    store = set(f for f in first.values() if f < L)
    for i in idx:
        kb, saved = px.run(i, sorted(S), store=store); fe += 1
        for T in cands:
            f_ = first[T]
            if f_ >= L:
                acc[T] += kb / len(idx)
            else:
                v, _ = px.run(i, sorted(T), start=f_, h=saved[f_]); acc[T] += v / len(idx); fe += (L - f_) / L
        del saved
    return acc, fe


# ----------------------------------------------------------------------------- SwapKL nhanh (10-14, yêu cầu người dùng)
def swapkl_fast(model, tok, gates, items, a, dev, log):
    """ĐÚNG thuật toán SwapKL v1.1 (cùng HP, cùng mẫu, cùng thứ tự quyết định) nhưng mọi phép đo J dùng TÁI DÙNG TIỀN TỐ
    (Prefix: ứng viên chỉ tính lại từ tầng đầu tiên khác tập hiện tại; phép đo vẫn chính xác, sai khác số học ~1e-6).
    Mục tiêu: cùng tập đầu ra với `swapkl`, ít thời gian hơn. `swapkl` không đổi."""
    hp = dict(HP)
    k, L = a.k, hyb.n_layers(model)
    hp['rm_keep'] = min(hp['rm_keep'], k)
    obj, A, B, ranks = setup(model, gates, items, a, dev, log, hp)
    px = Prefix(obj); fe = 0.0
    rot = lambda r, n: [A[(r * n + j) % len(A)] for j in range(n)]          # noqa: E731
    inits = {n: sorted(r[:k]) for n, r in ranks.items()}
    inits['copy+cake'] = _interleave_sets([ranks['copy'], ranks['cake']], k)
    uniq = list(dict.fromkeys(tuple(S) for S in inits.values()))
    seen, f = _measure(px, obj, list(uniq[0]), uniq, A[:hp['init_items']]); fe += f
    for n, S in inits.items():
        log(f'   khởi đầu {n} {S}: J8 {seen[tuple(S)]:.4f}')
    S0 = min(seen, key=seen.get)                    # = sorted(seen, key=seen.get)[0] của v1.1
    S = list(S0)
    full16, f = _measure(px, obj, S, [tuple(S)], A); fe += f
    log(f'   -> chọn khởi đầu {S}: J16 {full16[tuple(S)]:.4f}')
    for r in range(hp['rounds']):
        sub = rot(r, hp['rm_items'])
        if k > 1:
            rmv, f = px.removal_J(S, list(S), sub); fe += f
            rm = {l: rmv[l] for l in S}
        else:
            rm = {S[0]: 0.0}
        keep = sorted(rm, key=rm.get)[:hp['rm_keep']]
        sw = rot(r + 7, hp['swap_items'])
        cands = []
        for l in keep:
            for o in range(L):
                if o in S:
                    continue
                T = tuple(sorted([x for x in S if x != l] + [o]))
                if T not in cands:
                    cands.append(T)
        cand, f = _measure(px, obj, S, cands, sw); fe += f
        top = sorted(cand, key=cand.get)[:hp['top_full']]
        new = [T for T in top if T not in full16]
        if new:
            v, f = _measure(px, obj, S, new, A); fe += f
            full16.update(v)
        best = min(top, key=lambda T: full16[T])
        log(f'   vòng {r + 1}: bỏ-ít-hại {keep}; tốt nhất {list(best)} J16 {full16[best]:.4f} (hiện tại {full16[tuple(S)]:.4f}) '
            f'| {fe:.0f} lượt quy đổi')
        if full16[best] >= full16[tuple(S)]:
            break
        S = list(best)
    fin = [tuple(S)]
    order = sorted(full16, key=full16.get)
    fin += [T for T in order if T != fin[0]][:1]
    jb, f = _measure(px, obj, S, fin, B); fe += f
    win = min(jb, key=jb.get)
    log(f'   chọn cuối trên B: ' + '; '.join(f'{list(T)} {x:.4f}' for T, x in jb.items()) + f' -> {list(win)}')
    log(f'   tổng {fe:.0f} lượt xuôi quy đổi (tái dùng tiền tố) (+{len(items)} tham chiếu, +{2 * hp["n_grad"]} xuôi+ngược)')
    out = np.zeros(L)
    for pos, l in enumerate(ranks['kl']):
        out[l] = -pos * 1e-3
    out[list(win)] += 10.0
    return out, len(items)


def run(name, model, tok, gates, items, a, dev, log):
    if name.startswith(('swapkl2', 'swapkl3')):    # SwapKL v2: V2-A″ / V2-A‴ (nfat/swapkl_v2.py; results/swk_v2/LOG.md) — `swapkl` không đổi
        from nfat import swapkl_v2 as V2
        return V2.run(name, model, tok, gates, items, a, dev, log)
    if name.startswith('swkab'):                   # ablation SwapKL nhiều biến thể / một lượt (nfat/swk_ablate.py; 10-14)
        from nfat import swk_ablate as AB
        return AB.run(name, model, tok, gates, items, a, dev, log)
    if name.startswith('brs'):                     # Win80 (nfat/win80_methods.py)
        from nfat import win80_methods as W
        return W.run(name, model, tok, gates, items, a, dev, log)
    if name.startswith('swapd'):                   # hướng D (nfat/evresp.py)
        from nfat import evresp as E
        return E.run(name, model, tok, gates, items, a, dev, log)
    if name == 'swapkl':
        return swapkl(model, tok, gates, items, a, dev, log)
    if name == 'swapkl_fast':
        return swapkl_fast(model, tok, gates, items, a, dev, log)
    if name in SWK_VARIANTS:
        return swapkl(model, tok, gates, items, a, dev, log, SWK_VARIANTS[name])
    if name == 'poolelim':
        return poolelim(model, tok, gates, items, a, dev, log)
    if name == 'sleb_tail':
        return sleb_tail(model, tok, gates, items, a, dev, log)
    if name == 'frs':
        return frs(model, tok, gates, items, a, dev, log)
    if name == 'frs_selftest':
        return frs_selftest(model, tok, gates, items, a, dev, log)
    if name == 'nllg_greedy':
        return nllg_greedy(model, tok, gates, items, a, dev, log)
    if name == 'raceelim':
        return raceelim(model, tok, gates, items, a, dev, log)
    if name == 'raceelim_selftest':
        return raceelim_selftest(model, tok, gates, items, a, dev, log)
    raise SystemExit(f'không biết {name}')
