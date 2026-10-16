"""
Sáu baseline ĐÃ CÔNG BỐ cho bài toán chọn k tầng giữ attention đầy đủ (còn lại SWA+sink), không
huấn luyện trọng số. Mỗi phương pháp gốc được đưa về MỨC TẦNG rồi lấy top-k; mọi thứ khác (mô hình,
dữ liệu hiệu chỉnh, W, S, k, giao thức đánh giá) giữ đúng như ig / ig_greedy.

  duo      DuoAttention (Xiao et al., ICLR 2025). Trọng số đóng băng, HỌC cổng z in [0,1] theo tầng
           (bài gốc: theo đầu) bằng loss chưng cất = MSE trạng thái ẩn cuối (đầy đủ vs có cổng) tại
           các vị trí nhãn + reg_weight * mean|z|; AdamW lr 0.02, lịch warmup/decay 1/5, kẹp [0,1]
           sau mỗi bước — đúng duo_attn/train.py. Số bước = ngân sách tính toán của ig.
  rethead  Retrieval Head (Wu et al., ICLR 2025). Điểm truy hồi của mỗi đầu = tỉ lệ token đáp án mà
           argmax attention của đầu rơi ĐÚNG vào token giống hệt trong kim (copy-paste), trên prompt
           NIAH, mô hình gốc. Điểm tầng = tổng điểm các đầu.
  bi       Block Influence của ShortGPT (Men et al., Findings ACL 2025) áp riêng cho khối attention như
           Attention Drop (He et al., "What Matters in Transformers", TMLR): 1 - cos(X, X + Attn(X)),
           trung bình mọi token, mô hình gốc.
  moa      MoA (Fu et al., CoLM 2025): ảnh hưởng bậc 1 lên loss khi bỏ các phần tử attention mà mask
           cửa sổ cắt đi, sum_ij dL/dA_ij * dA_ij. Gộp theo tầng thì đúng bằng -dL/dz_l tại z=1
           (vì dL/dA_ij = dL/do_i . v_j và sum_j dA_ij v_j = o^SWA_i - o^full_i). Cũng là điểm
           Taylor của Michel et al. (NeurIPS 2019). 1 lượt xuôi + 1 ngược.
  cake     CAKE (Qin et al., ICLR 2025): P_l = H^(1/tau1) * V^(1/tau2) trên A[-Sw:, :-Sw] của
           prompt (Sw=32): H = entropy theo hàng (độ phân tán), V = tổng phương sai theo cột (độ
           dịch chuyển), trung bình các đầu. tau1 = tau2 = 1 (không grid-search).
  sleb     SLEB (Song et al., ICML 2024): LOẠI BỎ THAM LAM — xuất phát từ mô hình gốc, mỗi vòng thử
           hạ từng tầng còn đầy đủ xuống SWA, hạ tầng làm NLL đáp án tăng ít nhất, lặp tới khi còn
           k tầng. Tính lại sau mỗi bước nên có tính tương tác. Tốn sum_{j=k+1..L} j lượt xuôi.

  opg      (ĐỀ XUẤT, không phải baseline) tham lam tại điểm vận hành: gradient mở rộng đa tuyến tính tại
           các đỉnh ngẫu nhiên cỡ k, mục tiêu kép MSE trạng thái ẩn + NLL đáp án (chuẩn hoá theo hai đầu
           mút), rồi hoán đổi 1-đổi-1 ở biên. Xem score_opg.
  opgm     như opg nhưng chỉ MSE: dùng khi triển khai giải mã ĐẦY ĐỦ (token sinh ra tự truy hồi được, nên
           thành phần NLL chỉ kéo vào tầng chép đáp án thừa — kiểm chứng: opg 47,8 so với 54,2 trên
           LongMemEval Llama-8B-Instruct).

    python nfat/baselines.py --model Qwen/Qwen3-4B-Thinking-2507 --data longalign --W 2048 --S 10 \
        --k 9 --scorers bi,moa,cake,rethead,duo --tag q4t-w2048
Ghi results/layer_importance.csv (pick_layers.py đọc được) và results/select_cost.csv (thời gian).
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat import hyb                                               # noqa: E402
from nfat.csv_append import append_row                             # noqa: E402
from nfat.niah_utils import build, load_haystack_words            # noqa: E402


# ----------------------------------------------------------------------------- dữ liệu NIAH
def needle_span(ids, tok, gold):
    """Giống nfat/layer_importance.py."""
    for prefix in (' ', ''):
        g = tok.encode(prefix + gold, add_special_tokens=False)
        if not g:
            continue
        n = len(g)
        for s in range(len(ids) - n, -1, -1):
            if ids[s:s + n] == g:
                return s, s + n
    return None


def niah_items(tok, variant, L, W, depths, n_per_depth, seed, haystack):
    """Sao y make_items của nfat/layer_importance.py (cùng seed -> cùng prompt), kim ngoài cửa sổ."""
    rng = random.Random(seed)
    words = load_haystack_words(haystack)
    items = []
    for d in depths:
        for _ in range(n_per_depth):
            ids, golds, _ = build(tok, variant, L, d, rng, words)
            gid = tok.encode(' ' + ' '.join(str(g) for g in golds), add_special_tokens=False)
            span = needle_span(ids, tok, str(golds[0]))
            if span is not None and span[0] >= len(ids) - W:
                continue
            items.append(dict(ids=ids, gold=gid, span=span, value=str(golds[0])))
    return items


# ----------------------------------------------------------------------------- probe attention
def _attn_rows(q, k, scale, qpos):
    """Xác suất attention đầy đủ (nhân quả) của các truy vấn ở vị trí qpos: (H, n, K), fp32."""
    H, K = q.shape[1], k.shape[-2]
    qp = torch.as_tensor(qpos, device=q.device)
    ql = q[0][:, qp].float()                                                 # (H,n,D)
    kk = k[0].float().repeat_interleave(H // k.shape[1], dim=0)             # (H,K,D)
    sc = torch.matmul(ql, kk.transpose(-1, -2)) * scale
    sc = sc.masked_fill(torch.arange(K, device=q.device).view(1, 1, -1) > qp.view(1, -1, 1),
                        float('-inf'))
    return torch.softmax(sc, -1)


@torch.no_grad()
def _rethead_probe(l, q, k, scale, pr):
    A = _attn_rows(q, k, scale, pr['qpos'])
    j = A.argmax(-1)                                                          # (H,n)
    s0, s1 = pr['span']
    hit = (j >= s0) & (j < s1) & (pr['ids'][j] == pr['tgt'].view(1, -1))
    pr['scores'][l] = pr['scores'].get(l, 0) + hit.float().mean(-1).cpu().numpy()  # (H,)


@torch.no_grad()
def _cake_probe(l, q, k, scale, pr):
    K, Sw = k.shape[-2], pr['Sw']
    A = _attn_rows(q, k, scale, list(range(K - Sw, K)))[:, :, :K - Sw]       # A[-Sw:, :-Sw]
    Hs = -(A * torch.log(A.clamp_min(1e-30))).sum(-1).sum(-1)                # (H,) tổng entropy các hàng
    Vs = A.var(dim=1, unbiased=False).sum(-1)                                 # (H,) tổng phương sai cột
    Hm, Vm = float(Hs.mean()), float(Vs.mean())
    pr['scores'][l] = pr['scores'].get(l, 0) + Hm ** (1 / pr['tau1']) * Vm ** (1 / pr['tau2'])


# ----------------------------------------------------------------------------- thước đo
@torch.no_grad()
def score_rethead(model, tok, items, dev, log):
    """items phải có 'span' (NIAH). Mô hình gốc (hyb off)."""
    hyb.set_mode('off')
    n = hyb.n_layers(model)
    pr = dict(fn=_rethead_probe, scores={})
    hyb.STATE['probe'] = pr
    used = 0
    try:
        for it in items:
            if it.get('span') is None:
                continue
            full = it['ids'] + it['gold']
            P, m = len(it['ids']), len(it['gold'])
            pr['ids'] = torch.tensor(full, device=dev)
            pr['qpos'] = list(range(P - 1, P - 1 + m))
            pr['tgt'] = torch.tensor(it['gold'], device=dev)
            pr['span'] = it['span']
            model(input_ids=pr['ids'][None], use_cache=False, logits_to_keep=1)
            used += 1
    finally:
        hyb.STATE['probe'] = None
    heads = np.stack([pr['scores'][l] / used for l in range(n)])             # (L,H)
    log(f'   rethead: {used} prompt, số đầu có điểm > 0.1 theo tầng: '
        f'{(heads > 0.1).sum(1).tolist()}')
    return heads.sum(1)


@torch.no_grad()
def score_cake(model, items, dev, Sw=32, tau1=1.0, tau2=1.0):
    hyb.set_mode('off')
    n = hyb.n_layers(model)
    pr = dict(fn=_cake_probe, scores={}, Sw=Sw, tau1=tau1, tau2=tau2)
    hyb.STATE['probe'] = pr
    try:
        for it in items:
            model(input_ids=torch.tensor([it['ids']], device=dev), use_cache=False, logits_to_keep=1)
    finally:
        hyb.STATE['probe'] = None
    return np.array([pr['scores'][l] / len(items) for l in range(n)])


@torch.no_grad()
def score_bi(model, items, dev):
    """1 - cos(X, X + Attn(X)) cho khối attention của mỗi tầng; X là dòng dư vào tầng."""
    hyb.set_mode('off')
    layers = model.model.layers
    n = len(layers)
    acc, cnt, cur = np.zeros(n), 0, {}
    hs = []
    for i, lay in enumerate(layers):
        def pre(mod, args, kwargs, i=i):
            cur[i] = args[0] if args else kwargs['hidden_states']

        def post(mod, args, out, i=i):
            x = cur.pop(i)[0].float()
            a = (out[0] if isinstance(out, tuple) else out)[0].float()
            acc[i] += float((1 - F.cosine_similarity(x, x + a, dim=-1)).mean())
        hs.append(lay.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(lay.self_attn.register_forward_hook(post))
    try:
        for it in items:
            model(input_ids=torch.tensor([it['ids'] + it['gold']], device=dev), use_cache=False,
                  logits_to_keep=1)
            cnt += 1
    finally:
        for h in hs:
            h.remove()
    return acc / cnt


def score_moa(model, gates, items, dev):
    return hyb.grad_at(model, gates, items, 1.0, dev)


def score_duo(model, gates, items, dev, steps, lr=0.02, reg_weight=0.05, seed=0, log=print):
    """DuoAttention mức tầng. Trả (z cuối, giá trị trước kẹp của bước cuối) để phá hoà."""
    rng = random.Random(seed)
    order = []
    while len(order) < steps:
        o = list(range(len(items))); rng.shuffle(o); order += o
    order = order[:steps]
    gates.set(1.0)
    z = gates.z
    opt = torch.optim.AdamW([z], lr=lr, weight_decay=0)
    fifth = max(1, steps // 5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, max((s + 1) / fifth, 0.1),
                                                               max((steps - s) / fifth, 0.1)))
    raw = z.detach().clone()
    base = model.model
    for s, idx in enumerate(order):
        it = items[idx]
        ids = torch.tensor([it['ids'] + it['gold']], device=dev)
        P, m = len(it['ids']), len(it['gold'])
        with torch.no_grad():
            hyb.set_mode('off')
            ref = base(input_ids=ids, use_cache=False).last_hidden_state[0, P - 1:P - 1 + m].float()
        hyb.set_mode('gate', gates.W, gates.S, z=z)
        out = base(input_ids=ids, use_cache=False).last_hidden_state[0, P - 1:P - 1 + m].float()
        distill = (ref - out).pow(2).mean(-1).mean()
        reg = z.abs().mean()
        loss = distill + reg_weight * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sch.step()
        with torch.no_grad():
            raw = z.detach().clone()
            z.clamp_(0, 1)
        if s % 16 == 0 or s == steps - 1:
            log(f'   duo bước {s + 1}/{steps}: distill {float(distill):.4f} reg {float(reg):.3f} '
                f'| z = {np.round(z.detach().cpu().numpy(), 2).tolist()}')
    hyb.set_mode('gate', gates.W, gates.S, z=z)
    zf = z.detach().cpu().numpy().copy()
    # phá hoà giữa các cổng cùng bị kẹp ở 1 (hoặc 0) bằng giá trị trước kẹp, thang rất nhỏ
    return zf + 1e-3 * np.tanh(raw.cpu().numpy() - zf)


@torch.no_grad()
def score_sleb(model, items, dev, W, S, k, log=print):
    n = hyb.n_layers(model)
    full = set(range(n))
    removed = []

    def nll(fs):
        hyb.set_mode('hard', W, S, full=fs)
        return float(hyb.answer_nll(model, items, dev))
    base = nll(full)
    last = {}
    while len(full) > k:
        cand = {}
        for l in sorted(full):
            cand[l] = nll(full - {l})
        l = min(cand, key=cand.get)
        full.remove(l); removed.append(l); last = cand
        log(f'   sleb: hạ tầng {l} (NLL {cand[l]:.4f}, gốc {base:.4f}) -> còn {len(full)}')
    v = np.zeros(n)
    for pos, l in enumerate(removed):
        v[l] = pos                                     # hạ càng muộn càng quan trọng
    for l in full:     # k tầng còn lại LUÔN đứng trên mọi tầng đã hạ (sửa 10-04: NLL ứng viên < NLL gốc từng đẩy chúng xuống dưới)
        v[l] = len(removed) + 1 + (1.0 / (1.0 + np.exp(-(last.get(l, base) - base) * 10)) if last else 0.5)
    hyb.set_mode('off')
    return v


# ----------------------------------------------------------------------------- phương pháp đề xuất
class _DualObjective:
    """D(z) = MSE(z)/MSE(0) + (NLL(z) - NLL(1)) / (NLL(0) - NLL(1)), trên vị trí đáp án.
    MSE trạng thái ẩn cuối giữ tầng tích hợp ngữ cảnh (QA dài), NLL giữ mạch chép đáp án (truy hồi).
    Chuẩn hoá theo hai đầu mút để hai thành phần cùng thang, không cần siêu tham số."""

    def __init__(self, model, items, dev, W, S, w_mse=1.0, w_nll=1.0, pos='ans', stride=16):
        # pos='ctx': MSE trên trạng thái ẩn của các token NGỮ CẢNH ở ngoài cửa sổ đầu (vị trí W..P-2,
        # bước stride) — đúng thứ SWA làm hỏng khi triển khai giải mã đầy đủ (bộ đệm KV của ngữ cảnh).
        # Kiểm chứng: Spearman với độ chính xác LongMemEval +0,97 (mse tại đáp án +0,86, NLL đáp án +0,52).
        self.w_mse, self.w_nll, self.pos, self.stride = w_mse, w_nll, pos, stride
        self.base, self.head, self.dev, self.W, self.S = model.model, model.lm_head, dev, W, S
        self.items, self.ref = items, []
        with torch.no_grad():
            hyb.set_mode('off')
            for it in items:                         # trạng thái ẩn tham chiếu của mô hình gốc
                self.ref.append(self._hidden(it).to(torch.bfloat16))
            nll1 = float(np.mean([float(self._nll(self.ref[i][-len(it['gold']):].float(), it))
                                  for i, it in enumerate(items)]))
            hyb.set_mode('hard', W, S, full=())
            mse0, nll0 = [], []
            for i, it in enumerate(items):
                h = self._hidden(it).float()
                mse0.append(float(self._mse(h, i))); nll0.append(float(self._nll(h[-len(it['gold']):], it)))
        hyb.set_mode('off')
        self.mse0, self.nll0, self.nll1 = float(np.mean(mse0)), float(np.mean(nll0)), nll1
        self.gap = max(self.nll0 - self.nll1, 1e-4)

    def _hidden(self, it):
        ids = torch.tensor([it['ids'] + it['gold']], device=self.dev)
        P, m = len(it['ids']), len(it['gold'])
        h = self.base(input_ids=ids, use_cache=False).last_hidden_state[0]
        if self.pos == 'ctx':          # vị trí ngữ cảnh rồi tới vị trí đáp án (NLL dùng m vị trí cuối)
            return torch.cat([h[self.W:P - 1:self.stride], h[P - 1:P - 1 + m]])
        return h[P - 1:P - 1 + m]

    def _mse(self, h, i):
        r = self.ref[i].float(); m = len(self.items[i]['gold'])
        if self.pos == 'ctx':
            return (h[:-m].float() - r[:-m]).pow(2).mean()
        return (h.float() - r).pow(2).mean()

    def _nll(self, h, it):
        lg = torch.log_softmax(self.head(h.to(self.head.weight.dtype)).float(), -1)
        return -lg.gather(-1, torch.tensor(it['gold'], device=self.dev)[:, None]).mean()

    def value(self, i, h):
        it = self.items[i]
        v = self.w_mse * self._mse(h, i) / self.mse0
        if self.w_nll:
            v = v + self.w_nll * (self._nll(h[-len(it['gold']):], it) - self.nll1) / self.gap
        return v

    def grad_at(self, gates, zval, idx):
        """-dD/dz tại đỉnh zval, trung bình trên các mẫu idx."""
        gates.set(zval); gates.z.grad = None
        for i in idx:
            (self.value(i, self._hidden(self.items[i])) / len(idx)).backward()
        return -gates.z.grad.detach().float().cpu().numpy().copy()

    @torch.no_grad()
    def at_vertex(self, full, idx):
        hyb.set_mode('hard', self.W, self.S, full=full)
        return float(np.mean([float(self.value(i, self._hidden(self.items[i]))) for i in idx]))


def score_opg(model, gates, items, dev, k, W, S, n_masks=4, n_items=16, swap_items=8,
              swap_rounds=2, n_border=3, seed=0, obj='dual', log=print):
    """Chọn tham lam tại điểm vận hành (gradient mở rộng đa tuyến tính) với mục tiêu kép, rồi hoán đổi.

    Bước j (đã chọn S, |S| = j): lấy n_masks đỉnh ngẫu nhiên giữ S = 1 và mở thêm đúng k - j tầng
    trong số còn lại, mỗi đỉnh tính -dD/dz trên n_items mẫu ngẫu nhiên; thêm tầng có gradient trung
    bình lớn nhất. Cuối cùng thử hoán đổi 1-đổi-1 giữa n_border tầng biên trong và ngoài tập,
    đánh giá D thật tại đỉnh trên swap_items mẫu cố định, nhận nếu D giảm."""
    rng = np.random.default_rng(seed)
    L = gates.n
    obj = _DualObjective(model, items, dev, W, S, w_nll=0.0 if obj == 'mse' else 1.0)
    log(f'   opg: MSE(0)={obj.mse0:.4f} NLL(0)={obj.nll0:.4f} NLL(1)={obj.nll1:.4f}')
    sel, last = [], np.zeros(L)
    for step in range(k):
        rest = [l for l in range(L) if l not in sel]
        acc = np.zeros(L)
        for _ in range(n_masks):
            z = np.zeros(L); z[sel] = 1.0
            z[rng.choice(rest, k - len(sel), replace=False)] = 1.0
            idx = rng.choice(len(items), min(n_items, len(items)), replace=False)
            acc += obj.grad_at(gates, z, idx) / n_masks
        cand = acc.copy(); cand[sel] = -np.inf
        pick = int(np.argmax(cand)); sel.append(pick); last = acc
        log(f'   opg bước {step + 1}/{k}: +tầng {pick} ({acc[pick]:+.4f}) -> {sorted(sel)}')
    idx = rng.choice(len(items), min(swap_items, len(items)), replace=False)
    cur = obj.at_vertex(sel, idx)
    for r in range(swap_rounds):
        inner = sorted(sel, key=lambda l: last[l])[:n_border]            # yếu nhất trong tập
        outer = [l for l in np.argsort(-last) if l not in sel][:n_border]  # mạnh nhất ngoài tập
        best = (cur, None)
        for a in inner:
            for b in outer:
                v = obj.at_vertex([l for l in sel if l != a] + [int(b)], idx)
                if v < best[0]:
                    best = (v, (a, int(b)))
        log(f'   opg hoán đổi vòng {r + 1}: D {cur:.4f} -> {best[0]:.4f} {best[1]}')
        if best[1] is None:
            break
        a, b = best[1]; sel = [l for l in sel if l != a] + [b]; cur = best[0]
    hyb.set_mode('off')
    v = last.copy()
    v[sel] = 1e3 + np.arange(len(sel))[::-1]          # k tầng đã chọn đứng đầu
    return v


def score_opge(model, gates, items, dev, k, W, S, obj='mse', n_items=16, n_check=3, check_items=8,
               seed=0, log=print):
    """Loại bỏ LÙI có gradient dẫn đường. Xuất phát từ mô hình gốc (mọi tầng đầy đủ); mỗi bước tính
    -dD/dz tại đỉnh hiện tại (xấp xỉ bậc 1 chi phí hạ từng tầng xuống SWA), lấy n_check tầng rẻ nhất,
    đo D thật tại đỉnh bỏ từng tầng đó trên check_items mẫu cố định, hạ tầng rẻ nhất; tới khi còn k.
    Đi lùi giữ được mạch dạng "và" (tầng chép muộn chỉ có ích khi tầng sớm còn mở) mà chọn tiến bỏ sót.
    obj: 'mse' khi triển khai giải mã đầy đủ, 'nll' khi giải mã SWA (xem docstring đầu tệp)."""
    rng = np.random.default_rng(seed)
    L = gates.n
    D = _DualObjective(model, items, dev, W, S, w_mse=0.0 if obj == 'nll' else 1.0,
                       w_nll=0.0 if obj in ('mse', 'ctx') else 1.0, pos='ctx' if obj == 'ctx' else 'ans')
    chk = rng.choice(len(items), min(check_items, len(items)), replace=False)
    full, removed, g = list(range(L)), [], np.zeros(L)
    while len(full) > k:
        z = np.zeros(L); z[full] = 1.0
        idx = rng.choice(len(items), min(n_items, len(items)), replace=False)
        g = D.grad_at(gates, z, idx)                     # g_l ~ D tăng bao nhiêu nếu hạ tầng l
        cand = sorted(full, key=lambda l: g[l])[:n_check]
        vals = {l: D.at_vertex([x for x in full if x != l], chk) for l in cand}
        l = min(vals, key=vals.get)
        full.remove(l); removed.append(l)
        log(f'   opge: hạ tầng {l} (D {vals[l]:.4f}, ứng viên {cand}) -> còn {len(full)}')
    hyb.set_mode('off')
    v = np.zeros(L)
    for pos, l in enumerate(removed):
        v[l] = pos
    for l in full:
        v[l] = len(removed) + 1 + g[l]
    return v


def score_opgc(model, gates, items, dev, k, W, S, n_masks=4, n_items=16, swap_items=16,
               swap_rounds=4, n_border=4, seed=0, log=print):
    """Phương pháp cho triển khai giải mã đầy đủ. Mục tiêu: méo biểu diễn của token NGỮ CẢNH
    (Spearman +0,97 với LongMemEval). (1) gradient tại n_masks đỉnh ngẫu nhiên cỡ k (điểm vận hành,
    nơi gradient còn thông tin — gần z=1 thì mù) cho điểm khởi đầu, lấy top-k; (2) tìm kiếm cục bộ
    CHÍNH XÁC: mỗi vòng thử mọi hoán đổi giữa n_border tầng yếu nhất trong tập và n_border tầng mạnh
    nhất ngoài tập, đo mục tiêu thật tại đỉnh, nhận hoán đổi tốt nhất nếu giảm."""
    rng = np.random.default_rng(seed)
    L = gates.n
    D = _DualObjective(model, items, dev, W, S, w_mse=1.0, w_nll=0.0, pos='ctx')
    log(f'   opgc: MSE_ctx(0)={D.mse0:.4f}')
    g = np.zeros(L)
    for _ in range(n_masks):
        z = np.zeros(L); z[rng.choice(L, k, replace=False)] = 1.0
        idx = rng.choice(len(items), min(n_items, len(items)), replace=False)
        g += D.grad_at(gates, z, idx) / n_masks
    sel = sorted(np.argsort(-g)[:k].tolist())
    chk = rng.choice(len(items), min(swap_items, len(items)), replace=False)
    cur = D.at_vertex(sel, chk)
    log(f'   opgc khởi đầu {sel}: D {cur:.4f}')
    for r in range(swap_rounds):
        inner = sorted(sel, key=lambda l: g[l])[:n_border]
        outer = [int(l) for l in np.argsort(-g) if l not in sel][:n_border]
        best = (cur, None)
        for a in inner:
            for b in outer:
                v = D.at_vertex([l for l in sel if l != a] + [b], chk)
                if v < best[0]:
                    best = (v, (a, b))
        log(f'   opgc hoán đổi vòng {r + 1}: D {cur:.4f} -> {best[0]:.4f} {best[1]}')
        if best[1] is None:
            break
        a, b = best[1]
        sel = sorted([l for l in sel if l != a] + [b]); cur = best[0]
        g[b], g[a] = max(g[b], g[a]), min(g[b], g[a])        # giữ thứ tự biên nhất quán với tập mới
    hyb.set_mode('off')
    v = g.copy(); v[sel] = 1e3 + g[sel]
    return v


# ----------------------------------------------------------------------------- chạy
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', choices=['longalign', 'niah'], required=True)
    p.add_argument('--W', type=int, required=True)
    p.add_argument('--S', type=int, required=True)
    p.add_argument('--k', type=int, required=True, help='chỉ dùng cho sleb (dừng ở k tầng)')
    p.add_argument('--scorers', default='bi,moa,cake,rethead,duo')
    p.add_argument('--tag', required=True)
    p.add_argument('--seed', type=int, default=1234)
    # longalign (như hyb_select)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--min_len', type=int, default=16000)
    p.add_argument('--max_len', type=int, default=32000)
    # niah (như layer_importance: s2, 4096, 8/độ sâu, 4 độ sâu)
    p.add_argument('--niah_L', type=int, default=4096)
    p.add_argument('--niah_n', type=int, default=8)
    p.add_argument('--niah_variant', default='s2')
    p.add_argument('--depths', default='0,0.25,0.5,0.75')
    p.add_argument('--haystack', default='data/haystack_text')
    # tham số riêng
    p.add_argument('--duo_steps', type=int, default=256)
    p.add_argument('--duo_reg', type=float, default=0.05)
    p.add_argument('--sleb_n', type=int, default=8, help='số mẫu hiệu chỉnh cho sleb')
    p.add_argument('--opg_masks', type=int, default=4, help='số đỉnh ngẫu nhiên mỗi bước opg')
    p.add_argument('--opg_items', type=int, default=16, help='số mẫu mỗi đỉnh opg')
    p.add_argument('--out', default='results/layer_importance.csv')
    p.add_argument('--cost_out', default='results/select_cost.csv')
    a = p.parse_args()

    dev = 'cuda'
    model, tok = hyb.load(a.model, attn='hyb')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    n = hyb.n_layers(model)
    depths = [float(x) for x in a.depths.split(',')]
    if a.data == 'longalign':
        from nfat.hyb_select import calib_items
        items = calib_items(tok, a.n, a.min_len, a.max_len, 20, 256, a.seed)
    else:
        items = niah_items(tok, a.niah_variant, a.niah_L, a.W, depths, a.niah_n, a.seed, a.haystack)
    print(f'-> {len(items)} mẫu hiệu chỉnh ({a.data}), {n} tầng, W={a.W}, S={a.S}', flush=True)
    gates = hyb.Gates(model, a.W, a.S, dev)
    t0 = time.time()
    log = lambda s: print(f'{s}  [{time.time() - t0:.0f}s]', flush=True)   # noqa: E731

    for name in [s for s in a.scorers.split(',') if s]:
        torch.cuda.synchronize(); t = time.time()
        n_used = len(items)
        if name == 'bi':
            v = score_bi(model, items, dev)
        elif name == 'moa':
            v = score_moa(model, gates, items, dev)
        elif name == 'cake':
            v = score_cake(model, items, dev)
        elif name == 'rethead':
            if a.data == 'niah':
                rh = items
            else:   # giao thức gốc của Retrieval Head: NIAH, ở đây dài 16K, kim ngoài cửa sổ
                rh = niah_items(tok, 's2', 16384, a.W, depths, a.niah_n, a.seed, a.haystack)
            n_used = len(rh)
            v = score_rethead(model, tok, rh, dev, log)
        elif name == 'duo':
            v = score_duo(model, gates, items, dev, a.duo_steps, reg_weight=a.duo_reg, log=log)
            n_used = a.duo_steps
        elif name == 'sleb':
            sub = items[::max(1, len(items) // a.sleb_n)][:a.sleb_n]   # rải đều các độ sâu
            n_used = len(sub)
            v = score_sleb(model, sub, dev, a.W, a.S, a.k, log)
        elif name == 'opgc':
            v = score_opgc(model, gates, items, dev, a.k, a.W, a.S, n_masks=a.opg_masks,
                           n_items=a.opg_items, log=log)
        elif name in ('opge', 'opgen'):   # opge: MSE (giải mã đầy đủ); opgen: NLL (giải mã SWA)
            v = score_opge(model, gates, items, dev, a.k, a.W, a.S,
                           obj='nll' if name == 'opgen' else 'ctx', n_items=a.opg_items, log=log)
        elif name in ('opg', 'opgm'):     # opgm: chỉ MSE, cho triển khai giải mã đầy đủ
            v = score_opg(model, gates, items, dev, a.k, a.W, a.S, n_masks=a.opg_masks,
                          n_items=a.opg_items, obj='mse' if name == 'opgm' else 'dual', log=log)
            n_used = len(items)
        else:
            sys.exit(f'không biết thước đo {name}')
        torch.cuda.synchronize(); sec = time.time() - t
        order = list(np.argsort(-v))
        log(f'[{name}] {sec / 60:.1f} phút | top-{a.k}: {sorted(int(x) for x in order[:a.k])}')
        rows = [dict(tag=a.tag, model_config=a.model, scorer=name, layer=l, score=float(v[l]),
                     rank=int(order.index(l)), window=a.W, n_sink=a.S, variant=a.data,
                     L=a.max_len if a.data == 'longalign' else a.niah_L, n_items=n_used,
                     seed=a.seed) for l in range(n)]
        append_row(pd.DataFrame(rows), a.out)
        append_row(pd.DataFrame([dict(tag=a.tag, model=a.model, scorer=name, seconds=sec,
                                      n_items=n_used, gpu=torch.cuda.get_device_name())]),
                   a.cost_out)
        hyb.set_mode('off')


if __name__ == '__main__':
    main()
