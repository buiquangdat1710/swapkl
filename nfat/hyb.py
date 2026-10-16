"""
Attention lai theo tầng, KHÔNG phụ thuộc kiến trúc — cho transformers >= 4.57 (Qwen3, Llama,
Qwen2, Mistral...). Thay cho bản vá Llama-4.43 trong nfat/layer_importance.py.

Cơ chế: đăng ký một hàm attention tên "hyb" qua AttentionInterface. Mọi mô hình HF hiện đại
gọi hàm này với q,k,v ĐÃ áp RoPE, nên một chỗ sửa dùng được cho mọi họ mô hình. Mask do
chính hàm này dựng (đăng ký hàm mask trả None để HF không dựng ma trận L×L).

Ba chế độ:
  off   attention nhân quả đầy đủ (SDPA, backend flash) — trùng mô hình gốc.
  hard  tầng trong `full` dùng đầy đủ; các tầng khác dùng cửa sổ W + S token đầu (sink /
        keep_first). `fa_decode=True`: bước giải mã dùng đầy đủ ở MỌI tầng — giao thức SWAA
        mà NLL-guided dùng.
  gate  o = o_SWA + z_l (o_full - o_SWA), z là tensor có gradient — dùng để đo độ quan trọng.

Nhánh SWA ở bước nạp prompt dùng FlexAttention với block mask thưa: không dựng ma trận L×L,
có lan truyền ngược, nên đo được ở 16k–32k token.

Giả định: batch 1, không padding, cache KHÔNG bị cắt (vị trí suy ra từ kích thước). Đủ cho
hiệu chỉnh, chấm logprob, và sinh văn bản với cache đầy đủ.
"""
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATE = dict(mode='off', W=None, S=0, full=frozenset(), fa_decode=False, z=None,
             ans_full=False, P=None)   # ans_full: truy vấn ở vị trí >= P (đáp án) luôn thấy đủ ngữ cảnh
_BLOCK_MASKS = {}
_FLEX = None


# ----------------------------------------------------------------------------- nhân attention
def _flex():
    """FlexAttention đã biên dịch, SHAPE ĐỘNG.

    dynamic=False sẽ biên dịch lại cho mỗi độ dài chuỗi; 64 mẫu hiệu chỉnh có 64 độ dài khác
    nhau, vượt cache_size_limit của dynamo, và khi đó flex rơi về chế độ eager — chế độ này
    dựng đủ ma trận L×L và tràn bộ nhớ ở 32k token.
    """
    global _FLEX
    if _FLEX is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _FLEX = torch.compile(flex_attention, dynamic=True)
    return _FLEX


def _swa_block_mask(Q, K, W, S, device, P=None):
    key = (Q, K, W, S, P, str(device))
    if key not in _BLOCK_MASKS:
        from torch.nn.attention.flex_attention import create_block_mask
        off = K - Q
        Pt = torch.tensor(K if P is None else P, device=device)   # tensor: không biên dịch lại theo P

        def mm(b, h, q_idx, kv_idx):
            qa = q_idx + off
            return (kv_idx <= qa) & (((qa - kv_idx) < W) | (kv_idx < S) | (qa >= Pt))
        _BLOCK_MASKS[key] = create_block_mask(mm, B=None, H=None, Q_LEN=Q, KV_LEN=K,
                                              device=device)
    return _BLOCK_MASKS[key]


def _full(q, k, v, scale):
    Q, K = q.shape[-2], k.shape[-2]
    if Q == K:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale,
                                              enable_gqa=True)
    if Q == 1:                       # giải mã: truy vấn là vị trí cuối, thấy mọi khoá
        return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
    qpos = torch.arange(K - Q, K, device=q.device).view(-1, 1)
    kpos = torch.arange(K, device=q.device).view(1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=(kpos <= qpos), scale=scale,
                                          enable_gqa=True)


def _swa(q, k, v, scale, W, S, P=None):
    Q, K = q.shape[-2], k.shape[-2]
    if K <= W:                       # cửa sổ phủ hết -> trùng attention đầy đủ
        return _full(q, k, v, scale)
    if Q <= 64:                      # giải mã / đoạn ngắn: mask nhỏ, SDPA là đủ
        qpos = torch.arange(K - Q, K, device=q.device).view(-1, 1)
        kpos = torch.arange(K, device=q.device).view(1, -1)
        keep = (kpos <= qpos) & (((qpos - kpos) < W) | (kpos < S))
        if P is not None:
            keep = keep | ((kpos <= qpos) & (qpos >= P))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=scale,
                                              enable_gqa=True)
    bm = _swa_block_mask(Q, K, W, S, q.device, P)
    # fp32 + head_dim 128 vượt shared memory của L40S (147 KB > 99 KB) với block mặc định
    ko = ({'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 1}
          if q.dtype == torch.float32 else None)
    return _flex()(q, k, v, block_mask=bm, scale=scale, enable_gqa=True, kernel_options=ko)


def hyb_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                  **kwargs):
    """Chữ ký giống sdpa_attention_forward của transformers. Trả (B, Q, H, D), None."""
    st = STATE
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    if st.get('probe') is not None:
        st['probe'].get('fn', _lazy_probe)(module.layer_idx, query, key, scale, st['probe'])
    if st['mode'] == 'off':
        o = _full(query, key, value, scale)
    else:
        l = module.layer_idx
        W, S = st['W'], st['S']
        P = st['P'] if st['ans_full'] else None
        decode = query.shape[-2] == 1 and key.shape[-2] > 1
        if st['mode'] == 'hard':
            if l in st['full'] or (decode and st['fa_decode']):
                o = _full(query, key, value, scale)
            else:
                o = _swa(query, key, value, scale, W, S, P)
        else:                                                    # gate
            of = _full(query, key, value, scale)
            os_ = _swa(query, key, value, scale, W, S, P)
            z = st['z'][l]
            o = (os_.float() + z * (of.float() - os_.float())).to(query.dtype)
    return o.transpose(1, 2).contiguous(), None


@torch.no_grad()
def _lazy_probe(l, q, k, scale, pr):
    """Lazy ratio của LightTransfer (theo get_lazy_ratio.py của SWAA): với w_last truy vấn cuối,
    log[ khối lượng attention trên (w_sink token đầu ∪ w_recent khoá cuối) / tổng khối lượng ],
    trung bình trên các đầu và truy vấn. Ghi vào pr['scores'][l]."""
    wl, S, R = pr['w_last'], pr['w_sink'], pr['w_recent']
    H, K = q.shape[1], k.shape[-2]
    ql = q[:, :, -wl:].float()
    kk = k.float().repeat_interleave(H // k.shape[1], dim=1)
    sc = torch.matmul(ql, kk.transpose(-1, -2)) * scale                    # (B,H,wl,K)
    qpos = torch.arange(K - wl, K, device=q.device).view(-1, 1)
    kpos = torch.arange(K, device=q.device).view(1, -1)
    sc = sc.masked_fill(kpos > qpos, float('-inf'))
    lse_all = torch.logsumexp(sc, -1)
    idx = torch.cat([torch.arange(S, device=q.device), torch.arange(K - R, K, device=q.device)])
    lse_loc = torch.logsumexp(sc[..., idx], -1)
    pr['scores'][l] = float((lse_loc - lse_all).mean())


def register():
    from transformers import AttentionInterface
    AttentionInterface.register('hyb', hyb_attention)
    try:                                   # để HF không dựng mask L×L cho implementation này
        from transformers.masking_utils import AttentionMaskInterface
        AttentionMaskInterface.register('hyb', lambda *a, **k: None)
    except Exception:
        pass


def set_mode(mode, W=None, S=0, full=(), fa_decode=False, z=None):
    STATE.update(mode=mode, W=W, S=S, full=frozenset(int(i) for i in full),
                 fa_decode=fa_decode, z=z)


def load(model_id, dtype=torch.bfloat16, device='cuda', attn='hyb'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if attn == 'hyb':
        register()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                 attn_implementation=attn).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def n_layers(model):
    return model.config.num_hidden_layers


# ----------------------------------------------------------------------------- mất mát
def answer_logprobs(model, it, device):
    """Log-xác suất từng token đáp án; chỉ chiếu lm_head trên các vị trí đáp án."""
    ids = torch.tensor([it['ids'] + it['gold']], device=device)
    n = len(it['gold'])
    STATE['P'] = len(it['ids'])          # chỉ có tác dụng khi STATE['ans_full']
    try:
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=n + 1).logits[0, :-1].float()
    finally:
        STATE['P'] = None
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(it['gold'], device=device).unsqueeze(-1)
    return lp.gather(-1, tgt).squeeze(-1)


def answer_nll(model, items, device):
    tot = 0.0
    for it in items:
        tot = tot + (-answer_logprobs(model, it, device).mean())
    return tot / len(items)


# ----------------------------------------------------------------------------- thước đo
class Gates:
    def __init__(self, model, W, S, device):
        self.n = n_layers(model)
        self.W, self.S = W, S
        self.z = torch.ones(self.n, dtype=torch.float32, device=device, requires_grad=True)

    def set(self, value):
        with torch.no_grad():
            if np.isscalar(value):
                self.z.fill_(float(value))
            else:
                self.z.copy_(torch.as_tensor(value, dtype=torch.float32, device=self.z.device))
        set_mode('gate', self.W, self.S, z=self.z)


def grad_at(model, gates, items, t, device):
    gates.set(t)
    gates.z.grad = None
    for it in items:
        (answer_nll(model, [it], device) / len(items)).backward()
    return -gates.z.grad.detach().float().cpu().numpy().copy()


def integrated_gradients(model, gates, items, steps, device, log=print):
    acc = np.zeros(gates.n)
    for i in range(steps):
        t = (i + 0.5) / steps
        acc += grad_at(model, gates, items, t, device) / steps
        log(f'   ig bước {i + 1}/{steps} (t={t:.3f})')
    return acc


def ig_greedy(model, gates, items, device, k, steps, log=print):
    S, order = [], []
    for step in range(k):
        acc = np.zeros(gates.n)
        for i in range(steps):
            z = np.full(gates.n, (i + 0.5) / steps)
            z[S] = 1.0
            acc += grad_at(model, gates, items, z, device) / steps
        acc[S] = -np.inf
        pick = int(np.argmax(acc))
        S.append(pick); order.append(pick)
        log(f'   ig_greedy bước {step + 1}/{k}: +tầng {pick} (công {acc[pick]:+.4f}) -> {sorted(S)}')
    return order


@torch.no_grad()
def nll_full(model, gates, items, device, log=print):
    """Hạ riêng tầng l từ mô hình ĐẦY ĐỦ: L(tầng l -> SWA, còn lại đầy đủ) - L(đầy đủ).
    LƯU Ý: đây KHÔNG phải NLL-guided (bài đó đo từ mô hình toàn SWA, xem nll_swa)."""
    gates.set(1.0)
    ref = float(answer_nll(model, items, device))
    out = np.zeros(gates.n)
    for l in range(gates.n):
        z = np.ones(gates.n); z[l] = 0.0
        gates.set(z)
        out[l] = float(answer_nll(model, items, device)) - ref
        log(f'   nll_full tầng {l:2d}: {out[l]:+.4f}')
    gates.set(1.0)
    return out


@torch.no_grad()
def nll_swa(model, gates, items, device, log=print):
    """Đúng phép đo của NLL-guided (Eq. 1 của bài): mọi tầng khác là SWA,
    Δ_l = L(tầng l SWA) - L(tầng l đầy đủ) = L(0) - L(e_l)."""
    gates.set(0.0)
    ref = float(answer_nll(model, items, device))
    out = np.zeros(gates.n)
    for l in range(gates.n):
        z = np.zeros(gates.n); z[l] = 1.0
        gates.set(z)
        out[l] = ref - float(answer_nll(model, items, device))
        log(f'   nll_swa tầng {l:2d}: {out[l]:+.4f}')
    gates.set(0.0)
    return out


def periodic(n, k):
    """Chu kỳ đều, bắt đầu từ tầng 0 — đúng baseline 1/4-FA của NLL-guided ([0,4,...,32])."""
    step = n // k
    return list(range(0, n, step))[:k]


def lighttransfer(path, k):
    """k tầng ÍT 'lazy' nhất theo tệp tần suất của SWAA/LightTransfer."""
    import json
    d = json.load(open(path))
    fr = sorted(d['layer_frequencies'], key=lambda r: (r['frequency'], r['layer_idx']))
    return sorted(r['layer_idx'] for r in fr[:k])
