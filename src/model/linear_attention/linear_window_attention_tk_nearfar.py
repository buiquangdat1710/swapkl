"""
Near-Far Attention Transfer (NF-AT) for the LoLCATs linear + "terrace" sliding window layer.

Kiến trúc KHÔNG đổi so với `LolcatsTKWindowAttention` (cùng tham số, cùng state_dict).
Điểm mấu chốt: `hybrid_attention_quadratic` của LoLCATs dùng MẪU SỐ CHUNG, nên

    y_t = (sum_sm * o_near + sum_ln * o_far_hat) / (sum_sm + sum_ln)
        = (1 - f_hat_t) * o_near_t + f_hat_t * o_far_hat_t                       (eq. 19)

với  f_hat_t   = sum_ln / (sum_sm + sum_ln)            (cổng, không thêm tham số)
     o_near_t  = softmax chính xác trên nhóm GẦN      (không có tham số học)
     o_far_hat = sum_s a_ln[t,s] v_s / sum_ln         (bộ nhớ tuyến tính)

Phía thầy (softmax đầy đủ), với cùng phép chia GẦN/XA do `get_masks` định nghĩa:
     f_t     = sum_{s in XA} A[t,s]
     o_far_t = sum_{s in XA} A[t,s] v_s / f_t
     A_far_t = A[t, XA] / f_t

Loss (eq. 20):
     alpha * (f_hat - f)^2  +  beta * w_t ||o_far_hat - o_far||^2  +  gamma * w_t KL(A_far || A_far_hat)
với w_t = f_t (mặc định).

Các số hạng khác để so sánh / cắt bỏ (tất cả tính trong cùng một hàm):
     mse_full  : MSE(y_pred, y_true) trên đầu ra từng đầu  -> đúng LoLCATs (trainer distill_attention_xent_mse)
     xent_full : CE(A_true, A_pred) toàn hàng             -> đúng nhánh xent của LoLCATs/Hedgehog
"""
from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.cache_utils import Cache

from .linear_window_attention_tk import (
    LolcatsTKWindowAttention, get_masks, hybrid_attention_quadratic,
)
from .linear_attention import softmax_attention, LinearAttentionState

LOSS_KEYS = ('mse_full', 'xent_full', 'gate', 'far_mse', 'far_kl')
STAT_KEYS = ('f_mean', 'fhat_mean', 'gate_abs_err', 'far_ent_teacher', 'far_ent_student', 'far_top1', 'top1_mass_ratio')


# -----------------------------------------------------------------------------
# Near / far bookkeeping (identical to the architecture's own masks)
# -----------------------------------------------------------------------------
def terrace_masks(window_size: int, q_len: int, k_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Boolean (1,1,q,k) masks: NEAR = softmax window, FAR = linear part (both causal).
    NB: for max(q,k) <= window_size, LoLCATs' get_masks returns an all-causal linear
    mask as well (quirk: roll maps the single block onto itself, 1-2 != 0).
    For the *loss* we define FAR = empty in that case (a proper partition).
    """
    mw, ml = get_masks(window_size, q_len, k_len, device)
    mw, ml = mw.bool(), ml.bool()
    if max(q_len, k_len) <= window_size:
        ml = torch.zeros_like(ml)
    return mw, ml


# -----------------------------------------------------------------------------
# Loss for one group of heads (checkpointed by the caller)
# -----------------------------------------------------------------------------
def _nearfar_group_loss(q, k, v, f_q, f_k, window_factor, linear_factor,
                        mask_near, mask_far, far_weight_mode: str, f_min: float,
                        enabled: Tuple[bool, ...], perm: Optional[torch.Tensor],
                        eps: float = 1e-12):
    """
    Returns SUMS (not means) so that groups can be combined exactly:
      out[0:5]  = sums for LOSS_KEYS
      out[5:10] = sums for STAT_KEYS (over valid far rows)
      out[10]   = number of valid far rows
    All work in float32.
    """
    want_mse, want_xent, want_gate, want_fmse, want_fkl = enabled
    q, k, v = q.float(), k.float(), v.float()
    f_q, f_k = f_q.float(), f_k.float()
    d = k.shape[-1]
    L = q.shape[-2]
    logits = torch.einsum('bhmd,bhnd->bhmn', q, k) * (d ** -0.5)

    # ---------------- teacher (no grad) ----------------
    with torch.no_grad():
        causal = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        A = torch.softmax(logits.masked_fill(~causal, float('-inf')), dim=-1)
        A_far = A * mask_far                                     # b,h,L,L
        f = A_far.sum(-1)                                        # b,h,L
        y_true = A @ v                                           # b,h,L,d
        o_far = (A_far @ v) / f.clamp_min(eps)[..., None]
        row_has_far = mask_far.any(-1)                           # 1,1,L
        valid = row_has_far & (f > f_min)                        # b,h,L
        # per-row weights
        if far_weight_mode == 'f':
            w = f
        elif far_weight_mode == 'none':
            w = torch.ones_like(f)
        elif far_weight_mode == 'shuffle':   # ablation 5: same marginal, wrong tokens
            w = f.flatten(-1)[..., perm].view_as(f)
        else:
            raise ValueError(far_weight_mode)
        w = w * valid

    # ---------------- student (LoLCATs hybrid, same math) ----------------
    a_sm = logits.masked_fill(~mask_near, -1e8)
    a_sm = window_factor * torch.exp(a_sm - a_sm.amax(-1, keepdim=True))
    a_sm = a_sm * mask_near
    sum_sm = a_sm.sum(-1)                                        # b,h,L
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q, f_k)
    a_ln = linear_factor * a_ln * mask_far
    sum_ln = a_ln.sum(-1)
    den = sum_sm + sum_ln

    zero = logits.new_zeros(())
    out = []
    # (LoLCATs) MSE over full per-head output, mean over b,h,L,d later
    if want_mse:
        y_pred = ((a_sm + a_ln) @ v) / den[..., None]
        out.append(((y_pred - y_true) ** 2).sum())
    else:
        out.append(zero)
    # (LoLCATs xent / Hedgehog) full-row cross entropy, mean over b,h,L later
    if want_xent:
        a_pred = (a_sm + a_ln) / den[..., None]
        out.append(-(torch.xlogy(A, a_pred.clamp_min(eps))).sum())
    else:
        out.append(zero)

    f_hat = sum_ln / den
    # (a) gate
    if want_gate:
        out.append((((f_hat - f) ** 2) * valid).sum())
    else:
        out.append(zero)
    # (b) far value
    if want_fmse:
        o_far_hat = (a_ln @ v) / sum_ln.clamp_min(eps)[..., None]
        out.append((w[..., None] * (o_far_hat - o_far) ** 2).sum())
    else:
        out.append(zero)
    # (c) far distribution, KL(A_far || A_far_hat)
    A_far_n = A_far / f.clamp_min(eps)[..., None]
    A_hat_far = a_ln / sum_ln.clamp_min(eps)[..., None]
    if want_fkl:
        kl_rows = (torch.xlogy(A_far_n, A_far_n) - torch.xlogy(A_far_n, A_hat_far.clamp_min(eps))).sum(-1)
        out.append((w * kl_rows).sum())
    else:
        out.append(zero)

    # ---------------- stats (no grad) ----------------
    with torch.no_grad():
        vm = valid.float()
        H_t = -(torch.xlogy(A_far_n, A_far_n)).sum(-1)
        H_s = -(torch.xlogy(A_hat_far, A_hat_far.clamp_min(eps)) * mask_far).sum(-1)
        top1 = (A_far_n.argmax(-1) == A_hat_far.argmax(-1)).float()
        idx = A_far_n.argmax(-1, keepdim=True)
        mratio = (A_hat_far.gather(-1, idx) / A_far_n.gather(-1, idx).clamp_min(eps)).squeeze(-1)
        stats = [(f * vm).sum(), (f_hat * vm).sum(), ((f_hat - f).abs() * vm).sum(),
                 (H_t * vm).sum(), (H_s * vm).sum(), (top1 * vm).sum(), (mratio * vm).sum(), vm.sum()]
    return tuple(out) + tuple(s.detach() for s in stats)


def nearfar_losses(q, k, v, f_q, f_k, window_factor, linear_factor, window_size: int,
                   loss_cfg: dict, head_chunk: int = 8, use_checkpoint: bool = True
                   ) -> Dict[str, torch.Tensor]:
    """
    Mean-normalised loss terms (unweighted; weights applied by the trainer).
    Normalisation:
      mse_full : / (b*h*L*d)                 -> equals nn.MSELoss(reduction='mean') of LoLCATs
      xent_full: / (b*h*L)                   -> equals nn.CrossEntropyLoss(reduction='mean')
      gate, far_mse(/d), far_kl : / (#valid far rows)
    """
    b, h, L, d = q.shape
    enabled = (loss_cfg.get('mse_full', 0) > 0, loss_cfg.get('xent_full', 0) > 0,
               loss_cfg.get('gate', 0) > 0, loss_cfg.get('far_mse', 0) > 0,
               loss_cfg.get('far_kl', 0) > 0)
    mode = loss_cfg.get('far_weight', 'f')
    f_min = float(loss_cfg.get('f_min', 0.0 if mode == 'f' else 1e-3))
    mask_near, mask_far = terrace_masks(window_size, L, L, q.device)
    perm = torch.randperm(L, device=q.device) if mode == 'shuffle' else None

    wf = window_factor
    lf = linear_factor if torch.is_tensor(linear_factor) else torch.as_tensor(
        float(linear_factor), device=q.device)
    totals = None
    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        wf_g = wf[:, h0:h1] if wf.dim() == 4 and wf.shape[1] == h else wf
        lf_g = lf[:, h0:h1] if lf.dim() == 4 and lf.shape[1] == h else lf
        args = (q[:, h0:h1], k[:, h0:h1], v[:, h0:h1], f_q[:, h0:h1], f_k[:, h0:h1],
                wf_g, lf_g, mask_near, mask_far, mode, f_min, enabled, perm)
        if use_checkpoint and torch.is_grad_enabled():
            res = checkpoint(_nearfar_group_loss, *args, use_reentrant=False)
        else:
            res = _nearfar_group_loss(*args)
        totals = list(res) if totals is None else [a + r for a, r in zip(totals, res)]

    n_valid = totals[-1].clamp_min(1.0)
    out = {
        'mse_full': totals[0] / (b * h * L * d),
        'xent_full': totals[1] / (b * h * L),
        'gate': totals[2] / n_valid,
        'far_mse': totals[3] / (n_valid * d),
        'far_kl': totals[4] / n_valid,
    }
    for i, key in enumerate(STAT_KEYS):
        out[f'stat_{key}'] = totals[5 + i] / n_valid
    return out


# -----------------------------------------------------------------------------
# Memory-efficient EXACT inference of the same hybrid layer (for long context)
# -----------------------------------------------------------------------------
def hybrid_attention_chunked(q, k, v, f_q, f_k, window_factor, linear_factor,
                             window_size: int, q_chunk: int = 1024, eps: float = 1e-12):
    """
    Numerically identical to `hybrid_attention_quadratic` (no cache) but with
    O(L * q_chunk) memory. FAR set of query t is a prefix [0, B_t) where B_t is a
    block boundary (blocks aligned to the END of the sequence, as in get_masks).
    """
    b, h, L, d = q.shape
    W = window_size
    m = math.ceil(L / W)
    off = m * W - L                            # get_masks crops the first `off` rows/cols
    pos = torch.arange(L, device=q.device)
    blk = (pos + off) // W
    near_start = ((blk - 1).clamp_min(0) * W - off).clamp_min(0)   # first NEAR key of row t
    # rows in block 0 have no FAR tokens -> boundary 0; otherwise boundary = near_start
    far_end = torch.where(blk >= 1, near_start, torch.zeros_like(near_start))
    if m == 1:  # replicate get_masks quirk: linear part covers the whole causal prefix
        far_end = pos + 1

    fq, fk, vf = f_q.float(), f_k.float(), v.float()
    qf, kf = q.float(), k.float()
    out = torch.empty(b, h, L, v.shape[-1], device=q.device, dtype=torch.float32)

    # running prefix state
    S = torch.zeros(b, h, fk.shape[-1], vf.shape[-1], device=q.device)
    Z = torch.zeros(b, h, fk.shape[-1], device=q.device)
    cur = 0
    for c0 in range(0, L, q_chunk):
        c1 = min(L, c0 + q_chunk)
        k0 = int(near_start[c0].item())
        # --- near (softmax) part ---
        logits = torch.einsum('bhmd,bhnd->bhmn', qf[:, :, c0:c1], kf[:, :, k0:c1]) * (d ** -0.5)
        kpos = torch.arange(k0, c1, device=q.device)
        qpos = pos[c0:c1]
        mnear = (kpos[None, :] >= near_start[c0:c1, None]) & (kpos[None, :] <= qpos[:, None])
        logits = logits.masked_fill(~mnear, -1e8)
        a_sm = window_factor * torch.exp(logits - logits.amax(-1, keepdim=True)) * mnear
        num = a_sm @ vf[:, :, k0:c1]
        den = a_sm.sum(-1)
        # --- far (linear) part: group rows by boundary ---
        fe = far_end[c0:c1]
        lin_num = torch.zeros_like(num)
        lin_den = torch.zeros_like(den)
        for B in torch.unique(fe).tolist():
            if B == 0:
                continue
            if B < cur:  # should never happen (boundaries are monotone)
                raise RuntimeError('non-monotone boundary')
            if B > cur:
                S = S + torch.einsum('bhlf,bhld->bhfd', fk[:, :, cur:B], vf[:, :, cur:B])
                Z = Z + fk[:, :, cur:B].sum(-2)
                cur = B
            rows = (fe == B).nonzero().squeeze(-1)
            fqr = fq[:, :, c0:c1][:, :, rows]
            lin_num[:, :, rows] = torch.einsum('bhlf,bhfd->bhld', fqr, S)
            lin_den[:, :, rows] = torch.einsum('bhlf,bhf->bhl', fqr, Z)
        lf_den = linear_factor[..., 0] if torch.is_tensor(linear_factor) and linear_factor.dim() == 4 else linear_factor
        out[:, :, c0:c1] = (num + linear_factor * lin_num) / (den + lf_den * lin_den)[..., None]
    return out.to(q.dtype)


# -----------------------------------------------------------------------------
# Attention layer
# -----------------------------------------------------------------------------
class LolcatsTKWindowNearFarAttention(LolcatsTKWindowAttention):
    """
    Drop-in replacement of LolcatsTKWindowAttention.
    - train_attention=True : computes loss terms inside the layer (layer-local graph,
      head-chunked + checkpointed), returns them via `attn_weights` as a dict.
    - otherwise           : identical outputs; for long sequences uses the exact
      chunked implementation to avoid materialising L x L.
    """
    def __init__(self, *args, nearfar_loss: dict = None, head_chunk: int = 8,
                 inference_chunk: int = 1024, long_threshold: int = 2048, **kwargs):
        super().__init__(*args, **kwargs)
        self.nearfar_loss = dict(nearfar_loss) if nearfar_loss is not None else {
            'mse_full': 1.0}
        self.head_chunk = head_chunk
        self.inference_chunk = inference_chunk
        self.long_threshold = long_threshold

    def _factors(self):
        window_factors = F.sigmoid(self.window_factors)
        linear_factors = 1 - window_factors if self.affine_attention_factors else 1
        return window_factors, linear_factors

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value: Optional[Cache] = None, output_attentions: bool = False,
                use_cache: bool = False, **kwargs):
        b, l, _ = hidden_states.size()
        if not isinstance(past_key_value, LinearAttentionState):
            past_key_value = None      # e.g. HF DynamicCache created by a vanilla model class
        if getattr(self, 'force_softmax', False):
            # Tầng này chạy softmax THẬT. Dùng cho: (a) lượt "thầy" ở giai đoạn 2 chưng cất
            # logit, (b) quét tầng (layer_sweep_passkey). Không dùng KV cache -> chỉ hợp lệ
            # với use_cache=False.
            q, k, v, _ = self.process_qkv(hidden_states, attention_mask, position_ids, None)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            return self.o_proj(y), None, past_key_value

        if not self.train_attention and past_key_value is not None:
            # generation / stateful path: unchanged LoLCATs behaviour
            return super().forward(hidden_states, attention_mask, position_ids,
                                   past_key_value, output_attentions, use_cache, **kwargs)

        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask, position_ids, None)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)
        wf, lf = self._factors()

        if self.train_attention:
            with torch.no_grad():   # teacher output passed to the next layer (as LoLCATs)
                y_true = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)
            losses = nearfar_losses(q, k, v, f_q, f_k, wf, lf, self.window_size,
                                    self.nearfar_loss, head_chunk=self.head_chunk)
            return y_true, {'nearfar': losses}, past_key_value

        if l > self.long_threshold:
            y = hybrid_attention_chunked(q, k, v, f_q, f_k, wf, lf, self.window_size,
                                         q_chunk=self.inference_chunk)
        else:
            y, _ = hybrid_attention_quadratic(q, k, f_q, f_k, v, wf, lf,
                                              window_size=self.window_size)
        y = y.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
        return self.o_proj(y), None, past_key_value


# -----------------------------------------------------------------------------
# Phi (phi-1.5): parallel block, partial rotary, `dense` output projection
# -----------------------------------------------------------------------------
class LolcatsPhiTKWindowNearFarAttention(LolcatsTKWindowNearFarAttention):
    def __init__(self, base_attn: nn.Module, **kwargs):
        if getattr(base_attn, 'qk_layernorm', False):
            raise NotImplementedError('qk_layernorm Phi variants are not supported')
        base_attn.o_proj = base_attn.dense        # alias so LoLCATs init copies it
        super().__init__(base_attn=base_attn, **kwargs)
        self.rotary_ndims = self.rotary_emb.dim

    def process_qkv(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None):
        from transformers.models.phi.modeling_phi import apply_rotary_pos_emb as phi_rope
        b, l, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(b, l, *self.q_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, l, *self.k_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, l, *self.v_shape).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(l, device=q.device)[None]
        kv_seq_len = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        r = self.rotary_ndims
        q_rot, k_rot = phi_rope(q[..., :r], k[..., :r], cos, sin, position_ids)
        q = torch.cat([q_rot, q[..., r:]], dim=-1)
        k = torch.cat([k_rot, k[..., r:]], dim=-1)
        from .utils import repeat_kv
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len
