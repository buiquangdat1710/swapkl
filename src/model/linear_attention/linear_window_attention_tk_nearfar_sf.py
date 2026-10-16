"""
NF-AT-SF: Near-Far Attention Transfer **không teacher forcing** (student forcing).

Động cơ (kết quả chẩn đoán trên Llama 3.2 1B):
  * từng tầng riêng lẻ khớp rất tốt (far_top1 ~0.88-0.97, entropy ~ softmax, gate_err ~0.11)
  * nhưng tuyến tính hoá >= 4 tầng là passkey sụp -> SAI SỐ TÍCH LUỸ QUA TẦNG
  * mọi attention transfer hiện hành (LoLCATs, Hedgehog, MOHAWK) đều teacher-force từng tầng,
    nên hàm mục tiêu của chúng KHÔNG NHÌN THẤY sai số tích luỹ này.

Cách làm: hai lượt truyền xuôi trên cùng một batch.
  Lượt 1 (thầy, no_grad): mọi tầng chạy softmax thật; mỗi tầng lưu (q, k, v) của nó.
  Lượt 2 (trò):           mọi tầng chạy bản lai (cửa sổ + tuyến tính) và **đẩy đầu ra thật
                          sang tầng sau** (không teacher-force). Mỗi tầng tính loss giữa
                          (q, k, v) của chính nó và (q, k, v) đã lưu của thầy.

Do đó q, k, v của trò ở tầng m phản ánh mọi sai số mà các tầng 0..m-1 đã gây ra, và hàm mục
tiêu buộc tầng m phải bù lại. Chi phí: gấp đôi lượt truyền xuôi của thầy, không đổi kiến trúc,
không thêm tham số.

`detach_between_layers=True` (mặc định) cắt gradient ở đầu vào mỗi tầng: đầu vào vẫn là
hidden state của TRÒ (nên vẫn thấy sai số tích luỹ), nhưng đồ thị vẫn cục bộ theo tầng nên
bộ nhớ ngang bản teacher forcing. Đặt False để lan truyền gradient xuyên tầng (tốn hơn nhiều).
"""
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.cache_utils import Cache

from .linear_attention import LinearAttentionState
from .linear_window_attention_tk import hybrid_attention_quadratic
from .linear_window_attention_tk_nearfar import (
    LolcatsTKWindowNearFarAttention, terrace_masks, hybrid_attention_chunked, LOSS_KEYS,
)

STAT_KEYS_SF = ('f_mean', 'fhat_mean', 'gate_abs_err', 'far_ent_teacher', 'far_ent_student',
                'far_top1', 'top1_mass_ratio', 'qkv_drift')


def _sf_group_loss(q, k, v, f_q, f_k, qt, kt, vt, window_factor, linear_factor,
                   mask_near, mask_far, far_weight_mode: str, f_min: float,
                   enabled, eps: float = 1e-12):
    """
    Như `_nearfar_group_loss` nhưng mục tiêu của THẦY tính từ (qt, kt, vt) riêng, còn phía TRÒ
    tính từ (q, k, v) do chính mô hình đã tuyến tính hoá sinh ra. Trả về các TỔNG.
    """
    want_mse, want_xent, want_gate, want_fmse, want_fkl = enabled
    q, k, v = q.float(), k.float(), v.float()
    qt, kt, vt = qt.float(), kt.float(), vt.float()
    f_q, f_k = f_q.float(), f_k.float()
    d = k.shape[-1]
    L = q.shape[-2]

    # ---------------- teacher (no grad, từ hidden state của thầy) ----------------
    with torch.no_grad():
        causal = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        logits_t = torch.einsum('bhmd,bhnd->bhmn', qt, kt) * (d ** -0.5)
        A = torch.softmax(logits_t.masked_fill(~causal, float('-inf')), dim=-1)
        A_far = A * mask_far
        f = A_far.sum(-1)
        y_true = A @ vt
        o_far = (A_far @ vt) / f.clamp_min(eps)[..., None]
        valid = mask_far.any(-1) & (f > f_min)
        w = f if far_weight_mode == 'f' else torch.ones_like(f)
        w = w * valid

    # ---------------- student (từ hidden state của chính nó) ----------------
    logits = torch.einsum('bhmd,bhnd->bhmn', q, k) * (d ** -0.5)
    a_sm = logits.masked_fill(~mask_near, -1e8)
    a_sm = window_factor * torch.exp(a_sm - a_sm.amax(-1, keepdim=True)) * mask_near
    sum_sm = a_sm.sum(-1)
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q, f_k)
    a_ln = linear_factor * a_ln * mask_far
    sum_ln = a_ln.sum(-1)
    den = sum_sm + sum_ln

    zero = logits.new_zeros(())
    out = []
    # MSE trên đầu ra tầng: trò (student forcing) so với thầy  -> "LoLCATs + student forcing"
    if want_mse:
        y_pred = ((a_sm + a_ln) @ v) / den[..., None]
        out.append(((y_pred - y_true) ** 2).sum())
    else:
        out.append(zero)
    if want_xent:
        a_pred = (a_sm + a_ln) / den[..., None]
        out.append(-(torch.xlogy(A, a_pred.clamp_min(eps))).sum())
    else:
        out.append(zero)

    f_hat = sum_ln / den
    out.append((((f_hat - f) ** 2) * valid).sum() if want_gate else zero)          # (a)
    if want_fmse:
        o_far_hat = (a_ln @ v) / sum_ln.clamp_min(eps)[..., None]
        out.append((w[..., None] * (o_far_hat - o_far) ** 2).sum())                # (b)
    else:
        out.append(zero)
    A_far_n = A_far / f.clamp_min(eps)[..., None]
    A_hat_far = a_ln / sum_ln.clamp_min(eps)[..., None]
    if want_fkl:
        kl = (torch.xlogy(A_far_n, A_far_n) - torch.xlogy(A_far_n, A_hat_far.clamp_min(eps))).sum(-1)
        out.append((w * kl).sum())                                                 # (c)
    else:
        out.append(zero)

    # ---------------- stats ----------------
    with torch.no_grad():
        vm = valid.float()
        H_t = -(torch.xlogy(A_far_n, A_far_n)).sum(-1)
        H_s = -(torch.xlogy(A_hat_far, A_hat_far.clamp_min(eps)) * mask_far).sum(-1)
        idx = A_far_n.argmax(-1, keepdim=True)
        top1 = (idx.squeeze(-1) == A_hat_far.argmax(-1)).float()
        mratio = (A_hat_far.gather(-1, idx) / A_far_n.gather(-1, idx).clamp_min(eps)).squeeze(-1)
        # trôi dạt của q,k,v so với thầy: đo trực tiếp sai số tích luỹ đến tầng này
        drift = ((q - qt).pow(2).sum() + (k - kt).pow(2).sum()) / (
            qt.pow(2).sum() + kt.pow(2).sum()).clamp_min(eps)
        drift = drift * vm.sum()          # nhân sẵn để phép chia n_valid ở ngoài trả lại giá trị
        stats = [(f * vm).sum(), (f_hat * vm).sum(), ((f_hat - f).abs() * vm).sum(),
                 (H_t * vm).sum(), (H_s * vm).sum(), (top1 * vm).sum(), (mratio * vm).sum(),
                 drift, vm.sum()]
    return tuple(out) + tuple(s.detach() for s in stats)


def nearfar_losses_sf(q, k, v, f_q, f_k, qt, kt, vt, window_factor, linear_factor,
                      window_size: int, loss_cfg: dict, head_chunk: int = 8,
                      use_checkpoint: bool = True) -> Dict[str, torch.Tensor]:
    b, h, L, d = q.shape
    enabled = (loss_cfg.get('mse_full', 0) > 0, loss_cfg.get('xent_full', 0) > 0,
               loss_cfg.get('gate', 0) > 0, loss_cfg.get('far_mse', 0) > 0,
               loss_cfg.get('far_kl', 0) > 0)
    mode = loss_cfg.get('far_weight', 'f')
    f_min = float(loss_cfg.get('f_min', 0.0 if mode == 'f' else 1e-3))
    mask_near, mask_far = terrace_masks(window_size, L, L, q.device)
    lf = linear_factor if torch.is_tensor(linear_factor) else torch.as_tensor(
        float(linear_factor), device=q.device)
    totals = None
    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        wf_g = window_factor[:, h0:h1] if window_factor.dim() == 4 else window_factor
        lf_g = lf[:, h0:h1] if lf.dim() == 4 else lf
        args = (q[:, h0:h1], k[:, h0:h1], v[:, h0:h1], f_q[:, h0:h1], f_k[:, h0:h1],
                qt[:, h0:h1], kt[:, h0:h1], vt[:, h0:h1], wf_g, lf_g,
                mask_near, mask_far, mode, f_min, enabled)
        res = (checkpoint(_sf_group_loss, *args, use_reentrant=False)
               if use_checkpoint and torch.is_grad_enabled() else _sf_group_loss(*args))
        totals = list(res) if totals is None else [a + r for a, r in zip(totals, res)]
    n_valid = totals[-1].clamp_min(1.0)
    out = {'mse_full': totals[0] / (b * h * L * d), 'xent_full': totals[1] / (b * h * L),
           'gate': totals[2] / n_valid, 'far_mse': totals[3] / (n_valid * d),
           'far_kl': totals[4] / n_valid}
    for i, key in enumerate(STAT_KEYS_SF):
        out[f'stat_{key}'] = totals[5 + i] / n_valid
    return out


class LolcatsTKWindowNearFarSFAttention(LolcatsTKWindowNearFarAttention):
    """
    Ba chế độ, do trainer đặt qua `nf_mode`:
      'capture' : chạy softmax thật, lưu (q,k,v) của thầy  -> lượt 1
      'student' : chạy bản lai, đẩy đầu ra THẬT sang tầng sau, tính loss -> lượt 2
      None      : như lớp cha (suy luận bình thường)
    """
    def __init__(self, *args, detach_between_layers: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.nf_mode = None
        self.detach_between_layers = detach_between_layers
        self._t_qkv = None

    def clear_teacher(self):
        self._t_qkv = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value: Optional[Cache] = None, output_attentions: bool = False,
                use_cache: bool = False, **kwargs):
        if self.nf_mode is None:
            return super().forward(hidden_states, attention_mask, position_ids,
                                   past_key_value, output_attentions, use_cache, **kwargs)
        if not isinstance(past_key_value, LinearAttentionState):
            past_key_value = None
        b, l, _ = hidden_states.size()

        if self.nf_mode == 'capture':
            with torch.no_grad():
                q, k, v, _ = self.process_qkv(hidden_states, attention_mask, position_ids, None)
                self._t_qkv = (q.detach(), k.detach(), v.detach())
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y = y.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                return self.o_proj(y), None, past_key_value

        assert self._t_qkv is not None, 'thiếu lượt truyền xuôi của thầy (nf_mode="capture")'
        if self.detach_between_layers:
            hidden_states = hidden_states.detach()
        q, k, v, _ = self.process_qkv(hidden_states, attention_mask, position_ids, None)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)
        wf, lf = self._factors()
        qt, kt, vt = self._t_qkv
        losses = nearfar_losses_sf(q, k, v, f_q, f_k, qt, kt, vt, wf, lf, self.window_size,
                                   self.nearfar_loss, head_chunk=self.head_chunk)
        # đầu ra THẬT của trò -> tầng sau (không teacher forcing).
        # Khi cắt gradient giữa các tầng, tầng sau sẽ detach đầu vào, nên đồ thị của y là
        # vô dụng: tính dưới no_grad để không giữ các tensor L x L của MỌI tầng trong bộ nhớ.
        ctx = torch.no_grad() if self.detach_between_layers else torch.enable_grad()
        with ctx:
            if l > self.long_threshold:
                y = hybrid_attention_chunked(q, k, v, f_q, f_k, wf, lf, self.window_size,
                                             q_chunk=self.inference_chunk)
            else:
                y, _ = hybrid_attention_quadratic(q, k, f_q, f_k, v, wf, lf,
                                                  window_size=self.window_size)
            y = y.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            out = self.o_proj(y)
        return out, {'nearfar': losses}, past_key_value
