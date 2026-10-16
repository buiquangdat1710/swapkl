"""
Thước đo ĐỘ QUAN TRỌNG của từng tầng attention cho truy hồi tầm xa (passkey / NIAH).

Vấn đề: chọn k tầng giữ attention đầy đủ trong số L tầng là bài toán tổ hợp C(L,k).
`select_layers.py --mode greedy` đã đúng hướng nhưng vẫn là tìm kiếm: k*L phép đo, mỗi
phép đo là một lượt truyền xuôi trên cả batch. Ở 8B/32 tầng, k=8 là 256 lượt.

Ở đây định nghĩa thước đo để KHÔNG phải tìm kiếm. Đặt cổng z_l cho mỗi tầng:

    o_l = o_l^SWA + z_l * (o_l^full - o_l^SWA),     z_l in [0,1]

z=1 trên mọi tầng <=> mô hình gốc; z=0 <=> SWA+sink thuần. Gọi L(z) là NLL của token đáp
án (teacher forcing) trên các prompt NIAH có kim nằm NGOÀI cửa sổ. Năm thước đo:

  grad1   I_l = -dL/dz_l tại z=1.  Xấp xỉ bậc nhất của ĐỘ TĂNG NLL khi hạ tầng l xuống
          SWA trong khi mọi tầng khác giữ đầy đủ. Đây đúng là đại lượng mà bộ chọn
          NLL-guided (arXiv:2606.27791) đo bằng L+1 lượt xuôi; ở đây lấy được TOÀN BỘ L
          tầng bằng MỘT lượt xuôi + MỘT lượt ngược.

  grad0   I_l = -dL/dz_l tại z=0.  Lợi ích biên khi nâng tầng l lên đầy đủ trong khi mọi
          tầng khác vẫn là SWA. Đây mới là chế độ vận hành thật (đa số tầng là SWA).

  ig      Tích phân gradient dọc đường thẳng z = t*1, t: 0 -> 1.
              I_l = -\int_0^1 dL/dz_l |_{z=t} dt
          TÍNH CHẤT ĐẦY ĐỦ (completeness):  sum_l I_l = L(0) - L(1) = đúng bằng khoảng
          cách NLL giữa SWA thuần và attention đầy đủ. Nghĩa là mỗi tầng nhận đúng PHẦN
          CHIA của mình trong tổng thiệt hại, thay vì một điểm số không có đơn vị. grad1
          và grad0 chỉ là hai đầu mút của tích phân này, nên chúng bỏ qua tương tác giữa
          các tầng; ig thì không.

  delta   |o^full - o^SWA| / |o^full| tại các vị trí truy vấn cuối (vùng câu hỏi/đáp án).
          Thuần xuôi, KHÔNG lan truyền: chỉ đo nhiễu loạn CỤC BỘ tại tầng đó, họ hàng với
          "lazy ratio" của LightTransfer. Rẻ nhất, và là phép thử xem nhiễu loạn cục bộ có
          dự báo được thiệt hại đầu ra hay không.

  oow     Khối lượng attention rơi RA NGOÀI (cửa sổ hợp sink), lấy trung bình trên các đầu,
          tại các truy vấn cuối. Chính là phần mà SWA sẽ xoá mất.

  needle  Khối lượng attention rơi ĐÚNG vào token của kim. Bản theo-tầng của "retrieval
          head" (Wu và cộng sự, arXiv:2404.15574). Lấy cả trung bình lẫn CỰC ĐẠI trên các
          đầu: đầu truy hồi vốn thưa, nên trung bình trên 32 đầu sẽ pha loãng mất tín hiệu.

Ba thước đo sau thuần xuôi (1 lượt), hai thước đo đầu 1 lượt xuôi + 1 ngược, ig là m lượt.
Không cái nào phụ thuộc k, nên xếp hạng một lần rồi lấy top-k cho MỌI k.

Chạy:
    python nfat/layer_importance.py --model_config nf_llama3_2_1b_softmax \
        --window 512 --n_sink 4 --variant s2 --L 4096 --n 8 --tag 1b
    python nfat/layer_importance.py --model_config nf_llama3_2_1b_softmax --selftest
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfat.model_io import load_model                                   # noqa: E402
from nfat.csv_append import append_row                                 # noqa: E402
from nfat.swa_sink import decoder_layers, _causal_window_sink_keep     # noqa: E402
from nfat.niah_utils import build, load_haystack_words                 # noqa: E402

SCORERS = ('grad1', 'grad0', 'ig', 'ig_greedy', 'delta', 'oow', 'needle',
           'needle_max', 'nll_full', 'nll_swa')


# ----------------------------------------------------------------------------- cổng z
class LayerGates:
    """
    Bọc mọi tầng attention để đầu ra là pha trộn có cổng giữa nhánh đầy đủ và nhánh SWA.

    Hai nhánh dùng CHUNG q,k,v và chung trọng số; khác nhau duy nhất ở mask. Nên z=1 tái
    lập ĐÚNG mô hình gốc và z=0 tái lập ĐÚNG apply_swa_sink (có kiểm chứng: --selftest).

    Pha trộn tính ở float32 rồi ép lại dtype gốc: gradient theo z không bị nhiễu bf16.
    """

    def __init__(self, model, window: int, n_sink: int):
        self.model = model
        self.window = window
        self.n_sink = n_sink
        self.base = getattr(model, 'model', model)
        self.layers = decoder_layers(model)
        self.n = len(self.layers)
        dev = next(model.parameters()).device
        self.z = torch.ones(self.n, dtype=torch.float32, device=dev, requires_grad=True)
        self.collect_delta = False
        self.delta = None
        self.mass_ctx = None          # dict(needle=(a,b)) -> bật đo khối lượng attention
        self.mass = None
        self._installed = False

    # --- mask ---------------------------------------------------------------
    def _install_mask(self):
        base = self.base
        orig = getattr(base, '_nfat_orig_update_causal_mask', base._update_causal_mask)
        base._nfat_orig_update_causal_mask = orig
        window, n_sink = self.window, self.n_sink

        def patched(attention_mask, input_tensor, cache_position, past_key_values,
                    output_attentions=False, *a, **kw):
            full = orig(attention_mask, input_tensor, cache_position, past_key_values,
                        output_attentions, *a, **kw)
            device, dtype = input_tensor.device, input_tensor.dtype
            q_len = cache_position.shape[0]
            kv_len = full.shape[-1] if full is not None else int(cache_position[-1]) + 1
            keep = _causal_window_sink_keep(cache_position, kv_len, window, n_sink, device)
            min_val = torch.finfo(dtype).min
            if full is None:
                # đường sdpa trả None khi mask thuần nhân quả -> dựng tay cho nhánh SWA
                swa = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
                swa.masked_fill_(~keep, min_val)
                swa = swa.view(1, 1, q_len, kv_len).expand(
                    input_tensor.shape[0], 1, q_len, kv_len)
            else:
                swa = full.masked_fill(~keep.view(1, 1, q_len, kv_len), min_val)
            base._nfat_full_mask = full
            base._nfat_swa_mask = swa
            base._nfat_keep = keep
            base._nfat_cache_position = cache_position
            return full

        base._update_causal_mask = patched

    # --- attention hai nhánh -------------------------------------------------
    def _install_attn(self):
        gates = self
        for idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            attn._nfat_orig_forward = attn.forward

            def make(a_, i_):
                def fwd(*args, **kwargs):
                    if args:
                        raise RuntimeError('cần transformers gọi self_attn bằng keyword')
                    base = gates.base
                    kw_full = dict(kwargs); kw_full['attention_mask'] = base._nfat_full_mask
                    kw_swa = dict(kwargs); kw_swa['attention_mask'] = base._nfat_swa_mask
                    z = gates.z[i_]
                    if gates.mass_ctx is not None:
                        gates._measure_mass(a_, i_, kwargs)
                    if float(z) == 1.0 and not z.requires_grad:
                        return a_._nfat_orig_forward(**kw_full)
                    of = a_._nfat_orig_forward(**kw_full)
                    os_ = a_._nfat_orig_forward(**kw_swa)
                    f32f, f32s = of[0].float(), os_[0].float()
                    if gates.collect_delta:
                        p = gates.mass_probe
                        num = (f32f[:, -p:] - f32s[:, -p:]).norm(dim=-1)
                        den = f32f[:, -p:].norm(dim=-1).clamp_min(1e-6)
                        gates.delta[i_] += float((num / den).mean())
                    blended = (f32s + z * (f32f - f32s)).to(of[0].dtype)
                    return (blended,) + tuple(of[1:])
                return fwd
            attn.forward = make(attn, idx)

    # --- khối lượng attention (thuần xuôi, chỉ vài truy vấn cuối) ------------
    def _measure_mass(self, attn, idx, kwargs):
        """
        Tính lại q,k cho P truy vấn CUỐI rồi lấy softmax nhân quả đầy đủ.

        Chỉ P hàng nên tốn O(P*L) bộ nhớ thay vì O(L^2): ở 8B, ma trận attention đầy đủ
        fp32 là 8,6 GB mỗi tầng, còn cách này là vài MB.
        """
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        h = kwargs['hidden_states']
        pos = kwargs.get('position_ids')
        B, Q, _ = h.shape
        P = min(self.mass_probe, Q)
        nH, nKV = attn.config.num_attention_heads, attn.config.num_key_value_heads
        hd = attn.head_dim
        with torch.no_grad():
            q = attn.q_proj(h).view(B, Q, nH, hd).transpose(1, 2)
            k = attn.k_proj(h).view(B, Q, nKV, hd).transpose(1, 2)
            cos, sin = attn.rotary_emb(k, pos)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, nH // nKV)
            qs = q[:, :, -P:].float()                                   # (B,H,P,hd)
            sc = torch.matmul(qs, k.float().transpose(2, 3)) / math.sqrt(hd)
            qpos = torch.arange(Q - P, Q, device=h.device).view(-1, 1)
            kpos = torch.arange(Q, device=h.device).view(1, -1)
            causal = kpos <= qpos
            sc = sc.masked_fill(~causal.view(1, 1, P, Q), torch.finfo(torch.float32).min)
            p = torch.softmax(sc, dim=-1)                                # (B,H,P,Q)
            inside = causal & (((qpos - kpos) < self.window) | (kpos < self.n_sink))
            oow = p.masked_fill(inside.view(1, 1, P, Q), 0).sum(-1)      # (B,H,P)
            self.mass['oow'][idx] += float(oow.mean())
            span = self.mass_ctx.get('needle')
            if span is not None:
                a, b = span
                nm = p[..., a:b].sum(-1)                                 # (B,H,P)
                per_head = nm.mean(dim=(0, 2))                           # (H,)
                self.mass['needle'][idx] += float(per_head.mean())
                self.mass['needle_max'][idx] += float(per_head.max())

    # --- vòng đời ------------------------------------------------------------
    def install(self, mass_probe: int = 24):
        if self._installed:
            return self
        self.mass_probe = mass_probe
        self._install_mask()
        self._install_attn()
        self._installed = True
        return self

    def remove(self):
        base = self.base
        if hasattr(base, '_nfat_orig_update_causal_mask'):
            base._update_causal_mask = base._nfat_orig_update_causal_mask
            del base._nfat_orig_update_causal_mask
        for l in self.layers:
            if hasattr(l.self_attn, '_nfat_orig_forward'):
                l.self_attn.forward = l.self_attn._nfat_orig_forward
                del l.self_attn._nfat_orig_forward
        self._installed = False
        return self

    def set_z(self, value):
        with torch.no_grad():
            if np.isscalar(value):
                self.z.fill_(float(value))
            else:
                self.z.copy_(torch.as_tensor(value, dtype=torch.float32, device=self.z.device))
        return self


# ----------------------------------------------------------------------------- dữ liệu
def needle_span(ids, tok, gold):
    """Vị trí (bắt đầu, kết thúc) của token giá trị kim trong prompt. None nếu không thấy."""
    for prefix in (' ', ''):
        g = tok.encode(prefix + gold, add_special_tokens=False)
        if not g:
            continue
        n = len(g)
        for s in range(len(ids) - n, -1, -1):
            if ids[s:s + n] == g:
                return s, s + n
    return None


def make_items(tok, cfg, seed, n_per_depth):
    """Danh sách prompt NIAH + token đáp án + vị trí kim. Kim luôn NGOÀI cửa sổ."""
    rng = random.Random(seed)
    items = []
    for d in cfg['depths']:
        for _ in range(n_per_depth):
            ids, golds, _ = build(tok, cfg['variant'], cfg['L'], d, rng, cfg['words'])
            gold_txt = ' ' + ' '.join(str(g) for g in golds)
            gid = tok.encode(gold_txt, add_special_tokens=False)
            span = needle_span(ids, tok, str(golds[0]))
            if span is not None and span[0] >= len(ids) - cfg['window']:
                continue                      # kim nằm sẵn trong cửa sổ -> vô nghĩa
            items.append(dict(ids=ids, gold=gid, span=span, depth=d,
                              golds=[str(g) for g in golds]))
    return items


def gold_token_logprobs(model, it, device):
    """
    Log-xác suất từng token đáp án. CHỈ chạy lm_head trên các vị trí đáp án.

    Gọi model(...) thẳng sẽ chiếu toàn bộ 4096 vị trí qua lm_head: ở Llama 3.1 8B
    (từ vựng 128256) riêng `logits.float()` đã là 2,1 GB, rồi log_softmax thêm 2,1 GB
    nữa, và tất cả bị giữ trong đồ thị để lan truyền ngược -> tràn bộ nhớ ở 46 GB.
    Đáp án chỉ dài vài token, nên cắt TRƯỚC lm_head là đủ.
    """
    full = torch.tensor([it['ids'] + it['gold']], device=device)
    base = getattr(model, 'model', None)
    head = getattr(model, 'lm_head', None)
    a, b = len(it['ids']) - 1, len(it['ids']) + len(it['gold']) - 1
    if base is not None and head is not None:
        h = base(full, use_cache=False)[0][0, a:b]
        sl = head(h).float()
    else:                                    # mô hình không tách được -> đường cũ
        sl = model(full, use_cache=False).logits[0, a:b].float()
    lp = torch.log_softmax(sl, dim=-1)
    tgt = torch.tensor(it['gold'], device=device).unsqueeze(-1)
    return lp.gather(-1, tgt).squeeze(-1)


def answer_nll(model, items, device):
    """NLL trung bình mỗi token đáp án, trung bình trên các mẫu. Giữ đồ thị để lấy gradient."""
    total = 0.0
    for it in items:
        total = total + (-gold_token_logprobs(model, it, device).mean())
    return total / len(items)


# ----------------------------------------------------------------------------- thước đo
def grad_at(model, gates, items, t, device):
    """-dL/dz tại z = t (vô hướng -> t*1, hoặc vector cho từng tầng).

    Cộng dồn gradient qua từng mẫu một để không phải giữ nhiều đồ thị cùng lúc.
    """
    gates.set_z(t)
    if gates.z.grad is not None:
        gates.z.grad = None
    for it in items:
        loss = answer_nll(model, [it], device) / len(items)
        loss.backward()
    return -gates.z.grad.detach().float().cpu().numpy().copy()


def integrated_gradients(model, gates, items, steps, device, verbose=True):
    """
    Trung bình gradient trên lưới trung điểm của t in [0,1]. Nhân với (1-0)=1.

    Kiểm chứng completeness ở ngoài: sum_l I_l phải xấp xỉ L(z=0) - L(z=1).
    """
    acc = np.zeros(gates.n)
    ts = [(i + 0.5) / steps for i in range(steps)]
    for j, t in enumerate(ts):
        g = grad_at(model, gates, items, t, device)
        acc += g / steps
        if verbose:
            print(f'   ig bước {j + 1}/{steps} (t={t:.3f}) ok', flush=True)
    return acc




def ig_greedy(model, gates, items, device, k, steps, verbose=True):
    """
    CHỌN TIẾN dẫn hướng bằng gradient — sửa điểm yếu của xếp-hạng-rồi-lấy-top-k.

    Top-k của một điểm số ĐƠN LẺ giả định các tầng đóng góp độc lập. Đo thực tế ở 1B cho
    thấy điều đó sai ở k nhỏ: ig xếp [5,10,6,1] nhưng tập tốt là [5,6,10,14] — tầng 1 và
    tầng 14 dư thừa với nhau nên điểm đơn lẻ không phân biệt được.

    Ở đây mỗi bước tính lại tích phân gradient TẠI CẤU HÌNH HIỆN TẠI: các tầng đã chọn giữ
    z=1, các tầng còn lại chạy dọc đường t: 0 -> 1. Tầng nào có công lớn nhất khi đã biết
    các tầng kia làm gì thì được chọn. Chi phí k*m lượt, KHÔNG phải k*L phép đo như quét
    tham lam của select_layers.py (ở 8B, k=8: 32 lượt thay vì 256 phép đo).

    Trả về THỨ TỰ chọn; ghi ra dưới dạng điểm = -(vị trí) để top-k của nó đúng bằng tập
    tham lam ở bước k.
    """
    S, order = [], []
    for step in range(k):
        acc = np.zeros(gates.n)
        for i in range(steps):
            t = (i + 0.5) / steps
            z = np.full(gates.n, t)
            z[S] = 1.0
            acc += grad_at(model, gates, items, z, device) / steps
        acc[S] = -np.inf
        pick = int(np.argmax(acc))
        S.append(pick); order.append(pick)
        if verbose:
            print(f'   ig_greedy bước {step + 1}/{k}: thêm tầng {pick} '
                  f'(công {acc[pick]:+.4f}) -> {sorted(S)}', flush=True)
    return order


@torch.no_grad()
def finite_diff(model, gates, items, device, mode, verbose=True):
    """
    Bản HỮU HẠN SAI PHÂN của grad1/grad0 — đúng phép đo mà NLL-guided
    (arXiv:2606.27791) dùng, tốn L+1 lượt xuôi thay vì 1 lượt xuôi + 1 lượt ngược.

      mode='from_full': z=1 khắp nơi, hạ RIÊNG tầng l xuống 0.
                        I_l = L(tầng l thành SWA) - L(đầy đủ)   [đối chiếu với grad1]
      mode='from_swa' : z=0 khắp nơi, nâng RIÊNG tầng l lên 1.
                        I_l = L(SWA thuần) - L(riêng tầng l đầy đủ)  [đối chiếu với grad0]

    Chạy cái này để trả lời: gradient một lượt có tái lập được xếp hạng của phép đo
    L+1 lượt không? Nếu có thì bỏ hẳn được vòng quét.
    """
    base_z = 1.0 if mode == 'from_full' else 0.0
    other = 0.0 if mode == 'from_full' else 1.0
    gates.set_z(base_z)
    ref = float(answer_nll(model, items, device))
    out = np.zeros(gates.n)
    for l in range(gates.n):
        z = np.full(gates.n, base_z); z[l] = other
        gates.set_z(z)
        out[l] = float(answer_nll(model, items, device)) - ref
        if mode == 'from_swa':
            out[l] = -out[l]        # nâng tầng l lên phải LÀM GIẢM NLL -> đảo dấu
        if verbose:
            print(f'   {mode} tầng {l:2d}: {out[l]:+.4f}', flush=True)
    gates.set_z(base_z)
    return out


@torch.no_grad()
def forward_scores(model, gates, items, device, want_mass=True):
    """delta (nhiễu loạn cục bộ) và oow/needle (khối lượng attention) trong một lượt xuôi."""
    gates.delta = np.zeros(gates.n)
    gates.mass = dict(oow=np.zeros(gates.n), needle=np.zeros(gates.n),
                      needle_max=np.zeros(gates.n))
    gates.collect_delta = True
    gates.set_z(0.5)          # z bất kỳ khác 0/1 để cả hai nhánh cùng được tính
    n_mass = 0
    for it in items:
        gates.mass_ctx = dict(needle=it['span']) if want_mass else None
        if want_mass:
            n_mass += 1
        model(torch.tensor([it['ids']], device=device), use_cache=False)
    gates.collect_delta = False
    gates.mass_ctx = None
    out = dict(delta=gates.delta / len(items))
    if want_mass and n_mass:
        for k, v in gates.mass.items():
            out[k] = v / n_mass
    return out


# ----------------------------------------------------------------------------- selftest
def selftest(model, tok, gates, device, L=1024):
    """z=1 phải trùng mô hình gốc; z=0 phải trùng apply_swa_sink. Sai số bf16 là chấp nhận."""
    from nfat.swa_sink import apply_swa_sink, remove_swa_sink
    rng = random.Random(0)
    ids, golds, _ = build(tok, 's1', L, 0.25, rng, None)
    x = torch.tensor([ids], device=device)
    gates.remove()
    with torch.no_grad():
        ref_full = model(x, use_cache=False).logits.float()
    apply_swa_sink(model, window=gates.window, n_sink=gates.n_sink, verbose=False)
    with torch.no_grad():
        ref_swa = model(x, use_cache=False).logits.float()
    remove_swa_sink(model)
    gates.install()
    with torch.no_grad():
        gates.set_z(1.0)
        got_full = model(x, use_cache=False).logits.float()
        gates.set_z(0.0)
        got_swa = model(x, use_cache=False).logits.float()
    d1 = (got_full - ref_full).abs().max().item()
    d0 = (got_swa - ref_swa).abs().max().item()
    s1 = (ref_swa - ref_full).abs().max().item()
    print(f'  z=1 vs mô hình gốc      : lệch tối đa {d1:.3e}')
    print(f'  z=0 vs apply_swa_sink   : lệch tối đa {d0:.3e}')
    print(f'  (đối chứng: SWA vs gốc  : lệch tối đa {s1:.3e})')
    ok = d1 < 1e-2 and d0 < 1e-2 and s1 > 1e-1
    print('  ->', 'ĐẠT' if ok else 'HỎNG')
    return ok


# ----------------------------------------------------------------------------- chạy
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', required=True)
    p.add_argument('--window', type=int, default=512)
    p.add_argument('--n_sink', type=int, default=4)
    p.add_argument('--variant', default='s2')
    p.add_argument('--L', type=int, default=4096)
    p.add_argument('--n', type=int, default=8, help='mẫu mỗi độ sâu')
    p.add_argument('--depths', default='0,0.25,0.5,0.75')
    p.add_argument('--scorers', default='grad1,grad0,ig,delta,oow,needle')
    p.add_argument('--ig_steps', type=int, default=8)
    p.add_argument('--greedy_k', type=int, default=0,
                   help='>0: chạy thêm ig_greedy tới k tầng')
    p.add_argument('--greedy_steps', type=int, default=4, help='số bước ig mỗi vòng greedy')
    p.add_argument('--mass_probe', type=int, default=24, help='số truy vấn cuối để đo mass')
    p.add_argument('--haystack', default=os.environ.get('NFAT_NIAH_TEXT', ''))
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/layer_importance.csv')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--huggingface_token', default=None)
    a = p.parse_args()

    model, tok, _ = load_model(a.model_config, huggingface_token=a.huggingface_token)
    model.config.use_cache = False
    device = next(model.parameters()).device
    gates = LayerGates(model, a.window, a.n_sink).install(mass_probe=a.mass_probe)
    print(f'-> {gates.n} tầng | cửa sổ {a.window}+{a.n_sink} sink | {a.model_config}', flush=True)

    if a.selftest:
        ok = selftest(model, tok, gates, device)
        sys.exit(0 if ok else 1)

    cfg = dict(variant=a.variant, L=a.L, window=a.window,
               depths=[float(x) for x in a.depths.split(',')],
               words=load_haystack_words(a.haystack))
    items = make_items(tok, cfg, a.seed, a.n)
    n_span = sum(1 for it in items if it['span'] is not None)
    print(f'-> {len(items)} prompt hiệu lực ({n_span} định vị được kim) '
          f'| biến thể {a.variant} @ L={a.L}\n', flush=True)

    want = [s.strip() for s in a.scorers.split(',') if s.strip()]
    scores, t0 = {}, time.time()

    with torch.no_grad():
        gates.set_z(1.0); nll1 = float(answer_nll(model, items, device))
        gates.set_z(0.0); nll0 = float(answer_nll(model, items, device))
    print(f'NLL đáp án: attention đầy đủ {nll1:.4f} | SWA thuần {nll0:.4f} '
          f'| khoảng cách {nll0 - nll1:.4f}\n', flush=True)

    if 'grad1' in want:
        scores['grad1'] = grad_at(model, gates, items, 1.0, device)
        print(f'grad1 xong ({time.time() - t0:.0f}s)', flush=True)
    if 'grad0' in want:
        scores['grad0'] = grad_at(model, gates, items, 0.0, device)
        print(f'grad0 xong ({time.time() - t0:.0f}s)', flush=True)
    if 'ig' in want:
        scores['ig'] = integrated_gradients(model, gates, items, a.ig_steps, device)
        s = scores['ig'].sum()
        print(f'ig xong ({time.time() - t0:.0f}s) | completeness: tổng {s:.4f} '
              f'so với khoảng cách NLL {nll0 - nll1:.4f} '
              f'(sai lệch {abs(s - (nll0 - nll1)) / max(1e-9, abs(nll0 - nll1)) * 100:.1f}%)',
              flush=True)
    if a.greedy_k > 0:
        order = ig_greedy(model, gates, items, device, a.greedy_k, a.greedy_steps)
        vec = np.full(gates.n, -1e9)
        for pos, l in enumerate(order):
            vec[l] = -pos                      # top-k của điểm này = tập tham lam bước k
        scores['ig_greedy'] = vec
        print(f'ig_greedy xong ({time.time() - t0:.0f}s): thứ tự {order}', flush=True)
    if 'nll_full' in want:
        scores['nll_full'] = finite_diff(model, gates, items, device, 'from_full')
        print(f'nll_full xong ({time.time() - t0:.0f}s)', flush=True)
    if 'nll_swa' in want:
        scores['nll_swa'] = finite_diff(model, gates, items, device, 'from_swa')
        print(f'nll_swa xong ({time.time() - t0:.0f}s)', flush=True)
    if {'delta', 'oow', 'needle', 'needle_max'} & set(want):
        fs = forward_scores(model, gates, items, device,
                            want_mass=bool({'oow', 'needle', 'needle_max'} & set(want)))
        scores.update({k: v for k, v in fs.items() if k in want or k == 'needle_max'})
        print(f'delta/oow/needle xong ({time.time() - t0:.0f}s)', flush=True)

    rows = []
    for name, vec in scores.items():
        order = list(np.argsort(-vec))
        print(f'\n[{name}] xếp hạng: {order}')
        print('  ' + '  '.join(f'{i}:{vec[i]:+.4f}' for i in order[:8]) + ' ...')
        for l in range(gates.n):
            rows.append(dict(tag=a.tag, model_config=a.model_config, scorer=name,
                             layer=l, score=float(vec[l]),
                             rank=int(order.index(l)), window=a.window, n_sink=a.n_sink,
                             variant=a.variant, L=a.L, n_items=len(items), seed=a.seed,
                             nll_z1=nll1, nll_z0=nll0))
    append_row(pd.DataFrame(rows), a.out)
    print(f'\n-> ghi {len(rows)} dòng vào {a.out} | tổng {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
