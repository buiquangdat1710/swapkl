"""
Cửa sổ trượt + "attention sink" trên mô hình GỐC — KHÔNG huấn luyện gì cả.

Đây là đường cơ sở mà Jolicoeur-Martineau et al. (arXiv:2608.28444, 8/2026) dùng để
tuyên bố rằng SWA + sink, không hề hậu huấn luyện, đạt chất lượng ngang hoặc hơn các
mô hình đã tuyến tính hoá (gồm cả LoLCATs), với chi phí bộ nhớ tương đương. Nếu điều
đó đúng trên Llama 3.2 1B thì toàn bộ khung LoLCATs cần được đặt lại vấn đề; nếu sai
thì đó cũng là một kết quả đáng công bố. Dù thế nào, phép đo này KHÔNG tốn một bước
huấn luyện nào.

Cách hiện thực: bọc `LlamaModel._update_causal_mask`. Mask 4 chiều mà hàm này trả về
được dùng chung cho MỌI tầng, nên chỉ cần sửa một chỗ là toàn mô hình đổi hành vi —
không đụng tới lớp attention, không phụ thuộc eager/sdpa, không phụ thuộc phiên bản
transformers.

Token được giữ lại tại vị trí truy vấn i:
    j <= i           (nhân quả, như cũ)
  VÀ ( i - j < W     (nằm trong cửa sổ trượt W token gần nhất, kể cả chính nó)
      HOẶC j < S )   (nằm trong S token đầu chuỗi — các "sink")

Dùng qua biến môi trường, mọi script đánh giá sẵn có đều chạy được không cần sửa:

    NFAT_SWA_WINDOW=64 NFAT_SWA_SINK=4 \
        python nfat/eval_lm_harness.py --model_config nf_llama3_2_1b_softmax --tag swa64s4

Vì sao phải có sink: Kuzina et al. (arXiv:2607.07706) chỉ ra sai số xấp xỉ softmax lớn
nhất ở ĐẦU chuỗi, nơi "projection residual" cao và "key concentration" thấp, cộng thêm
số hạng chệch đều 1/t bị bỏ sót. LoLCATs không xử lý riêng vùng này. Vài token đầu là
chỗ rẻ nhất để sửa.
"""
import os

import torch
from transformers.cache_utils import DynamicCache


def _causal_window_sink_keep(cache_position, kv_len, window, n_sink, device):
    """Ma trận bool (q_len, kv_len): True = được phép nhìn."""
    qpos = cache_position.view(-1, 1)                                  # (q, 1)
    kpos = torch.arange(kv_len, device=device).view(1, -1)             # (1, kv)
    causal = kpos <= qpos
    in_window = (qpos - kpos) < window
    is_sink = kpos < n_sink
    return causal & (in_window | is_sink)


def decoder_layers(model):
    """Danh sách tầng decoder, chạy được với cả mô hình gốc lẫn mô hình đã bọc PEFT."""
    m = model
    for attr in ('get_base_model',):
        if hasattr(m, attr):
            m = getattr(m, attr)()
    for path in (('model', 'layers'), ('layers',), ('model', 'model', 'layers')):
        cur = m
        try:
            for a in path:
                cur = getattr(cur, a)
            return list(cur)
        except AttributeError:
            continue
    raise RuntimeError(f'không tìm thấy danh sách tầng trong {type(model).__name__}')


def apply_swa_sink(model, window: int = 64, n_sink: int = 4, full_layers=(),
                   verbose: bool = True):
    """
    Ép mô hình chỉ nhìn `window` token gần nhất cộng `n_sink` token đầu chuỗi.

    `full_layers`: chỉ số các tầng được MIỄN TRỪ, tức vẫn dùng attention đầy đủ. Đây là
    cấu hình lai theo tầng: phần lớn tầng dùng cửa sổ trượt (trạng thái cố định), vài tầng
    giữ attention chính xác toàn chuỗi để làm nhiệm vụ truy xuất tầm xa. Cách hiện thực:
    bản vá mask dùng chung vẫn chạy cho mọi tầng, nhưng các tầng được miễn sẽ nhận lại
    mask nhân quả ĐẦY ĐỦ đã được cất giữ trước đó.

    Không đổi một trọng số nào. Gọi được nhiều lần (lần sau thay lần trước).
    """
    remove_swa_sink(model)
    full_layers = tuple(sorted(set(int(i) for i in full_layers)))
    base = getattr(model, 'model', model)
    if not hasattr(base, '_update_causal_mask'):
        raise RuntimeError(
            f'{type(base).__name__} không có _update_causal_mask; '
            'cần transformers >= 4.40 với kiến trúc Llama/Phi.')

    # gỡ bản vá cũ nếu có, để gọi lại không bị chồng lớp
    orig = getattr(base, '_nfat_orig_update_causal_mask', base._update_causal_mask)
    base._nfat_orig_update_causal_mask = orig

    def patched(attention_mask, input_tensor, cache_position, past_key_values,
                output_attentions=False, *a, **kw):
        # Với SinkWindowCache, bộ nhớ đệm ĐÃ bị cắt đúng bằng S+W, nên khi giải mã
        # từng token mọi thứ còn trong đệm đều được phép nhìn -> không cần mask.
        # (Gọi `orig` ở đây sẽ dựng mask theo độ dài tuyệt đối và lệch kích thước.)
        if isinstance(past_key_values, SinkWindowCache) and cache_position.shape[0] == 1:
            base._nfat_full_mask = None      # đệm đã đúng cho từng tầng -> không cần mask
            return None
        mask = orig(attention_mask, input_tensor, cache_position, past_key_values,
                    output_attentions, *a, **kw)
        base._nfat_full_mask = mask          # các tầng được miễn trừ đọc lại giá trị này
        device = input_tensor.device
        dtype = input_tensor.dtype
        q_len = cache_position.shape[0]
        # đường sdpa của transformers trả về None khi mask thuần nhân quả -> tự dựng
        kv_len = mask.shape[-1] if mask is not None else int(cache_position[-1]) + 1
        keep = _causal_window_sink_keep(cache_position, kv_len, window, n_sink, device)
        min_val = torch.finfo(dtype).min
        if mask is None:
            out = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
            out.masked_fill_(~keep, min_val)
            bsz = input_tensor.shape[0]
            return out.view(1, 1, q_len, kv_len).expand(bsz, 1, q_len, kv_len)
        # masked_fill chứ không cộng: cộng hai giá trị min của bf16 sẽ tràn thành -inf
        return mask.masked_fill(~keep.view(1, 1, q_len, kv_len), min_val)

    base._update_causal_mask = patched

    layers = decoder_layers(model)
    for i in full_layers:
        attn = layers[i].self_attn
        attn._nfat_orig_forward = attn.forward

        def make(a_):
            def fwd(*args, **kwargs):
                full = getattr(base, '_nfat_full_mask', None)
                if 'attention_mask' in kwargs:
                    kwargs['attention_mask'] = full
                elif len(args) >= 2:
                    args = (args[0], full) + args[2:]
                return a_._nfat_orig_forward(*args, **kwargs)
            return fwd
        attn.forward = make(attn)

    base._nfat_swa = (window, n_sink, full_layers)
    if verbose:
        extra = (f', {len(full_layers)} tầng attention đầy đủ {list(full_layers)}'
                 if full_layers else '')
        print(f'-> SWA+sink: cửa sổ {window} token, {n_sink} token sink{extra}'
              f', KHÔNG huấn luyện', flush=True)
    return model


class SinkWindowCache(DynamicCache):
    """
    Bộ nhớ đệm KV kích thước CỐ ĐỊNH: `n_sink` token đầu chuỗi + `window` token gần nhất.

    Bản vá mask ở trên chỉ mô phỏng SWA về mặt CHẤT LƯỢNG — nó vẫn giữ đủ KV cache và
    vẫn tốn $O(L^2)$. Lớp này mới là hiện thực thật: đệm không bao giờ vượt quá
    $S+W$ token, nên bộ nhớ phẳng theo độ dài ngữ cảnh, đúng tính chất mà linear
    attention được ca ngợi. Có lớp này thì mới so tốc độ với LoLCATs một cách sòng phẳng.

    Khoá được lưu SAU khi đã áp RoPE ở vị trí tuyệt đối thật, và `get_seq_length` trả về
    số token đã thấy chứ không phải độ dài đệm, nên vị trí RoPE của token mới vẫn đúng.
    Nhờ vậy đầu ra trùng khớp bản vá mask (đã kiểm thử), chứ không phải một xấp xỉ khác.
    """

    def __init__(self, window: int, n_sink: int, full_layers=()):
        super().__init__()
        self.window = window
        self.n_sink = n_sink
        self.full_layers = set(int(i) for i in full_layers)   # các tầng KHÔNG bị cắt đệm

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # số token ĐÃ THẤY, không phải số token còn giữ -> vị trí RoPE vẫn đúng
        return self._seen_tokens

    def _trim(self, x):
        return torch.cat([x[..., :self.n_sink, :], x[..., -self.window:, :]], dim=-2)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx in self.full_layers:        # tầng attention đầy đủ: giữ nguyên đệm
            return k, v
        if k.shape[-2] <= self.n_sink + self.window:
            return k, v
        kt, vt = self._trim(k), self._trim(v)
        self.key_cache[layer_idx], self.value_cache[layer_idx] = kt, vt
        if key_states.shape[-2] > 1:
            # Lượt nạp prompt: lớp attention cắt mask theo độ dài khoá
            # (`causal_mask[..., :key_states.shape[-2]]`), nên phải TRẢ VỀ đủ khoá để
            # mask còn khớp. Chỉ phần LƯU LẠI mới bị cắt — bước giải mã sau đó đã đúng.
            return k, v
        return kt, vt


def make_cache_for(model):
    """Trả về SinkWindowCache nếu mô hình đang bật SWA+sink, ngược lại None."""
    base = getattr(model, 'model', model)
    cfg = getattr(base, '_nfat_swa', None)
    return SinkWindowCache(*cfg) if cfg else None


def remove_swa_sink(model):
    base = getattr(model, 'model', model)
    if hasattr(base, '_nfat_orig_update_causal_mask'):
        base._update_causal_mask = base._nfat_orig_update_causal_mask
        del base._nfat_orig_update_causal_mask
    try:
        for l in decoder_layers(model):
            if hasattr(l.self_attn, '_nfat_orig_forward'):
                l.self_attn.forward = l.self_attn._nfat_orig_forward
                del l.self_attn._nfat_orig_forward
    except RuntimeError:
        pass
    base._nfat_swa = None
    base._nfat_full_mask = None
    return model


def maybe_apply_from_env(model):
    """Được `nfat/model_io.load_model` gọi. Không có biến môi trường thì không làm gì."""
    w = int(os.environ.get('NFAT_SWA_WINDOW', 0) or 0)
    if w <= 0:
        return model
    s = int(os.environ.get('NFAT_SWA_SINK', 0) or 0)
    fl = os.environ.get('NFAT_FULL_LAYERS', '').strip()
    full = [int(x) for x in fl.replace(',', ' ').split()] if fl else ()
    return apply_swa_sink(model, window=w, n_sink=s, full_layers=full)
