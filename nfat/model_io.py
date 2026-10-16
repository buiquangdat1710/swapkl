"""
Nạp mô hình (gốc / LoRA-only / tuyến tính hoá) từ config + checkpoint để đánh giá.

    model, tok, cfg = load_model('nf_llama3_2_1b',
                                 distill_ckpt='checkpoints/.../..._distill.pt',
                                 finetune_ckpt='checkpoints/.../..._ft.pt')
`finetune_config` được suy ra từ tên checkpoint (`-f=<name>-`) nếu không truyền.
"""
import os
import re
import sys
from os.path import join

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, join(ROOT, 'src'))

from src.model.pretrained import get_pretrained_loader           # noqa: E402
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune  # noqa: E402
from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers  # noqa: E402


def _seed(seed):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def finetune_name_from_ckpt(path):
    m = re.search(r'-f=([^-]+)-', os.path.basename(path))
    return m.group(1) if m else None


def load_model(model_config: str, distill_ckpt: str = None, finetune_ckpt: str = None,
               finetune_config: str = None, lk_zero_init: bool = True, seed: int = 0,
               config_dir: str = join(ROOT, 'configs'), huggingface_token: str = None):
    cfg = OmegaConf.load(join(config_dir, 'model', f'{model_config}.yaml'))
    at = cfg.attention.attention_type
    if at != 'softmax' and lk_zero_init:
        cfg.attention.learned_kernel_kwargs.zero_init = True
    loader = get_pretrained_loader(**cfg.model, huggingface_token=huggingface_token)
    tok = loader.load_tokenizer()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _seed(seed)   # same seed as distill_llama.py -> reproduces random init of untrained kernels
    model = loader.load(model_type=at)
    if at != 'softmax':
        model.state_chunk_len = cfg.attention.state_chunk_len
        model, _ = load_and_convert_attns(model, cfg, attention_type=at,
                                          checkpoint_path=distill_ckpt, merge_loras=False,
                                          train_converted=False, train_attention=False)
        model = toggle_attention(model, train=False)
        model = remove_base_attention(model)
    if finetune_ckpt is not None:
        name = finetune_config or finetune_name_from_ckpt(finetune_ckpt)
        assert name is not None, 'cannot infer finetune config; pass finetune_config='
        ft_cfg = OmegaConf.load(join(config_dir, 'experiment', f'{name}.yaml'))
        model, _ = load_and_convert_finetune(model, ft_cfg, checkpoint_path=finetune_ckpt,
                                             merge_loras=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.config.use_cache = False
    if torch.cuda.is_available():
        model.to('cuda')
    # Bản vá SWA+sink theo biến môi trường. Hàm này vốn được viết để gọi từ đây (xem
    # docstring của swa_sink.maybe_apply_from_env) nhưng call site bị thiếu, nên
    # NFAT_SWA_WINDOW / NFAT_SWA_SINK / NFAT_FULL_LAYERS không có tác dụng gì:
    # mọi lệnh "SWA thuần" trong docstring của eval_niah.py thực chất chạy MÔ HÌNH GỐC.
    # Dấu hiệu nhận ra: SWA-512 đạt đúng 1.000 trên NIAH ở 8192, bằng hệt mô hình gốc,
    # và thời gian chạy trùng khít từng giây.
    from nfat.swa_sink import maybe_apply_from_env      # noqa: E402  (tránh vòng import)
    model = maybe_apply_from_env(model)
    return model, tok, cfg


def is_linearized(model) -> bool:
    try:
        return any(getattr(l.self_attn, 'converted', False) for l in traverse_layers(model))
    except Exception:
        return False
