"""
RULER đầy đủ (13 task của NVIDIA/RULER) bằng vLLM 0.11 + bản vá SWAA — benchmark KHÓ.

Dữ liệu từ bộ sinh ruler_niah.py (bản v2: key space wonderwords, template nguyên văn
constants.py của RULER). Mô hình base: prompt = BOS + body (body kết thúc bằng answer
prefix của RULER). Chấm recall chuỗi như RULER.

    python nfat/run_ruler_vllm.py --model meta-llama/Llama-3.1-8B --name base
    python nfat/run_ruler_vllm.py --model meta-llama/Llama-3.1-8B --name igg12 \
        --W 512 --keep_first 4 --layers 2,4,5,...
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser('~'))
SWAA = os.environ.get('SWAA_REPO', './ext/sliding-window-attention-adaptation')
TASKS = 'S-NIAH-PK,S-NIAH-N,S-NIAH-W,MK-1,MK-2,MK-3,MV,MQ,VT,CWE,FWE'


def _save_details(a, task, L, sc, dhash, append_row):
    """Điểm TỪNG MẪU -> results/details/ruler_vllm/<name>.csv (thêm 10-12 cho SwapKL v2: kiểm định ghép cặp theo mẫu).
    Không đổi điểm gộp; lỗi ở đây chỉ cảnh báo, không dừng đánh giá."""
    try:
        rows = pd.DataFrame([dict(name=a.name, model=a.model, W=a.W, keep_first=a.keep_first, fa_decode=a.fa_decode,
                                  layers=a.layers, seed=a.seed, task=task, length=L, i=j, score=float(v), data_hash=dhash)
                             for j, v in enumerate(sc)])
        append_row(rows, os.path.join(ROOT, 'results', 'details', 'ruler_vllm', f'{a.name}.csv'))
    except Exception as e:                                        # noqa: BLE001
        print(f'!! không ghi được điểm từng mẫu: {e}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--W', type=int, default=0)
    p.add_argument('--keep_first', type=int, default=4)
    p.add_argument('--fa_decode', action='store_true')
    p.add_argument('--layers', default='')
    p.add_argument('--tasks', default=TASKS)
    p.add_argument('--lengths', default='8192,16384,32768')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--chat', action='store_true')
    p.add_argument('--gpu_mem', type=float, default=0.9)
    p.add_argument('--out', default='results/ruler_hard.csv')
    a = p.parse_args()

    # engine V1 ở tiến trình con 'spawn' sẽ mất monkey-patch SWAA -> giữ engine trong tiến trình này
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    sys.path.insert(0, SWAA)
    from swaa_patch import SWAAConfig, hack_vllm_swaa
    hack_vllm_swaa()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import ruler_niah as R
    from nfat.csv_append import append_row
    from nfat.ruler_common import BaseProcessor, ChatProcessor, RULER_NAME

    tok = AutoTokenizer.from_pretrained(a.model)
    proc = ChatProcessor(tok) if a.chat else BaseProcessor(tok)
    gen = R.RulerNIAH(proc, R.load_essay_sentences(split='test'))
    cache = f"results/ruler_cache/{os.path.basename(a.model)}{'-chat' if a.chat else '-base'}"
    layers = [int(x) for x in a.layers.split(',') if x != '']
    cfg = SWAAConfig(sliding_window_size=a.W or None, keep_first=a.keep_first,
                     force_fa_decode=a.fa_decode, non_sliding_layers=layers)
    lengths = [int(x) for x in a.lengths.split(',')]
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=max(lengths) + 512,
              gpu_memory_utilization=a.gpu_mem, enforce_eager=True, swaa_config=cfg)
    print(f'-> {a.name}: {cfg}', flush=True)

    for L in lengths:
        for task in a.tasks.split(','):
            path = R.build_eval_set(gen, task, L, a.n, a.seed, cache_dir=cache)
            data = R.load_eval_set(path)
            sp = SamplingParams(temperature=0.0, max_tokens=R.max_new_tokens(task))
            t0 = time.time()
            prompts = [dict(prompt_token_ids=tok(s['prompt'], add_special_tokens=False).input_ids)
                       for s in data]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            sc = [R.score(o.outputs[0].text, s['answers']) for o, s in zip(outs, data)]
            acc = float(np.mean(sc))
            print(f'[{task:>9s} L={L:>5d}] {acc:.3f} ({time.time() - t0:.0f}s)', flush=True)
            append_row(pd.DataFrame([dict(name=a.name, model=a.model, task=task,
                                          ruler_task=RULER_NAME.get(task, task), length=L,
                                          acc=acc, n=len(sc), W=a.W, keep_first=a.keep_first,
                                          fa_decode=a.fa_decode, layers=a.layers,
                                          data_hash=R.file_hash(path),
                                          secs=time.time() - t0)]), a.out)
            _save_details(a, task, L, sc, R.file_hash(path), append_row)


if __name__ == '__main__':
    main()
