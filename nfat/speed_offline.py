"""
Đo tốc độ THẬT với vLLM 0.11 + kernel SWAA, engine chạy TRONG tiến trình (không mất monkey-patch).
Mỗi cấu hình: prompt token ngẫu nhiên dài L.
  - prefill: max_tokens=1  -> thời gian ~ TTFT (cả lô)
  - decode : max_tokens=D, ignore_eos -> (t_decode - t_prefill) / D  ~ TPOT
Ghi results/speed_offline.csv.

python nfat/speed_offline.py --name igg --W 2048 --keep_first 10 --fa_decode --layers 7,13 --L 32768
"""
import argparse
import os
import sys
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
SWAA = os.environ.get('SWAA_REPO', './ext/sliding-window-attention-adaptation')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507')
    p.add_argument('--name', required=True)
    p.add_argument('--W', type=int, default=0)
    p.add_argument('--keep_first', type=int, default=0)
    p.add_argument('--fa_decode', action='store_true')
    p.add_argument('--layers', default='')
    p.add_argument('--L', type=int, default=32768)
    p.add_argument('--n', type=int, default=4, help='số prompt mỗi lô')
    p.add_argument('--D', type=int, default=256, help='số token sinh khi đo decode')
    p.add_argument('--out', default='results/speed_offline.csv')
    a = p.parse_args()

    sys.path.insert(0, SWAA)
    from swaa_patch import SWAAConfig, hack_vllm_swaa
    hack_vllm_swaa()
    import numpy as np
    import pandas as pd
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    layers = [int(x) for x in a.layers.split(',') if x]
    cfg = SWAAConfig(sliding_window_size=a.W or None, keep_first=a.keep_first,
                     force_fa_decode=a.fa_decode, non_sliding_layers=layers)
    llm = LLM(model=a.model, dtype='float16', max_model_len=a.L + a.D + 16, enforce_eager=True,
              gpu_memory_utilization=0.9, swaa_config=cfg, enable_prefix_caching=False,
              max_num_batched_tokens=16384)
    vocab = llm.get_tokenizer().vocab_size

    def batch(seed):
        r = np.random.default_rng(seed)
        return [TokensPrompt(prompt_token_ids=r.integers(1000, vocab - 1000, a.L).tolist()) for _ in range(a.n)]

    def run(max_tokens, seed):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        t = time.perf_counter()
        llm.generate(batch(seed), sp, use_tqdm=False)
        return time.perf_counter() - t

    run(1, 99)                                  # khởi động
    t_pre = float(np.median([run(1, s) for s in range(2)]))
    t_dec = run(a.D, 7)
    row = dict(name=a.name, L=a.L, n=a.n, D=a.D, W=a.W, keep_first=a.keep_first, fa_decode=a.fa_decode,
               layers=a.layers, prefill_s=t_pre, prefill_tok_s=a.n * a.L / t_pre,
               tpot_ms=1000 * (t_dec - t_pre) / a.D, total_s=t_dec)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([row]).to_csv(a.out, mode='a', header=not os.path.exists(a.out), index=False)


if __name__ == '__main__':
    main()
