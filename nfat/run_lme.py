"""
LongMemEval_24k — tái lập giao thức SWAA / NLL-guided (arXiv:2512.10411, 2606.27791).

Dùng NGUYÊN VĂN từ repo SWAA: bản vá vLLM (keep_first, force_fa_decode, non_sliding_layers),
chat template không thêm addition_prompt, fp16, enforce_eager, nhiệt độ 0, tối đa 10000 token
cho mô hình Thinking, hàm trích câu trả lời sau </think>, và ACCURACY_PROMPT của judge.

Khác biệt DUY NHẤT so với họ: judge. Họ dùng GPT-5-mini qua API; máy này không có khoá,
nên judge mặc định là một mô hình mở chạy cục bộ bằng vLLM với CÙNG prompt (xem --judge).
Thứ hạng giữa các bộ chọn tầng được so dưới cùng một judge; số tuyệt đối có thể lệch so với
bảng của họ. Đặt OPENAI_API_KEY và --judge openai:gpt-5-mini để chấm lại đúng như họ.

  # sinh
  python nfat/run_lme.py gen --model Qwen/Qwen3-4B-Thinking-2507 --name full
  python nfat/run_lme.py gen --model Qwen/Qwen3-4B-Thinking-2507 --name nllg \
      --W 2048 --keep_first 10 --fa_decode --layers 2,4,8,...
  # chấm
  python nfat/run_lme.py judge --names full,nllg
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

SWAA = os.environ.get('SWAA_REPO', './ext/sliding-window-attention-adaptation')
OUT_DIR = 'results/lme'


def load_data(tok, max_prompt_len, first=None, sample=None):
    from datasets import load_dataset
    df = load_dataset('yuyijiong/LongMemEval_24k', 'longmemeval_24k', split='train').to_pandas()
    if 'prompt' not in df.columns and 'context' in df.columns:
        df['prompt'] = 'Context: ' + df['context'] + '\n\nQuestion: ' + df['question']
    df['templated_prompt'] = df['prompt'].apply(lambda x: tok.apply_chat_template(
        [{'role': 'user', 'content': x}], tokenize=False, add_generation_prompt=True))
    df['prompt_len'] = df['templated_prompt'].apply(lambda x: len(tok.encode(x)))
    df = df[df.prompt_len <= max_prompt_len].reset_index(drop=True)
    if first:
        df = df.iloc[:first].reset_index(drop=True)
    if sample:                                 # tập con phân tầng theo question_type, seed cố định
        df = (df.groupby('question_type', group_keys=False)
                .apply(lambda g: g.sample(n=round(len(g) * sample / len(df)), random_state=0))
                .sort_index().reset_index(drop=True))
    return df


def gen(a):
    # BẮT BUỘC: engine V1 chạy ở tiến trình con 'spawn' (CUDA đã khởi tạo) sẽ MẤT monkey-patch SWAA
    # -> mọi cấu hình âm thầm chạy attention đầy đủ. Giữ engine trong tiến trình này.
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    sys.path.insert(0, SWAA)
    from swaa_patch import SWAAConfig, hack_vllm_swaa
    hack_vllm_swaa()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    thinking = 'thinking' in a.model.lower()
    max_comp = a.max_completion or (10000 if thinking else 2000)
    tok = AutoTokenizer.from_pretrained(a.model)
    df = load_data(tok, a.max_prompt_len, a.first, a.sample)
    layers = [int(x) for x in a.layers.split(',') if x != ''] if a.layers else []
    cfg = SWAAConfig(sliding_window_size=a.W or None, keep_first=a.keep_first,
                     force_fa_decode=a.fa_decode, non_sliding_layers=layers)
    print(f'-> {a.name}: {len(df)} mẫu | {cfg} | max_completion {max_comp}', flush=True)

    llm = LLM(model=a.model, dtype='float16', max_model_len=a.max_prompt_len + max_comp,
              gpu_memory_utilization=a.gpu_mem, tensor_parallel_size=1, enforce_eager=True,
              trust_remote_code=True, swaa_config=cfg)
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_comp)
    prompts = df.templated_prompt.tolist()
    resp, t0 = [], time.time()
    for i in range(0, len(prompts), a.batch):
        outs = llm.generate(prompts=prompts[i:i + a.batch], sampling_params=sp, use_tqdm=False)
        llm.reset_prefix_cache()
        resp += [o.outputs[0].text for o in outs]
        print(f'   {len(resp)}/{len(prompts)}  ({time.time() - t0:.0f}s)', flush=True)
    df['response'] = resp
    df['gen_seconds'] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = dict(name=a.name, model=a.model, W=a.W, keep_first=a.keep_first,
                fa_decode=a.fa_decode, layers=layers, n=len(df), max_completion=max_comp,
                seconds=time.time() - t0)
    df.drop(columns=['templated_prompt']).to_parquet(f'{OUT_DIR}/{a.name}.parquet')
    json.dump(meta, open(f'{OUT_DIR}/{a.name}.json', 'w'), indent=1)
    print(f'-> {OUT_DIR}/{a.name}.parquet ({time.time() - t0:.0f}s)', flush=True)


def judge(a):
    sys.path.insert(0, os.path.join(SWAA, 'Eval'))
    from gpt_eval import ACCURACY_PROMPT, extract_short_response
    from transformers import AutoTokenizer
    frames = {}
    for name in a.names.split(','):
        df = pd.read_parquet(f'{OUT_DIR}/{name}.parquet')
        meta = json.load(open(f'{OUT_DIR}/{name}.json'))
        thinking = 'thinking' in meta['model'].lower()
        df['short_response'] = df.response.apply(
            lambda x: extract_short_response(x, thinking, max_answer_len=4000))
        frames[name] = (df, meta)

    if a.judge.startswith('openai:'):
        from openai import OpenAI
        cli, jm = OpenAI(), a.judge.split(':', 1)[1]

        def ask(ps):
            return [cli.chat.completions.create(model=jm, temperature=1.0, messages=[
                {'role': 'user', 'content': p}]).choices[0].message.content for p in ps]
    else:
        from vllm import LLM, SamplingParams
        jtok = AutoTokenizer.from_pretrained(a.judge)
        jl = LLM(model=a.judge, max_model_len=8192, gpu_memory_utilization=a.gpu_mem,
                 enforce_eager=False)
        sp = SamplingParams(temperature=0.0, max_tokens=256)

        def ask(ps):
            chat = [jtok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                             add_generation_prompt=True) for p in ps]
            return [o.outputs[0].text for o in jl.generate(chat, sp, use_tqdm=False)]

    rows = []
    for name, (df, meta) in frames.items():
        ps, idx = [], []
        for i, r in df.iterrows():
            if r.short_response:
                ps.append(ACCURACY_PROMPT.format(question=r.question, gold_answer=r.answer,
                                                 generated_answer=r.short_response))
                idx.append(i)
        labels = ask(ps) if ps else []
        df['judge'] = 0
        df['judge_text'] = ''
        for i, t in zip(idx, labels):
            df.at[i, 'judge'] = 0 if 'wrong' in t.lower() else 1         # đúng quy tắc SWAA
            df.at[i, 'judge_text'] = t
        df.to_parquet(f'{OUT_DIR}/{name}.judged.parquet')
        row = dict(name=name, judge=a.judge, acc=100 * df.judge.mean(), n=len(df),
                   no_answer=int((df.short_response == '').sum()), W=meta['W'],
                   keep_first=meta['keep_first'], fa_decode=meta['fa_decode'],
                   layers=','.join(map(str, meta['layers'])), gen_seconds=meta['seconds'])
        if 'question_type' in df.columns:
            for qt, g in df.groupby('question_type'):
                row[f'acc_{qt}'] = 100 * g.judge.mean()
        rows.append(row)
        print(f"[{name:>12s}] acc {row['acc']:.1f}%  (không trả lời: {row['no_answer']})", flush=True)
    from nfat.csv_append import append_row
    append_row(pd.DataFrame(rows), 'results/lme.csv')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['gen', 'judge'])
    p.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507')
    p.add_argument('--name')
    p.add_argument('--W', type=int, default=0, help='0 = attention đầy đủ')
    p.add_argument('--keep_first', type=int, default=0)
    p.add_argument('--fa_decode', action='store_true')
    p.add_argument('--layers', default='', help='non_sliding_layers (giữ đầy đủ)')
    p.add_argument('--max_prompt_len', type=int, default=40000)
    p.add_argument('--max_completion', type=int, default=None)
    p.add_argument('--sample', type=int, default=None, help='tập con phân tầng N mẫu (seed 0)')
    p.add_argument('--first', type=int, default=None, help='chỉ chạy N mẫu đầu (thử nhanh)')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--gpu_mem', type=float, default=0.92)
    p.add_argument('--names', default='')
    p.add_argument('--judge', default='Qwen/Qwen3-30B-A3B-Instruct-2507-FP8')
    a = p.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gen(a) if a.cmd == 'gen' else judge(a)


if __name__ == '__main__':
    main()
