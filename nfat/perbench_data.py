"""
Dữ liệu hiệu chỉnh THEO BENCHMARK cho giao thức PerBench (results/perbench/LOG.md, mục "Giao thức").
Mỗi benchmark có tập tầng riêng, chọn trên dữ liệu cùng loại tác vụ nhưng KHÔNG phải dữ liệu test:

  sniah     bộ sinh RULER (ruler_niah.py) S-NIAH-PK/N/W, essay split 'train' (test dùng 'test'),
            seed 1234 (test 777), 4096 token, độ sâu xoay vòng DEPTH_GRID, xếp xoay vòng tác vụ.
  ruler     như trên nhưng 11 tác vụ RULER, 8192 token (test 8K/16K/32K). Một tập cho cả RULER.
  babilong  RMT-team/babilong-train-5k-samples (bAbI train + nhiễu) qa1-5 ở 2k và 4k, prompt và
            "\\nAnswer:" như nfat/eval_babilong.py; loại mọi mẫu chứa trọn một câu chuyện test (cùng câu hỏi).
  longalign / niah   như báo cáo chính (LongMemEval; NIAH trong phân phối).

Mỗi mẫu: dict(ids, gold, span, task). span = (s0, s1) vùng mà Retrieval Head coi là kim (None = bỏ qua);
với babilong span = toàn ngữ cảnh (đầu chép bất kỳ token trùng đáp án trong ngữ cảnh).
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser('~'))

SNIAH_TASKS = ['S-NIAH-PK', 'S-NIAH-N', 'S-NIAH-W']
RULER_TASKS = ['S-NIAH-PK', 'S-NIAH-N', 'S-NIAH-W', 'MK-1', 'MK-2', 'MK-3', 'MV', 'MQ', 'VT', 'CWE', 'FWE']
NEEDLE_TASKS = {'S-NIAH-PK', 'S-NIAH-N', 'S-NIAH-W', 'MK-1', 'MK-2', 'MK-3', 'MV', 'MQ'}
BAB_TASKS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']


def _span(ids, tok, ans):
    from nfat.baselines import needle_span
    return needle_span(ids, tok, ans)


def ruler_items(tok, tasks, length, n, W, seed=1234):
    """n mẫu, xếp xoay vòng tác vụ; độ sâu xoay vòng DEPTH_GRID theo từng tác vụ. Bỏ mẫu có kim
    (đáp án đầu tiên) nằm sẵn trong cửa sổ cuối (không mang tín hiệu chọn tầng)."""
    import ruler_niah as R
    from nfat.ruler_common import BaseProcessor
    gen = R.RulerNIAH(BaseProcessor(tok), R.load_essay_sentences(split='train'))
    rng = random.Random(seed)
    per = {t: 0 for t in tasks}
    items, guard = [], 0
    while len(items) < n and guard < 20 * n:
        t = tasks[guard % len(tasks)]; guard += 1
        d = R.DEPTH_GRID[per[t] % len(R.DEPTH_GRID)]; per[t] += 1
        s = gen.sample(t, length, d, rng)
        ids = tok(s['prompt'], add_special_tokens=False).input_ids
        gold = tok(R.target_text(s), add_special_tokens=False).input_ids
        span = _span(ids, tok, s['answers'][0]) if t in NEEDLE_TASKS else None
        if span is not None and span[0] >= len(ids) - W:
            continue
        items.append(dict(ids=ids, gold=gold, span=span, task=t, depth=d, value=str(s['answers'][0])))
    return items


def _sents(x):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', x) if s.strip()]


def babilong_items(tok, n, lengths=('2k', '4k'), seed=1234, root='data/babilong_train/data', n_test=100):
    from datasets import load_dataset
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    rng = random.Random(seed)
    pools = {}
    for t in BAB_TASKS:
        te = load_dataset('RMT-team/babilong', '0k', split=t).select(range(n_test))
        tests = [(_sents(s['input']), s['question']) for s in te]
        for L in lengths:
            d = json.load(open(os.path.join(ROOT, root, t, f'{L}.json')))
            idx = list(range(len(d))); rng.shuffle(idx)
            keep = []
            for i in idx[:200]:
                s = d[i]
                leak = any(q == s['question'] and all(f in s['input'] for f in F) for F, q in tests)
                if not leak:
                    keep.append(s)
            pools[(t, L)] = keep
    keys = [(t, L) for L in lengths for t in BAB_TASKS]
    items, ptr = [], {k: 0 for k in keys}
    while len(items) < n:
        k = keys[len(items) % len(keys)]
        s = pools[k][ptr[k]]; ptr[k] += 1
        pr = DEFAULT_PROMPTS[k[0]]
        text = get_formatted_input(s['input'], s['question'], pr['examples'], pr['instruction'],
                                   pr['post_prompt'], template=DEFAULT_TEMPLATE)
        ids = tok(text + '\nAnswer:').input_ids
        gold = tok(' ' + s['target'], add_special_tokens=False).input_ids
        items.append(dict(ids=ids, gold=gold, span=(0, len(ids)), task=f'{k[0]}-{k[1]}', value=str(s['target'])))
    return items


def bench_items(tok, bench, W, n=32, seed=1234):
    if bench == 'sniah':
        return ruler_items(tok, SNIAH_TASKS, 4096, n, W, seed)
    if bench == 'ruler':
        return ruler_items(tok, RULER_TASKS, 8192, n, W, seed)
    if bench == 'babilong':
        return babilong_items(tok, n, seed=seed)
    if bench == 'longalign':
        from nfat.hyb_select import calib_items
        return calib_items(tok, n, 16000, 32000, 20, 256, seed)
    if bench == 'niah':
        from nfat.baselines import niah_items
        it = niah_items(tok, 's2', 4096, W, [0, 0.25, 0.5, 0.75], n // 4, seed, 'data/haystack_text')
        # bộ sinh trả theo từng độ sâu -> xếp xoay vòng để mọi tập con đầu phủ đủ độ sâu (lỗi DM-SLEB vòng trước)
        by = {}
        for x in it:
            g = round(x['span'][0] * 4 / max(1, len(x['ids']))) if x['span'] else 0
            x['task'] = f'niah-d{g}'                     # tầng lớp độ sâu (để chia A/B trong từng độ sâu)
            by.setdefault(g, []).append(x)
        groups = [by[g] for g in sorted(by)]
        out = []
        for i in range(max(len(g) for g in groups)):
            out += [g[i] for g in groups if i < len(g)]
        return out
    raise ValueError(bench)


if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True)
    p.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--W', type=int, default=512)
    p.add_argument('--n', type=int, default=32)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    it = bench_items(tok, a.bench, a.W, a.n)
    import collections
    print(len(it), 'mẫu;', collections.Counter(x['task'] for x in it))
    print('độ dài TB', sum(len(x['ids']) for x in it) / len(it), '| gold TB', sum(len(x['gold']) for x in it) / len(it))
    print('span None:', sum(x['span'] is None for x in it))
    x = it[0]; print(repr(tok.decode(x['ids'][-60:])), '->', repr(tok.decode(x['gold'])))
