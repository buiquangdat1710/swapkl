"""
Quét k (lưới rút gọn, người dùng chọn 10-04): mọi baseline (Duo, RH, CAKE, MoA, BI, SLEB) + SwapKL v1.1 + FRS-4 trên mọi benchmark
của giao thức PerBench. Đánh giá theo TẬP TẦNG (tập trùng nhau giữa phương pháp chỉ đánh giá một lần). Tự bỏ qua bước đã xong.
Kết quả: results/perbench/ksweep.csv (setting, method, k, layers, eval_tag, score); chi phí chọn tầng lấy từ results/select_cost.csv.

  python nfat/ksweep.py run [--only niah1b,sniah512]     # chạy (trong srun trên GPU)
  python nfat/ksweep.py table                              # in bảng
Đánh giá rút gọn cho quét k: S-NIAH chỉ 4K (3 tác vụ × 50, đúng chỉ số TB@4K); RULER chỉ 8K, 25 mẫu/tác vụ; BABILong chỉ 2K/4K;
NIAH với PYTHONHASHSEED=0 (cùng tập test); LME chỉ k thuộc lưới riêng (RH/MoA/BI thêm 10-11 cho báo cáo SwapKL).
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PS = os.environ.get('PY_VLLM', sys.executable)   # env with vLLM + SWAA patch
PN = os.environ.get('PY_HF', sys.executable)     # env with transformers (HF eval)
L8B, L8I, L1B, Q = 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-1B', 'Qwen/Qwen3-4B-Thinking-2507'
M8, M1 = 'base_llama3_1_8b_sdpa', 'nf_llama3_2_1b_softmax'
OUT = 'results/perbench/ksweep.csv'
BASE = ['duo', 'rethead', 'cake', 'moa', 'bi', 'sleb']
K8 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]   # 10-12: thêm 3,5,7,10,14,20 cho đường cong mượt hơn (báo cáo SwapKL)
S = {  # setting: model, bench, W, Snk, decode, L, ks, tag cũ của baseline, k_run của SLEB, tag mới (swapkl/frs ở k mặc định), k mặc định, kiểu đánh giá
    'niah1b':   dict(model=L1B, bench='niah', W=512, Sk=4, dec='swa', L=16, ks=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], btag='1b-w512', krun=4,
                     ntag={4: 'pb-niah-1b-k4', 6: 'pb-niah-1b-k6'}, ev='niah1b', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'sniah512': dict(model=L8B, bench='sniah', W=512, Sk=4, dec='swa', L=32, ks=K8, btag='pb-sniah-w512', krun=8,
                     ntag={8: 'pb-sniah-w512'}, ev='sniah4k', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'bab':      dict(model=L8B, bench='babilong2', W=256, Sk=4, dec='swa', L=32, ks=K8, btag='pb-bab2-w256', krun=8,
                     ntag={8: 'pb-bab2-w256'}, ev='bab24', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'sniah256': dict(model=L8B, bench='sniah', W=256, Sk=4, dec='swa', L=32, ks=K8, btag='pb-sniah-w256', krun=8,
                     ntag={8: 'pb-sniah-w256'}, ev='sniah4k', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'ruler':    dict(model=L8B, bench='ruler', W=512, Sk=4, dec='swa', L=32, ks=K8, btag='pb-ruler-w512', krun=8,
                     ntag={8: 'pb-ruler-w512'}, ev='ruler8k', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'niah8b':   dict(model=L8B, bench='niah', W=512, Sk=4, dec='swa', L=32, ks=K8, btag='8b-w512', krun=8,
                     ntag={8: 'pb-niah-8b'}, ev='niah8b', methods=BASE + ['swapkl', 'frs', 'nll_swa']),
    'lme_l8i':  dict(model=L8I, bench='longalign', W=2048, Sk=10, dec='full', L=32, ks=[4, 8, 16], btag='l8i-w2048', krun=8,
                     ntag={8: 'pb-lme-l8i'}, ev='lme:l8i', methods=['cake', 'duo', 'sleb', 'rethead', 'moa', 'bi', 'swapkl', 'frs', 'nll_swa']),
    'lme_q4t':  dict(model=Q, bench='longalign', W=2048, Sk=10, dec='full', L=36, ks=[4, 9, 18], btag='q4t-w2048', krun=9,
                     ntag={9: 'pb-lme-q4t'}, ev='lme:q4t', methods=['cake', 'duo', 'sleb', 'rethead', 'moa', 'bi', 'swapkl', 'frs', 'nll_swa']),
}
ENV = dict(os.environ, HF_HOME=os.path.expanduser('~/.cache/huggingface'), TOKENIZERS_PARALLELISM='false',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', BABILONG_REPO='./ext/babilong',
           PYTHONPATH='./ext/babilong')


def sh(cmd, env=None, log=True):
    if log:
        print('$', ' '.join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, env=env or ENV, shell=isinstance(cmd, str), capture_output=False, text=True).returncode


def pick(tag, scorer, k):
    r = subprocess.run([PS, 'nfat/pick_layers.py', '--tag', tag, '--scorer', scorer, '--k', str(k)], capture_output=True, text=True)
    out = r.stdout.strip().splitlines()
    return out[-1] if r.returncode == 0 and out else ''


def has_scorer(tag, scorer):
    li = pd.read_csv('results/layer_importance.csv', usecols=['tag', 'scorer'])
    return bool(((li.tag == tag) & (li.scorer == scorer)).any())


def select(st, m, k):
    """Trả tập tầng (chuỗi) của phương pháp m ở k; chạy chọn tầng nếu cần."""
    c = S[st]
    if m in ('nll_swa', 'nll_swa_dm'):                     # NLL-guided chọn trên tag PerBench (top-k tĩnh -> mọi k)
        k0, tag = min(c['ntag'].items())
        if not has_scorer(tag, m):                         # 10-14: LME chưa có điểm NLL-guided -> chấm một lần (32 mẫu hiệu chỉnh)
            sh([PS, 'nfat/perbench.py', '--model', c['model'], '--bench', c['bench'], '--W', str(c['W']), '--S', str(c['Sk']),
                '--k', str(k0), '--decode', c['dec'], '--scorers', m, '--tag', tag])
        return pick(tag, m, k)
    if m in ('duo', 'rethead', 'cake', 'moa', 'bi'):
        return pick(c['btag'], m, k)                       # bảng xếp hạng một lần -> mọi k
    if m == 'sleb':
        if k >= c['krun']:
            return pick(c['btag'], 'sleb', k)              # thứ tự loại bỏ -> tập ở mọi k >= k_run
        tt = c['btag'] + '-tail'
        if not has_scorer(tt, 'sleb_tail'):
            start = pick(c['btag'], 'sleb', c['krun']).replace(',', '_')
            sh([PS, 'nfat/perbench.py', '--model', c['model'], '--bench', c['bench'], '--W', str(c['W']), '--S', str(c['Sk']),
                '--k', '1', '--decode', c['dec'], '--scorers', 'sleb_tail', '--tag', tt, '--extra', f'start={start}'])
        return pick(tt, 'sleb_tail', k)
    tag = c['ntag'].get(k, f"{c['ntag'][min(c['ntag'])]}-k{k}")      # swapkl / frs: chọn riêng cho từng k
    if not has_scorer(tag, m):
        sh([PS, 'nfat/perbench.py', '--model', c['model'], '--bench', c['bench'], '--W', str(c['W']), '--S', str(c['Sk']),
            '--k', str(k), '--decode', c['dec'], '--scorers', m, '--tag', tag])
    return pick(tag, m, k)


# ----------------------------------------------------------------------------- đánh giá (khử trùng theo tập tầng)
def norm(ly):
    return ','.join(str(x) for x in sorted(int(v) for v in ly.split(',')))


def lname(ly):
    return ly.replace(',', '_')


def score_existing(ev, W, ly):
    """Điểm của tập ly nếu đã có lần đánh giá phù hợp (bất kỳ tag nào). None nếu chưa."""
    if ev == 'sniah4k':
        d = pd.read_csv('results/ruler.csv', dtype={'full_layers': str}).dropna(subset=['full_layers'])
        d = d[(d.window.astype(float) == W) & (d.length == 4096) & d.task.isin(['S-NIAH-PK', 'S-NIAH-N', 'S-NIAH-W'])]
        d = d[d.full_layers.map(norm) == ly]
        if d.task.nunique() == 3:
            return d.drop_duplicates('task', keep='last').acc.mean(), d.tag.iloc[-1]
    elif ev == 'ruler8k':
        d = pd.read_csv('results/ruler_hard.csv', dtype={'layers': str}).dropna(subset=['layers'])
        d = d[(d.W == W) & (d.length == 8192) & (d.layers.map(norm) == ly)]
        if d.task.nunique() == 11:
            return d.drop_duplicates('task', keep='last').acc.mean(), d.name.iloc[-1]
    elif ev == 'bab24':
        d = pd.read_csv('results/babilong.csv', dtype={'full_layers': str}).dropna(subset=['full_layers'])
        d = d[(d.window.astype(float) == W) & d.length.isin(['2k', '4k']) & (d.full_layers.map(norm) == ly)]
        if len(d.drop_duplicates(['task', 'length'])) == 10:
            return d.drop_duplicates(['task', 'length'], keep='last').acc.mean(), d.tag.iloc[-1]
    elif ev in ('niah8b', 'niah1b'):
        mp = pd.read_csv(MAP) if os.path.exists(MAP) else pd.DataFrame(columns=['ev', 'layers', 'tag'])
        t = mp[(mp.ev == ev) & (mp.layers == ly)]
        if len(t):
            ni = pd.read_csv('results/niah.csv'); g = ni[ni.tag == t.tag.iloc[-1]]
            if len(g) >= 6:
                return g.acc.mean(), t.tag.iloc[-1]
    elif ev.startswith('lme:'):
        pre = ev.split(':')[1]
        d = pd.read_csv('results/lme.csv'); d = d[d.name.str.startswith(pre + '_') & (d.layers.astype(str) == ly) & (d.n == 500)]
        if len(d):
            return d.acc.iloc[-1] / 100, d.name.iloc[-1]
    return None


MAP = 'results/perbench/ksweep_niah_map.csv'


def seed_niah_map():
    """Các lần đánh giá NIAH hs0 đã có (tag -> tập tầng)."""
    rows = []
    for model, k, tag, ev in [('llama3.1-8b', 8, '8b-w512', 'niah8b'), ('llama3.2-1b', 4, '1b-w512', 'niah1b'), ('llama3.2-1b', 6, '1b-w512', 'niah1b')]:
        out = subprocess.run([PS, 'nfat/perbench_niah_sets.py', model, str(k), tag], capture_output=True, text=True).stdout.split()
        rows += [dict(ev=ev, layers=norm(x.split('=')[1]), tag=x.split('=')[0]) for x in out]
    for m in ['swapkl', 'poolelim', 'raceelim', 'frs', 'nllg_greedy']:
        for ev, tag, k, t in [('niah8b', 'pb-niah-8b', 8, f'hs0-pb-niah8b-{m}-k8'), ('niah1b', 'pb-niah-1b-k4', 4, f'hs0-pb-1b-{m}-k4'),
                              ('niah1b', 'pb-niah-1b-k6', 6, f'hs0-pb-1b-{m}-k6')]:
            ly = pick(tag, m, k)
            if ly:
                rows.append(dict(ev=ev, layers=norm(ly), tag=t))
    old = pd.read_csv(MAP) if os.path.exists(MAP) else pd.DataFrame(columns=['ev', 'layers', 'tag'])
    pd.concat([old, pd.DataFrame(rows)]).drop_duplicates().to_csv(MAP, index=False)


def evaluate(st, ly):
    c = S[st]; ev, W = c['ev'], c['W']
    got = score_existing(ev, W, ly)
    if got:
        return got
    tag = f'ks-{st}-{lname(ly)}'
    env = dict(ENV, NFAT_SWA_WINDOW=str(W), NFAT_SWA_SINK='4', NFAT_FULL_LAYERS=ly)
    if ev == 'sniah4k':
        sh([PN, 'nfat/eval_ruler.py', '--model_config', M8, '--tasks', 'S-NIAH-PK,S-NIAH-N,S-NIAH-W', '--lengths', '4096', '--n', '50',
            '--tag', tag], env=env)
    elif ev == 'ruler8k':
        sh([PS, 'nfat/run_ruler_vllm.py', '--model', L8B, '--n', '25', '--name', tag, '--W', '512', '--keep_first', '4',
            '--layers', ly, '--lengths', '8192'])
    elif ev == 'bab24':
        sh([PN, 'nfat/eval_babilong.py', '--model_config', M8, '--tasks', 'qa1,qa2,qa3,qa4,qa5', '--lengths', '2k,4k', '--n', '50',
            '--tag', tag], env=env)
    elif ev in ('niah8b', 'niah1b'):
        t = 'hs0-' + tag
        sh(f'MODEL={M8 if ev == "niah8b" else M1} PYTHONHASHSEED=0 bash slurm/run_niah_selected.sh {t}={ly}')
        pd.concat([pd.read_csv(MAP), pd.DataFrame([dict(ev=ev, layers=ly, tag=t)])]).to_csv(MAP, index=False)
    elif ev.startswith('lme:'):
        pre = ev.split(':')[1]; name = f'{pre}_ks_{lname(ly)}'
        if not os.path.exists(f'results/lme/{name}.parquet'):
            sh([PS, 'nfat/run_lme.py', 'gen', '--model', c['model'], '--name', name, '--W', '2048', '--keep_first', '10',
                '--fa_decode', '--layers', ly])
        sh([PS, 'nfat/run_lme.py', 'judge', '--names', name])
    return score_existing(ev, W, ly)


def run(only, methods=()):
    seed_niah_map()
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['setting', 'method', 'k', 'layers', 'eval_tag', 'score'])
    for st in (only or list(S)):
        c = S[st]
        # baseline trước (chọn tầng miễn phí), rồi SwapKL/FRS-4 (chọn lại theo k)
        for m in [x for x in c['methods'] if x not in ('swapkl', 'frs')] + ['swapkl', 'frs']:
            if methods and m not in methods:
                continue
            ks_run = c['ks']
            if os.environ.get('KS_STD') == '1':          # 10-14: chỉ lưới CHUẨN của báo cáo SwapKL (bỏ k dày) — dùng cho NLL-guided
                ks_run = [k for k in ks_run if k in ([1, 2, 3, 4, 6, 8, 12] if st == 'niah1b' else [1, 2, 4, 6, 8, 12, 16, 24, 4, 9, 18, 16])]
            for k in ks_run:
                if ((done.setting == st) & (done.method == m) & (done.k == k)).any():
                    continue
                ly = select(st, m, k)
                if not ly:
                    print(f'!! không có tập {st}/{m}/k{k}', flush=True); continue
                ly = norm(ly)
                got = evaluate(st, ly)
                if not got:
                    print(f'!! đánh giá lỗi {st}/{m}/k{k} [{ly}]', flush=True); continue
                row = dict(setting=st, method=m, k=k, layers=ly, eval_tag=got[1], score=got[0])
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(OUT, index=False)
                print(f'== {st} {m} k={k} [{ly}] -> {got[0]:.4f} ({got[1]})', flush=True)
        print(f'XONG {st}', flush=True)
    print('XONG ksweep', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['run', 'table']); p.add_argument('--only', default='')
    p.add_argument('--methods', default='', help='chỉ chạy các phương pháp này (rỗng = tất cả)')
    a = p.parse_args()
    if a.cmd == 'run':
        run([x for x in a.only.split(',') if x], [x for x in a.methods.split(',') if x])
    else:
        d = pd.read_csv(OUT)
        for st, g in d.groupby('setting'):
            print(st); print(g.pivot_table(index='k', columns='method', values='score').round(3).to_string())
