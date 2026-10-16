# SwapKL: Training-Free Layer Selection for Hybrid Full/Sliding-Window LLMs

Code for the paper *"SwapKL: Which Layers Need Full Attention? Training-Free Layer Selection for Hybrid Full/Sliding-Window LLMs"*.

A pretrained LLM is converted into a hybrid model: `k` of its `L` layers keep full attention, and the others use sliding-window
attention (SWA) with attention sinks. Weights are unchanged. SwapKL chooses the `k` layers in two parts:

1. **Cheap starting sets.** It takes the best of five starts. Four come from one-shot rankings: copy score, CAKE score, and the
   gradients of KL and of hidden-state MSE with respect to continuous layer gates at the deployed density `z = k/L`. The fifth
   interleaves the copy and CAKE rankings.
2. **Fixed-budget swap search.** It swaps one layer at a time, accepting a swap only if the decode-matched KL divergence to the
   original model decreases on a small calibration split. A held-out split makes the final choice.

Code comments are partly in Vietnamese; identifiers, docstrings of the main entry points and this README are in English.

## Layout

| Path | Content |
|---|---|
| `nfat/perbench_methods.py` | **SwapKL** (`swapkl`, `swapkl_fast` with exact prefix reuse), the objective `J` (`Obj`), starting rankings (`setup`, `_interleave_sets`) |
| `nfat/hyb.py` | hybrid full/SWA+sink attention (hard masks and continuous gates), SWA vs. full decoding of answer tokens |
| `nfat/baselines.py` | layer-level baselines: DuoAttention-style gates, Retrieval Head, CAKE, MoA-style gradient, BI (ShortGPT), NLL-guided |
| `nfat/perbench.py`, `nfat/perbench_data.py` | entry point for selection: `--scorers swapkl` (or a baseline) on 32 calibration samples |
| `nfat/ksweep.py` | runs every selector over the pre-set `k` grid and evaluates each layer set on the test benchmarks |
| `nfat/swk_ablate.py`, `nfat/swk_ablate_sweep.py`, `nfat/swk_ablate_report.py` | multi-variant ablation run in one pass with a shared measurement cache |
| `nfat/swapkl_v2.py`, `nfat/win80_methods.py` | measurement cache (`Meter`) and prefix-reuse evaluator (`PrefixX`) used by the ablations |
| `nfat/eval_ruler.py`, `nfat/run_ruler_vllm.py`, `nfat/eval_babilong.py`, `nfat/eval_niah.py`, `nfat/run_lme.py` | test evaluation (S-NIAH, RULER, BABILong, NIAH, LongMemEval) |
| `nfat/speed_offline.py` | speed measurement in vLLM with an SWA kernel |
| `nfat/paper_*.py`, `nfat/swapkl_report_*.py`, `nfat/ksweep_plot.py` | tables and figures of the paper (break-even, diagnostics, paired bootstrap, oracle) |
| `ruler_niah.py` | RULER-style data generator (S-NIAH and RULER tasks) |
| `src/model/` | model-loading utilities inherited from the LoLCATs code base |
| `configs/model/` | model configs used by the HF evaluation |
| `results/perbench/ksweep.csv` | **selected layer sets and test accuracy** of every method at every grid `k` (main results) |
| `results/swk_ablate/ksweep_ablate.csv` | layer sets and accuracy of every ablation variant |
| `patches/swaa_hack_vllm_0110.patch` | small compatibility fix to the SWAA vLLM patch for vLLM 0.11.0 |

## Setup

Two Python environments were used:

- **vLLM environment** (`PY_VLLM`): selection, RULER, LongMemEval and speed. Python 3.11, torch 2.8.0, transformers 4.57,
  vLLM 0.11.0, plus the SWAA vLLM patch and its flash-attention SWA kernel.
- **HF environment** (`PY_HF`): S-NIAH, BABILong and NIAH evaluation with transformers. Python 3.10, torch 2.4.1,
  transformers 4.43.1.

Both environments also need pandas, numpy, scipy, omegaconf, datasets and matplotlib.

External repositories go under `./ext/`:

```bash
git clone https://github.com/yuyijiong/sliding-window-attention-adaptation ext/sliding-window-attention-adaptation
git -C ext/sliding-window-attention-adaptation checkout 06fb05591da949c9d97aac51b6653c514800ff97
git -C ext/sliding-window-attention-adaptation apply ../../patches/swaa_hack_vllm_0110.patch
git clone https://github.com/yuyijiong/flash-attention-SWAA ext/flash-attention-SWAA      # build per its README (commit f2f28e6)
git clone https://github.com/booydar/babilong ext/babilong                                # commit 7a6efee
export SWAA_REPO=$PWD/ext/sliding-window-attention-adaptation BABILONG_REPO=$PWD/ext/babilong PYTHONPATH=$PWD/ext/babilong
export PY_VLLM=/path/to/vllm-env/bin/python PY_HF=/path/to/hf-env/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONHASHSEED=0      # NIAH test seeds depend on hash()
```

Data:

- `data/haystack_text/rp.txt`: a plain-text corpus used as the NIAH haystack (we used about 13 MB of web text; it is not
  redistributed here).
- `data/babilong_train/`: from RMT-team/babilong-train-5k-samples, for BABILong calibration.
- The LongMemEval test set (`yuyijiong/LongMemEval_24k`) and the calibration pool (THUDM/LongAlign-10k,
  yuyijiong/fusang-v1-filtered) are downloaded from the Hugging Face Hub.

## Running

Select layers with SwapKL for one budget, for example RULER 8K on Llama-3.1-8B with W=512, 4 sinks and SWA decoding:

```bash
$PY_VLLM nfat/perbench.py --model meta-llama/Llama-3.1-8B --bench ruler --W 512 --S 4 --k 8 --decode swa \
    --scorers swapkl --tag my-ruler-k8
$PY_VLLM nfat/pick_layers.py --tag my-ruler-k8 --scorer swapkl --k 8      # prints the selected layer set
```

Benchmarks (`--bench`): `sniah`, `ruler`, `babilong2`, `niah` (SWA decoding), and `longalign` with `--decode full` for LongMemEval.

Run every selector over the full grid and evaluate on the test sets (results appended to `results/perbench/ksweep.csv`):

```bash
$PY_VLLM nfat/ksweep.py run --only ruler
```

Run the ablations over the grid (tier 1 and 2 variants; tier 3 runs the NLL-objective variant and measures `J` of all evaluated sets):

```bash
$PY_VLLM nfat/swk_ablate_sweep.py run --only ruler --tier 2
```

Regenerate the tables and figures:

```bash
SWK_REPORT=nosleb python nfat/swapkl_report_tables.py
python nfat/paper_figs.py; python nfat/paper_time.py; python nfat/paper_diag.py; python nfat/swk_ablate_report.py
```

Hyper-parameters are fixed for all models and benchmarks:

- 32 calibration samples, split into A and B within strata;
- 8 samples for the start race and for removal harm;
- 4 samples for screening; all of A for confirmation;
- 2 kept layers and 2 confirmed candidates per round;
- at most 3 rounds.
