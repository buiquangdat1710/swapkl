#!/bin/bash
# Đo độ chính xác NIAH thật (sinh văn bản + cache SWA có giới hạn) cho các cấu hình
# do layer_importance chọn ra. Seed 777 = seed GIỮ RIÊNG, khác seed hiệu chỉnh 1234.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
P=${PY:-python}
MODEL=${MODEL:-nf_llama3_2_1b_softmax}
VARIANTS=${VARIANTS:-s1,s2}
LENGTHS=${LENGTHS:-2048,4096,8192}
N=${N:-10}
WINDOW=${WINDOW:-512}
SINK=${SINK:-4}

run () {   # run <tag> <full_layers|"-">
  local tag=$1 layers=$2
  if [ "$layers" = "base" ]; then
    unset NFAT_SWA_WINDOW NFAT_SWA_SINK NFAT_FULL_LAYERS
  else
    export NFAT_SWA_WINDOW=$WINDOW NFAT_SWA_SINK=$SINK
    export NFAT_FULL_LAYERS=$layers
  fi
  echo "### $tag  (full_layers='${layers}')"
  $P nfat/eval_niah.py --model_config "$MODEL" --variants "$VARIANTS" \
     --lengths "$LENGTHS" --n "$N" --haystack data/haystack_text --seed 777 \
     --tag "$tag" || echo "!! $tag lỗi"
}

while [ $# -gt 0 ]; do
  run "${1%%=*}" "${1#*=}"
  shift
done
