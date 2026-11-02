#!/bin/bash
# Teacher-forced residuals at the C1 probe positions, one GPU (CURC Blanca). Submit from the repo root:
#   sbatch scripts/extract_resid.sbatch                         # Qwen/Qwen3-4B, all positions in the CSV
#   sbatch scripts/extract_resid.sbatch Qwen/Qwen3-4B --limit 20 --shard 10 --out data/activations/smoke_resid   # smoke
# Needs on the cluster: data/generations/c1_traces_{tag}.jsonl and data/activations/c1_positions_{tag}.csv
# Resubmitting is safe: finished shards are skipped.
#SBATCH --job-name=c1-resid
#SBATCH --partition=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODEL=${1:-Qwen/Qwen3-4B}
shift || true                                   # remaining args go straight to extract_resid.py

export HF_HOME=${HF_HOME:-/projects/$USER/hf}   # model weights already cached here by the trace runs
source .venv-vllm/bin/activate                  # has torch + transformers (4.51, supports Qwen3) + pandas

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

python scripts/extract_resid.py --model "$MODEL" "$@"
