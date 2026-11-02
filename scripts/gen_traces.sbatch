#!/bin/bash
# C1 trace generation on one GPU (CURC Blanca). Submit from the repo root:
#   sbatch scripts/gen_traces.sbatch                       # Qwen/Qwen3-4B, full set
#   sbatch scripts/gen_traces.sbatch Qwen/Qwen3-8B         # another model
#   sbatch scripts/gen_traces.sbatch Qwen/Qwen3-4B --limit 8 --n-natural 1 --n-prefill 1   # smoke test
# Resubmitting is safe: the script skips generations already in the output file.
#
# Check partition / qos / account names with:  sacctmgr show assoc user=$USER format=account,partition,qos
#SBATCH --job-name=c1-traces
#SBATCH --partition=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODEL=${1:-Qwen/Qwen3-4B}
shift || true                                   # remaining args go straight to gen_traces.py

export HF_HOME=${HF_HOME:-/projects/$USER/hf}   # home quota is small; keep model weights here
export VLLM_CACHE_ROOT=/projects/$USER/.cache/vllm   # vLLM's torch.compile cache (defaults to ~/.cache)
source .venv-vllm/bin/activate                  # separate venv: vLLM pins its own torch/transformers

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

python scripts/gen_traces.py --backend vllm --model "$MODEL" "$@"
