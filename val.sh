#!/bin/bash -e
#SBATCH --job-name=jepa-inference
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH -t 2:00:00
#SBATCH --output=logs/jepa-training_%j.out
#SBATCH --error=logs/jepa-training_%j.err
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=l40s_courant
# #SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="/scratch/jsc9903/RLFM/diffusion-forcing-jepa"
cd "$PROJECT_ROOT"

mkdir -p logs
export HOME="/scratch/jsc9903/home"
mkdir -p "$HOME"
export XDG_CACHE_HOME="$HOME/.cache"
export WANDB_DIR="/scratch/jsc9903/wandb"
export WANDB_CACHE_DIR="/scratch/jsc9903/wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Make conda activate work in batch mode
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/jsc9903/envs/dfot

python -m main \
  @DiT/B \
  @diffusion/continuous \
  +name=jepa_minecraft_tf_training \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  wandb.entity=local \
  wandb.mode=disabled \
  load=pretrained:DFoT_MCRAFT.ckpt  \
  algorithm.checkpoint.strict=false \
  "experiment.tasks=[validation]" \
  experiment.validation.batch_size=1 \
  dataset.num_eval_videos=50 \
  dataset.max_frames=32 \
  dataset.context_length=16 \
  dataset.n_frames=32 \
  experiment.find_unused_parameters=true \
  experiment.ema.enable=false
