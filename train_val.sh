#!/bin/bash -e
#SBATCH --job-name=jepa-training_dual_vis
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 0:30:00
#SBATCH --output=logs/jepa-training_%j.out
#SBATCH --error=logs/jepa-training_%j.err
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_courant
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

# Allow PyTorch to use fragmented reserved-but-unallocated memory.
# Needed when DFoT+JEPA activations fill most VRAM and the decoder aux loss
# needs a small allocation from the reserved pool.
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Make conda activate work in batch mode
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/jsc9903/envs/dfot

srun python main.py \
  +name=jepa_dual_encoder_decoder_vis \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  dataset_experiment=minecraft_video_generation_jepa \
  @DiT/B \
  @diffusion/continuous \
  wandb.entity=jsc9903-new-york-university \
  wandb.mode=online \
  wandb.project=dfot-jepa \
  load=pretrained:DFoT_MCRAFT.ckpt \
  algorithm.checkpoint.strict=false \
  experiment.training.batch_size=4 \
  experiment.validation.batch_size=1 \
  algorithm.vae.batch_size=4 \
  experiment.training.max_epochs=15 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.checkpointing.enable_version_counter=true \
  experiment.training.checkpointing.every_n_epochs=null \
  dataset.subdataset_size=null \
  experiment.validation.limit_batch=1.0 \
  experiment.find_unused_parameters=true \
  dataset.max_frames=12 \
  dataset.context_length=6 \
  dataset.n_frames=12 \
  "+experiment.training.checkpointing.save_top_k=-1" \
  algorithm.jepa.decoder_lpips_every=1 \
  experiment.validation.val_every_n_step=1 \
  dataset.num_eval_videos=50 \
