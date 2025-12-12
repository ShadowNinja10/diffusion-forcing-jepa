#!/bin/bash
# JEPA + DFoT Ablation Study Runner
# Usage: ./run_ablations.sh [ablation_name]

set -e

# Base command
BASE_CMD="python main.py experiment=video_generation dataset=minecraft algorithm=dfot_video_jepa"

# Wandb project for ablations
WANDB_PROJECT="dfot-jepa-ablations"

echo "============================================"
echo "JEPA + DFoT Ablation Studies"
echo "============================================"

run_ablation() {
    local name=$1
    local override=$2
    echo ""
    echo "Running: $name"
    echo "Override: $override"
    echo "--------------------------------------------"
    $BASE_CMD +name="$name" wandb.project=$WANDB_PROJECT $override
}

case "${1:-all}" in
    # ==========================================
    # Loss Weight Ablations
    # ==========================================
    "loss_weight")
        run_ablation "jepa_loss_0.1" "algorithm.jepa.loss_weight=0.1"
        run_ablation "jepa_loss_0.5" "algorithm.jepa.loss_weight=0.5"
        run_ablation "jepa_loss_1.0" "algorithm.jepa.loss_weight=1.0"
        run_ablation "jepa_loss_2.0" "algorithm.jepa.loss_weight=2.0"
        run_ablation "jepa_loss_5.0" "algorithm.jepa.loss_weight=5.0"
        ;;

    # ==========================================
    # Predictor Depth Ablations
    # ==========================================
    "predictor_depth")
        run_ablation "jepa_depth_2" "algorithm.jepa.predictor_depth=2"
        run_ablation "jepa_depth_4" "algorithm.jepa.predictor_depth=4"
        run_ablation "jepa_depth_6" "algorithm.jepa.predictor_depth=6"
        run_ablation "jepa_depth_8" "algorithm.jepa.predictor_depth=8"
        ;;

    # ==========================================
    # Predictor Hidden Dim Ablations
    # ==========================================
    "predictor_hidden")
        run_ablation "jepa_hidden_256" "algorithm.jepa.predictor_hidden_dim=256"
        run_ablation "jepa_hidden_512" "algorithm.jepa.predictor_hidden_dim=512"
        run_ablation "jepa_hidden_768" "algorithm.jepa.predictor_hidden_dim=768"
        run_ablation "jepa_hidden_1024" "algorithm.jepa.predictor_hidden_dim=1024"
        ;;

    # ==========================================
    # VAE Training Ablations
    # ==========================================
    "vae_training")
        run_ablation "jepa_vae_frozen" "algorithm.jepa.train_vae_encoder=false algorithm.jepa.train_vae_decoder=false"
        run_ablation "jepa_vae_encoder" "algorithm.jepa.train_vae_encoder=true algorithm.jepa.train_vae_decoder=false"
        run_ablation "jepa_vae_both" "algorithm.jepa.train_vae_encoder=true algorithm.jepa.train_vae_decoder=true"
        ;;

    # ==========================================
    # VAE Learning Rate Ablations
    # ==========================================
    "vae_lr")
        run_ablation "jepa_vae_lr_1e6" "algorithm.jepa.train_vae_encoder=true algorithm.jepa.vae_lr=1e-6"
        run_ablation "jepa_vae_lr_1e5" "algorithm.jepa.train_vae_encoder=true algorithm.jepa.vae_lr=1e-5"
        run_ablation "jepa_vae_lr_1e4" "algorithm.jepa.train_vae_encoder=true algorithm.jepa.vae_lr=1e-4"
        ;;

    # ==========================================
    # Action Embedding Dim Ablations
    # ==========================================
    "action_embed")
        run_ablation "jepa_action_64" "algorithm.jepa.action_embed_dim=64"
        run_ablation "jepa_action_128" "algorithm.jepa.action_embed_dim=128"
        run_ablation "jepa_action_256" "algorithm.jepa.action_embed_dim=256"
        run_ablation "jepa_action_512" "algorithm.jepa.action_embed_dim=512"
        ;;

    # ==========================================
    # Target Mode (sample vs mode)
    # ==========================================
    "target_mode")
        run_ablation "jepa_mode_true" "algorithm.jepa.use_mode_for_targets=true"
        run_ablation "jepa_mode_false" "algorithm.jepa.use_mode_for_targets=false"
        ;;

    # ==========================================
    # Predictor Dropout Ablations
    # ==========================================
    "dropout")
        run_ablation "jepa_dropout_0.0" "algorithm.jepa.predictor_dropout=0.0"
        run_ablation "jepa_dropout_0.1" "algorithm.jepa.predictor_dropout=0.1"
        run_ablation "jepa_dropout_0.2" "algorithm.jepa.predictor_dropout=0.2"
        ;;

    # ==========================================
    # Combined: Best vs Baseline
    # ==========================================
    "comparison")
        # Baseline: No JEPA (pure DFoT)
        run_ablation "baseline_no_jepa" "algorithm.jepa.loss_weight=0.0"
        
        # JEPA with frozen VAE
        run_ablation "jepa_frozen_vae" "algorithm.jepa.loss_weight=1.0 algorithm.jepa.train_vae_encoder=false"
        
        # JEPA with trainable VAE encoder
        run_ablation "jepa_train_encoder" "algorithm.jepa.loss_weight=1.0 algorithm.jepa.train_vae_encoder=true"
        ;;

    # ==========================================
    # Quick test (small configs)
    # ==========================================
    "quick_test")
        run_ablation "quick_test" \
            "experiment.training.max_epochs=1 \
             experiment.training.batch_size=4 \
             algorithm.jepa.predictor_depth=2 \
             algorithm.jepa.predictor_hidden_dim=256 \
             dataset.subdataset_size=1000"
        ;;

    # ==========================================
    # All ablations
    # ==========================================
    "all")
        echo "Running all ablations sequentially..."
        $0 comparison
        $0 loss_weight
        $0 predictor_depth
        $0 vae_training
        ;;

    *)
        echo "Usage: $0 [ablation_name]"
        echo ""
        echo "Available ablations:"
        echo "  loss_weight     - Test JEPA loss weight (0.1, 0.5, 1.0, 2.0, 5.0)"
        echo "  predictor_depth - Test predictor depth (2, 4, 6, 8)"
        echo "  predictor_hidden- Test predictor hidden dim (256, 512, 768, 1024)"
        echo "  vae_training    - Test VAE training strategies"
        echo "  vae_lr          - Test VAE learning rates"
        echo "  action_embed    - Test action embedding dims"
        echo "  target_mode     - Test mode() vs sample() for targets"
        echo "  dropout         - Test predictor dropout"
        echo "  comparison      - Compare baseline vs JEPA variants"
        echo "  quick_test      - Quick sanity check with small config"
        echo "  all             - Run key ablations"
        ;;
esac

echo ""
echo "============================================"
echo "Ablation studies complete!"
echo "============================================"

