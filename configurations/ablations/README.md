# JEPA + DFoT Ablation Studies

## Quick Start

```bash
# Make the script executable
chmod +x run_ablations.sh

# Run a quick sanity test
./run_ablations.sh quick_test

# Run specific ablation
./run_ablations.sh loss_weight

# Run all key ablations
./run_ablations.sh all
```

## Ablation Parameters

### 1. Loss Configuration

| Parameter | Default | Test Values | Command |
|-----------|---------|-------------|---------|
| `jepa.loss_weight` | 1.0 | 0.1, 0.5, 1.0, 2.0, 5.0 | `algorithm.jepa.loss_weight=X` |
| `jepa.use_mode_for_targets` | true | true, false | `algorithm.jepa.use_mode_for_targets=X` |

**Hypothesis:** Higher loss weight → more predictable latents, but may hurt generation quality.

### 2. VAE Training

| Parameter | Default | Test Values | Command |
|-----------|---------|-------------|---------|
| `jepa.train_vae_encoder` | true | true, false | `algorithm.jepa.train_vae_encoder=X` |
| `jepa.train_vae_decoder` | false | true, false | `algorithm.jepa.train_vae_decoder=X` |
| `jepa.vae_lr` | 1e-5 | 1e-6, 1e-5, 1e-4 | `algorithm.jepa.vae_lr=X` |

**Hypothesis:** Training encoder helps learn predictable representations; decoder training may hurt reconstruction.

### 3. Predictor Architecture

| Parameter | Default | Test Values | Command |
|-----------|---------|-------------|---------|
| `jepa.predictor_depth` | 4 | 2, 4, 6, 8 | `algorithm.jepa.predictor_depth=X` |
| `jepa.predictor_hidden_dim` | 512 | 256, 512, 768, 1024 | `algorithm.jepa.predictor_hidden_dim=X` |
| `jepa.predictor_heads` | 8 | 4, 8, 12 | `algorithm.jepa.predictor_heads=X` |
| `jepa.predictor_dropout` | 0.1 | 0.0, 0.1, 0.2 | `algorithm.jepa.predictor_dropout=X` |

**Hypothesis:** Deeper/wider predictors → better predictions but slower; dropout helps generalization.

### 4. Action Encoding

| Parameter | Default | Test Values | Command |
|-----------|---------|-------------|---------|
| `jepa.action_embed_dim` | 256 | 64, 128, 256, 512 | `algorithm.jepa.action_embed_dim=X` |
| `jepa.action_hidden_dim` | 256 | 128, 256, 512 | `algorithm.jepa.action_hidden_dim=X` |

**Hypothesis:** Larger action embedding → better action representation, especially for complex actions.

## Recommended Ablation Order

1. **First: Baseline comparison** (`./run_ablations.sh comparison`)
   - Does JEPA help at all?
   - Does training VAE encoder help?

2. **Second: Loss weight** (`./run_ablations.sh loss_weight`)
   - Find the sweet spot for JEPA loss

3. **Third: Predictor capacity** (`./run_ablations.sh predictor_depth`)
   - How much model capacity is needed?

4. **Fourth: VAE training** (`./run_ablations.sh vae_training`)
   - What's the best VAE training strategy?

## Evaluation Metrics

Track these metrics during ablations:

### JEPA Metrics
- `jepa/pred_loss` - Prediction loss (lower = better predictions)
- `jepa/mse` - MSE between predicted and target states
- `jepa/cos_sim` - Cosine similarity (higher = better)

### DFoT Metrics
- `training/loss` - Total training loss
- `training/dfot_loss` - Diffusion loss
- `validation/fvd` - Fréchet Video Distance (lower = better quality)
- `validation/fid` - Fréchet Inception Distance
- `validation/lpips` - Perceptual similarity

## Example Command

```bash
# Single ablation run
python main.py \
  +name="jepa_ablation_loss_2.0" \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  algorithm.jepa.loss_weight=2.0 \
  algorithm.jepa.train_vae_encoder=true \
  wandb.project=dfot-jepa-ablations

# Override multiple parameters
python main.py \
  +name="jepa_large_predictor" \
  experiment=video_generation \
  dataset=minecraft \
  algorithm=dfot_video_jepa \
  algorithm.jepa.predictor_depth=6 \
  algorithm.jepa.predictor_hidden_dim=768 \
  algorithm.jepa.predictor_heads=12
```

## Results Template

| Ablation | JEPA Loss | Cos Sim | FVD | FID | Notes |
|----------|-----------|---------|-----|-----|-------|
| Baseline (no JEPA) | - | - | X | X | Pure DFoT |
| JEPA λ=0.5 | X | X | X | X | |
| JEPA λ=1.0 | X | X | X | X | Default |
| JEPA λ=2.0 | X | X | X | X | |
| VAE frozen | X | X | X | X | |
| VAE encoder trainable | X | X | X | X | |

