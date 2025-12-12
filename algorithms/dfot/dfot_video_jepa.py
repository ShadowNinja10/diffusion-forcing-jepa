"""
JEPA-style VAE training integrated with DFoT for Minecraft.

Pipeline (Teacher Forcing):
1. Encode all frames: Image_t → VAE.encode().mode() → s_t
2. Encode actions: a_t → ActionEncoder → action_embed_t  
3. Combine into sequence: [s_0⊕a_0, s_1⊕a_1, ..., s_{T-1}⊕a_{T-1}]
4. ViT Predictor with causal attention predicts: [s_1_pred, s_2_pred, ..., s_T_pred]
5. JEPA Loss = distance(s_t_pred, s_t_target) for t=1..T

This is trained jointly with DFoT online.
"""

from typing import Optional, Any, Dict, Tuple
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat
from transformers import get_scheduler

from algorithms.vae import ImageVAE
from utils.torch_utils import freeze_model
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from .dfot_video import DFoTVideo


# =============================================================================
# JEPA Components
# =============================================================================

class FeedForward(nn.Module):
    """Feed-forward network with pre-norm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CausalAttention(nn.Module):
    """Multi-head attention with causal masking for temporal sequences."""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Causal mask: position i can only attend to positions <= i
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        dots = dots.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention."""
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ViTPredictor(nn.Module):
    """
    ViT-based predictor for JEPA with causal attention.
    
    Takes a sequence of (state, action) pairs and predicts next states.
    Uses teacher forcing: at position t, predicts s_{t+1} from s_0:t and a_0:t.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Combine state and action embeddings
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Transformer
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection: predict next state
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            states: (B, T, state_dim) - flattened latent states
            actions: (B, T, action_dim) - action embeddings
            
        Returns:
            pred_next_states: (B, T, state_dim) - predicted next states
                At position t, this is the prediction of s_{t+1}
        """
        B, T, _ = states.shape
        
        # Project states and actions
        state_emb = self.state_proj(states)  # (B, T, hidden_dim)
        action_emb = self.action_proj(actions)  # (B, T, hidden_dim)
        
        # Combine: each position has (state, action) info
        combined = torch.cat([state_emb, action_emb], dim=-1)  # (B, T, hidden_dim*2)
        x = self.combine_proj(combined)  # (B, T, hidden_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout_layer(x)
        
        # Transformer with causal attention
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Predict next state at each position
        # Output at position t is prediction for s_{t+1}
        pred_next_states = self.output_proj(x)  # (B, T, state_dim)
        
        return pred_next_states


class ActionEncoder(nn.Module):
    """Encodes actions into embeddings."""
    
    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        return self.net(actions)


# =============================================================================
# Main Algorithm
# =============================================================================

class DFoTVideoJEPA(DFoTVideo):
    """
    DFoT with JEPA-style VAE training.

    Key features:
    1. VAE encoder is trainable (learns predictable representations)
    2. ViT predictor learns temporal dynamics in latent space
    3. JEPA loss + DFoT loss jointly train the system
    4. Uses mode() for stable latent targets
    """

    def __init__(self, cfg: DictConfig):
        # JEPA-specific config
        self.jepa_cfg = cfg.jepa
        self.jepa_loss_weight = cfg.jepa.loss_weight
        self.train_vae_encoder = cfg.jepa.train_vae_encoder
        self.train_vae_decoder = cfg.jepa.train_vae_decoder
        self.use_mode_for_targets = cfg.jepa.get("use_mode_for_targets", True)

        # Force online latent mode for JEPA
        assert cfg.latent.enable, "JEPA training requires latent diffusion"
        assert cfg.latent.type == "online", "JEPA training requires online latent processing"

        super().__init__(cfg)

    def _build_model(self):
        # Build DFoT model (diffusion + metrics)
        super()._build_model()

        # Build JEPA components
        self._build_jepa_model()

    def _build_jepa_model(self):
        """Build ViT predictor and action encoder for JEPA."""
        jepa_cfg = self.jepa_cfg

        # Compute state dimension (flattened latent)
        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        self.state_dim = latent_channels * latent_h * latent_w
        
        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=self.external_cond_dim,
            embed_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.action_hidden_dim,
        )
        
        # ViT Predictor
        self.predictor = ViTPredictor(
            state_dim=self.state_dim,
            action_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.predictor_hidden_dim,
            depth=jepa_cfg.predictor_depth,
            heads=jepa_cfg.predictor_heads,
            dim_head=jepa_cfg.get("predictor_dim_head", 64),
            mlp_ratio=jepa_cfg.get("predictor_mlp_ratio", 4.0),
            max_seq_len=self.max_tokens + 1,
            dropout=jepa_cfg.get("predictor_dropout", 0.1),
        )
        
        rank_zero_print(cyan(f"JEPA State dim: {self.state_dim}"))
        rank_zero_print(cyan(f"JEPA Predictor hidden dim: {jepa_cfg.predictor_hidden_dim}"))

    def _load_vae(self) -> None:
        """Load the VAE model - optionally trainable for JEPA."""
        assert not self.is_latent_video_vae, "JEPA currently only supports ImageVAE"

        self.vae = ImageVAE.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)

        # Optionally freeze parts of VAE
        if not self.train_vae_encoder:
            freeze_model(self.vae.encoder)
            freeze_model(self.vae.quant_conv)
            rank_zero_print(cyan("VAE encoder is frozen"))
        else:
            rank_zero_print(cyan("VAE encoder is TRAINABLE"))

        if not self.train_vae_decoder:
            freeze_model(self.vae.decoder)
            freeze_model(self.vae.post_quant_conv)
            rank_zero_print(cyan("VAE decoder is frozen"))
        else:
            rank_zero_print(cyan("VAE decoder is TRAINABLE"))

    def _encode_for_jepa(self, x: Tensor, use_mode: bool = True) -> Tensor:
        """
        Encode images to latents for JEPA.
        
        Args:
            x: (B, T, C, H, W) videos normalized to [0, 1]
            use_mode: If True, use mode() (deterministic). If False, use sample().
        """
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        
        # Normalize to [-1, 1] for VAE
        x_normalized = 2.0 * x_flat - 1.0
        
        # Encode
        posterior = self.vae.encode(x_normalized)
        
        if use_mode:
            latents_flat = posterior.mode()  # Deterministic
        else:
            latents_flat = posterior.sample()  # Stochastic
        
        # Reshape back
        latent_shape = latents_flat.shape[1:]
        latents = latents_flat.view(B, T, *latent_shape)
        
        return latents

    def configure_optimizers(self):
        """Configure optimizers including VAE and JEPA components."""
        params_groups = []

        # 1. Diffusion model parameters
        diffusion_params = list(self.diffusion_model.parameters())
        params_groups.append({
            "params": diffusion_params,
            "lr": self.cfg.lr,
            "name": "diffusion",
        })

        # 2. JEPA components (action encoder + predictor)
        jepa_params = (
            list(self.action_encoder.parameters()) +
            list(self.predictor.parameters())
        )
        params_groups.append({
            "params": jepa_params,
            "lr": self.jepa_cfg.lr,
            "name": "jepa",
        })

        # 3. VAE parameters (if trainable)
        vae_params = []
        if self.train_vae_encoder:
            vae_params.extend(list(self.vae.encoder.parameters()))
            vae_params.extend(list(self.vae.quant_conv.parameters()))
        if self.train_vae_decoder:
            vae_params.extend(list(self.vae.decoder.parameters()))
            vae_params.extend(list(self.vae.post_quant_conv.parameters()))

        if vae_params:
            params_groups.append({
                "params": vae_params,
                "lr": self.jepa_cfg.vae_lr,
                "name": "vae",
            })

        optimizer = torch.optim.AdamW(
            params_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Preprocess batch for JEPA + DFoT training.
        
        Returns:
            xs: (B, T, C, H, W) normalized latent tokens for DFoT
            conditions: (B, T, cond_dim) processed conditions for DFoT
            masks: (B, T) valid frame masks
            gt_videos: (B, T, 3, H, W) ground truth videos
            actions_raw: (B, T, action_dim) raw actions for JEPA
        """
        # Store raw videos and actions for JEPA
        gt_videos = batch.get("videos", None)
        actions_raw = batch.get("conds", None)
        
        # For DFoT: encode videos and normalize
        if self.is_latent_diffusion and self.is_latent_online:
            # Use sample() for DFoT (matches original behavior)
            xs = self._encode(batch["videos"])
        else:
            xs = batch.get("latents", batch["videos"])
        
        xs = self._normalize_x(xs)

        # Process conditions for DFoT
        conditions = batch.get("conds", None)

        # Build masks
        if "masks" in batch:
            masks = batch["masks"]
        else:
            masks = torch.ones(*xs.shape[:2], dtype=torch.bool, device=self.device)

        return xs, conditions, masks, gt_videos, actions_raw

    def _compute_jepa_loss(
        self,
        videos: Tensor,
        actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute JEPA prediction loss using Teacher Forcing.
        
        Pipeline:
        1. Encode all frames with VAE (using mode() for stable targets)
        2. Flatten latents to state vectors
        3. Encode actions
        4. Feed (states, actions) to ViT predictor with causal attention
        5. Predictor output at position t is prediction for s_{t+1}
        6. Compute loss between predictions and targets
        
        Args:
            videos: (B, T, 3, H, W) raw videos in [0, 1]
            actions: (B, T, action_dim) raw actions
            masks: (B, T) valid frame masks
        """
        B, T = videos.shape[:2]
        
        if T < 2:
            return torch.tensor(0.0, device=videos.device), {}
        
        # 1. Encode all frames to latents using mode() for stable targets
        latents = self._encode_for_jepa(videos, use_mode=self.use_mode_for_targets)
        # latents: (B, T, C, H, W)
        
        # 2. Flatten latents to state vectors
        states = latents.view(B, T, -1)  # (B, T, state_dim)
        
        # 3. Encode actions
        action_embeds = self.action_encoder(actions)  # (B, T, action_embed_dim)
        
        # 4. Prepare inputs for predictor (Teacher Forcing)
        # Input: states and actions from t=0 to T-2
        # Target: states from t=1 to T-1
        states_input = states[:, :-1]  # (B, T-1, state_dim) - s_0 to s_{T-2}
        actions_input = actions[:, :-1]  # (B, T-1, action_dim) - a_0 to a_{T-2}
        # Note: We use raw actions here, encoder is applied inside
        
        # Re-encode actions for the input sequence
        action_embeds_input = self.action_encoder(actions_input)  # (B, T-1, action_embed_dim)
        
        states_target = states[:, 1:]  # (B, T-1, state_dim) - s_1 to s_{T-1}
        
        # 5. Predict with ViT (Teacher Forcing)
        # At position t (0-indexed), predictor sees s_0:t and a_0:t, predicts s_{t+1}
        states_pred = self.predictor(states_input, action_embeds_input)  # (B, T-1, state_dim)
        
        # 6. Compute loss
        # Transition masks: valid if both source and target frames are valid
        transition_masks = masks[:, :-1] & masks[:, 1:]  # (B, T-1)
        
        # Smooth L1 loss for robustness
        pred_loss = F.smooth_l1_loss(
            states_pred,
            states_target.detach(),  # Detach target to avoid trivial solution
            reduction="none"
        )  # (B, T-1, state_dim)
        
        # Average over state dimension
        pred_loss = pred_loss.mean(dim=-1)  # (B, T-1)
        
        # Apply masks
        if transition_masks.sum() > 0:
            pred_loss = (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
        else:
            pred_loss = pred_loss.mean()
        
        # Compute metrics
        with torch.no_grad():
            valid_mask = transition_masks.unsqueeze(-1).expand_as(states_pred)
            if valid_mask.sum() > 0:
                mse = F.mse_loss(
                    states_pred[valid_mask],
                    states_target[valid_mask]
                )
                # Cosine similarity
                pred_flat = states_pred[transition_masks]
                target_flat = states_target[transition_masks]
                cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            else:
                mse = torch.tensor(0.0, device=videos.device)
                cos_sim = torch.tensor(0.0, device=videos.device)
        
        log_dict = {
            "jepa/pred_loss": pred_loss,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
        }
        
        return pred_loss, log_dict

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """Training step with combined DFoT + JEPA loss."""
        xs, conditions, masks, gt_videos, actions_raw = batch

        # =============== DFoT Loss ===============
        noise_levels, masks_dfot = self._get_training_noise_levels(xs, masks)
        xs_pred, dfot_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        dfot_loss = self._reweight_loss(dfot_loss, masks_dfot)

        # =============== JEPA Loss ===============
        jepa_loss = torch.tensor(0.0, device=xs.device)
        jepa_log_dict = {}

        if gt_videos is not None and actions_raw is not None and self.jepa_loss_weight > 0:
            jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                gt_videos, actions_raw, masks
            )

        # =============== Combined Loss ===============
        total_loss = dfot_loss + self.jepa_loss_weight * jepa_loss

        # =============== Logging ===============
        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/dfot_loss", dfot_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/jepa_loss", jepa_loss, on_step=True, sync_dist=True)
            for key, value in jepa_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": total_loss,
            "dfot_loss": dfot_loss,
            "jepa_loss": jepa_loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """
        Validation step - handles the extended batch format from JEPA.
        Converts 5-element batch back to 4-element for parent validation methods.
        """
        # JEPA batch has 5 elements, parent expects 4
        xs, conditions, masks, gt_videos, actions_raw = batch
        
        # Create 4-element batch for parent methods
        parent_batch = (xs, conditions, masks, gt_videos)
        
        # 1. If running validation while training a model, directly evaluate
        if self.trainer.state.fn == "FIT":
            self._eval_denoising_jepa(parent_batch, batch_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):
            all_videos = self._sample_all_videos(parent_batch, batch_idx, namespace)
            if all_videos is not None:
                self._update_metrics(all_videos)
                self._log_videos(all_videos, namespace)

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        """
        Evaluate denoising performance - adapted for JEPA.
        Uses parent's training_step format (4 elements).
        """
        xs, conditions, masks, gt_videos = batch

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        # Create 5-element batch for JEPA training_step
        # Use conditions as actions_raw (they're the same data)
        jepa_batch = (xs, conditions, masks, gt_videos, conditions)
        output = self.training_step(jepa_batch, batch_idx, namespace=namespace)

        gt_videos_vis = gt_videos if self.is_latent_diffusion else output["xs"]
        recons = output["xs_pred"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos_vis.shape[1]:
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos_vis.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos_vis, recons = self.gather_data((gt_videos_vis, recons))

        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video
        
        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            gt_videos_vis.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos_vis[:num_videos_to_log],
            step=self.global_step,
            namespace="denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
        )

    def _should_include_in_checkpoint(self, key: str) -> bool:
        """Include JEPA components in checkpoint."""
        base_include = super()._should_include_in_checkpoint(key)
        jepa_include = (
            key.startswith("action_encoder") or
            key.startswith("predictor")
        )
        vae_include = (
            (self.train_vae_encoder or self.train_vae_decoder) and
            key.startswith("vae")
        )
        return base_include or jepa_include or vae_include

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save JEPA config alongside checkpoint."""
        super().on_save_checkpoint(checkpoint)
        checkpoint["jepa_cfg"] = self.jepa_cfg

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Handle loading from pre-trained DFoT checkpoint.
        
        When finetuning from a pre-trained DFoT model:
        - DFoT diffusion model weights are loaded from checkpoint
        - JEPA components (action_encoder, predictor) are randomly initialized
        - VAE is loaded separately via from_pretrained
        
        This allows you to:
        1. Load a pre-trained DFoT model: load=pretrained:XXX or load=wandb_run_id
        2. Load a pre-trained VAE: algorithm.vae.pretrained_path=pretrained:ImageVAE_MCRAFT.ckpt
        3. Train JEPA on top with fresh action_encoder and predictor
        """
        # Let parent handle DFoT model loading
        super().on_load_checkpoint(checkpoint)
        
        # Print info about what was loaded vs initialized
        loaded_keys = set(checkpoint.get("state_dict", {}).keys())
        jepa_keys = [k for k in self.state_dict().keys() 
                     if k.startswith("action_encoder") or k.startswith("predictor")]
        
        loaded_jepa = [k for k in jepa_keys if k in loaded_keys]
        new_jepa = [k for k in jepa_keys if k not in loaded_keys]
        
        if loaded_jepa:
            rank_zero_print(cyan(f"Loaded JEPA weights: {len(loaded_jepa)} parameters"))
        if new_jepa:
            rank_zero_print(cyan(f"Randomly initialized JEPA weights: {len(new_jepa)} parameters"))
            rank_zero_print(cyan("  (This is expected when finetuning from pre-trained DFoT)"))
        
        # VAE is loaded separately in _load_vae(), not from this checkpoint
        rank_zero_print(cyan(f"VAE will be loaded from: {self.cfg.vae.pretrained_path}"))
