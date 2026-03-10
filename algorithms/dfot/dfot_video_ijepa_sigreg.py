"""
Joint temporal JEPA + spatial I-JEPA + SigREG variant for DFoT.

Extends DFoTVideoJEPASpatialSigREG with a per-frame spatial I-JEPA objective:
  - Temporal JEPA:  predict z_{t+1} from (z_t, a_t) — across time.
  - Spatial I-JEPA:  mask latent patches, predict masked from visible — within each frame.
  - SigREG:          push embeddings toward isotropic Gaussian.

The spatial masking forces the online encoder to maintain fine-grained spatial
structure in its latents, directly combating the blur observed when decoding
from a temporal-only JEPA encoder.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor
from transformers import get_scheduler

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_jepa_spatial_sigreg import DFoTVideoJEPASpatialSigREG


# =============================================================================
# I-JEPA Block Mask Generator
# =============================================================================


class BlockMaskGenerator:
    """
    I-JEPA style block mask generator on a 2D patch grid.

    Samples multiple rectangular *target* blocks; everything outside the
    union of target blocks is *context* (visible to the predictor).
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        num_targets: int = 4,
        target_scale_range: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
        min_context_ratio: float = 0.25,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_targets = num_targets
        self.target_scale_range = target_scale_range
        self.target_aspect_ratio_range = target_aspect_ratio_range
        self.min_context_ratio = min_context_ratio

    def _sample_one(self) -> Tensor:
        """Return a flat (P,) bool target mask for a single sample."""
        mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        num_patches = self.grid_h * self.grid_w

        for _ in range(self.num_targets):
            scale = random.uniform(*self.target_scale_range)
            ar = random.uniform(*self.target_aspect_ratio_range)

            n = max(1, int(scale * num_patches))
            bh = max(1, min(int(round(math.sqrt(n * ar))), self.grid_h))
            bw = max(1, min(int(round(math.sqrt(n / ar))), self.grid_w))

            top = random.randint(0, max(0, self.grid_h - bh))
            left = random.randint(0, max(0, self.grid_w - bw))
            mask[top : top + bh, left : left + bw] = True

        flat = mask.flatten()

        max_targets = int((1.0 - self.min_context_ratio) * num_patches)
        if flat.sum() > max_targets:
            idx = flat.nonzero(as_tuple=True)[0]
            keep = idx[torch.randperm(len(idx))[:max_targets]]
            flat = torch.zeros(num_patches, dtype=torch.bool)
            flat[keep] = True

        if flat.sum() == 0:
            flat[random.randint(0, num_patches - 1)] = True

        return flat

    def __call__(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            context_mask: (B, P) bool — True = visible context
            target_mask:  (B, P) bool — True = target to predict
        """
        target_masks = torch.stack([self._sample_one() for _ in range(batch_size)]).to(
            device
        )
        context_masks = ~target_masks
        return context_masks, target_masks


# =============================================================================
# I-JEPA Spatial Predictor
# =============================================================================


class SpatialIJEPAPredictor(nn.Module):
    """
    I-JEPA predictor for spatial latent patches.

    Given all patch tokens (visible context tokens kept, learnable mask tokens
    placed at target positions), runs self-attention and produces predictions
    at masked positions in the original patch-vector space.
    """

    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        num_patches: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.embed = nn.Linear(patch_dim, hidden_dim)
        self.unembed = nn.Linear(hidden_dim, patch_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.unembed.weight)
        nn.init.zeros_(self.unembed.bias)

    def forward(
        self,
        patches: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            patches:      (B, P, patch_dim) — from online encoder
            context_mask: (B, P) bool — True = keep original token
            target_mask:  (B, P) bool — True = replace with mask token
        Returns:
            (B, P, patch_dim) — predictions at every position (use target_mask for loss).
        """
        B, P, _ = patches.shape

        tokens = self.embed(patches)  # (B, P, D)

        mask_tok = self.mask_token.expand(B, P, -1)
        tokens = torch.where(
            target_mask.unsqueeze(-1).expand_as(tokens), mask_tok, tokens
        )

        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        return self.unembed(tokens)


# =============================================================================
# Main Algorithm
# =============================================================================


class DFoTVideoIJEPASigREG(DFoTVideoJEPASpatialSigREG):
    """
    DFoT with joint temporal JEPA + spatial I-JEPA + SigREG.

    Loss = temporal_pred + sigreg + ijepa_spatial
    """

    def __init__(self, cfg: DictConfig):
        self.ijepa_loss_weight = float(cfg.jepa.get("ijepa_loss_weight", 1.0))
        self.ijepa_patch_size = int(cfg.jepa.get("ijepa_patch_size", 2))
        self.ijepa_hidden_dim = int(cfg.jepa.get("ijepa_hidden_dim", 256))
        self.ijepa_depth = int(cfg.jepa.get("ijepa_depth", 4))
        self.ijepa_heads = int(cfg.jepa.get("ijepa_heads", 8))
        self.ijepa_dropout = float(cfg.jepa.get("ijepa_dropout", 0.1))
        self.ijepa_num_target_blocks = int(cfg.jepa.get("ijepa_num_target_blocks", 4))
        self.ijepa_target_scale = tuple(cfg.jepa.get("ijepa_target_scale", [0.15, 0.2]))
        self.ijepa_target_aspect = tuple(cfg.jepa.get("ijepa_target_aspect", [0.75, 1.5]))
        self.ijepa_min_context = float(cfg.jepa.get("ijepa_min_context", 0.25))
        self.ijepa_vis_every = int(cfg.jepa.get("ijepa_vis_every", 500))
        super().__init__(cfg)

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_jepa_model(self):
        super()._build_jepa_model()

        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h

        ps = self.ijepa_patch_size
        if latent_h % ps != 0 or latent_w % ps != 0:
            raise ValueError(
                f"Latent size ({latent_h}, {latent_w}) not divisible by ijepa_patch_size={ps}"
            )

        self.ijepa_grid_h = latent_h // ps
        self.ijepa_grid_w = latent_w // ps
        num_patches = self.ijepa_grid_h * self.ijepa_grid_w
        self.ijepa_patch_dim = latent_channels * ps * ps

        self.mask_generator = BlockMaskGenerator(
            grid_h=self.ijepa_grid_h,
            grid_w=self.ijepa_grid_w,
            num_targets=self.ijepa_num_target_blocks,
            target_scale_range=self.ijepa_target_scale,
            target_aspect_ratio_range=self.ijepa_target_aspect,
            min_context_ratio=self.ijepa_min_context,
        )

        self.spatial_predictor = SpatialIJEPAPredictor(
            patch_dim=self.ijepa_patch_dim,
            hidden_dim=self.ijepa_hidden_dim,
            depth=self.ijepa_depth,
            heads=self.ijepa_heads,
            num_patches=num_patches,
            dropout=self.ijepa_dropout,
        )

        n_sp = sum(p.numel() for p in self.spatial_predictor.parameters())
        rank_zero_print(
            cyan(
                f"I-JEPA spatial predictor: grid={self.ijepa_grid_h}x{self.ijepa_grid_w}, "
                f"patch_dim={self.ijepa_patch_dim}, hidden={self.ijepa_hidden_dim}, "
                f"depth={self.ijepa_depth}, params={n_sp / 1e6:.1f}M"
            )
        )
        rank_zero_print(cyan(f"I-JEPA loss weight: {self.ijepa_loss_weight}"))

    # -----------------------------------------------------------------
    # Patchification for I-JEPA
    # -----------------------------------------------------------------

    def _patchify_ijepa(self, z: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, P, C*ps*ps)"""
        ps = self.ijepa_patch_size
        return rearrange(
            z,
            "b c (gh ps1) (gw ps2) -> b (gh gw) (c ps1 ps2)",
            ps1=ps,
            ps2=ps,
            gh=self.ijepa_grid_h,
            gw=self.ijepa_grid_w,
        )

    # -----------------------------------------------------------------
    # I-JEPA spatial loss
    # -----------------------------------------------------------------

    def _compute_ijepa_loss(
        self,
        online_latents: Tensor,
        target_latents: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Per-frame spatial I-JEPA: mask patches, predict masked from visible.

        Args:
            online_latents: (B, T, C, H, W)  — gradients flow through
            target_latents: (B, T, C, H, W)  — detached target encoder
            masks:          (B, T) bool       — per-frame validity
        """
        B, T = online_latents.shape[:2]
        device = online_latents.device

        online_flat = rearrange(online_latents, "b t c h w -> (b t) c h w")
        target_flat = rearrange(target_latents, "b t c h w -> (b t) c h w")
        frame_valid = masks.reshape(-1)  # (B*T,)

        online_patches = self._patchify_ijepa(online_flat)
        target_patches = self._patchify_ijepa(target_flat)

        context_mask, target_mask = self.mask_generator(B * T, device)

        pred_patches = self.spatial_predictor(online_patches, context_mask, target_mask)

        if frame_valid.all():
            pred_sel = pred_patches[target_mask]
            gt_sel = target_patches[target_mask].detach()
        else:
            valid_target = target_mask & frame_valid.unsqueeze(-1)
            pred_sel = pred_patches[valid_target]
            gt_sel = target_patches[valid_target].detach()

        if pred_sel.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "ijepa/spatial_pred_loss": zero,
                "ijepa/spatial_mse": zero,
                "ijepa/spatial_cos_sim": zero,
                "ijepa/mask_ratio": zero,
            }

        loss = F.smooth_l1_loss(pred_sel, gt_sel)

        with torch.no_grad():
            mse = F.mse_loss(pred_sel, gt_sel)
            cos_sim = F.cosine_similarity(pred_sel, gt_sel, dim=-1).mean()
            mask_ratio = target_mask.float().mean()

        return loss, {
            "ijepa/spatial_pred_loss": loss,
            "ijepa/spatial_mse": mse,
            "ijepa/spatial_cos_sim": cos_sim,
            "ijepa/mask_ratio": mask_ratio,
        }

    # -----------------------------------------------------------------
    # Override JEPA loss to add spatial I-JEPA
    # -----------------------------------------------------------------

    def _compute_jepa_loss(
        self,
        online_latents: Tensor,
        actions: Tensor,
        masks: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss, log_dict = super()._compute_jepa_loss(
            online_latents, actions, masks, timings, _tick
        )

        b, t = online_latents.shape[:2]
        if t < 1:
            return total_loss, log_dict

        target_latents = self._cached_target_latents[:, :t]
        ijepa_loss, ijepa_log = self._compute_ijepa_loss(
            online_latents, target_latents, masks
        )

        total_loss = total_loss + self.ijepa_loss_weight * ijepa_loss
        log_dict.update(ijepa_log)
        log_dict["ijepa/weighted_spatial_loss"] = self.ijepa_loss_weight * ijepa_loss

        return total_loss, log_dict

    # -----------------------------------------------------------------
    # Optimizers — add spatial predictor to JEPA group
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        params_groups = []
        params_groups.append(
            {
                "params": list(self.diffusion_model.parameters()),
                "lr": self.cfg.lr,
                "name": "diffusion",
            }
        )

        jepa_params = (
            list(self.predictor.parameters())
            + list(self.action_encoder.parameters())
            + list(self.spatial_predictor.parameters())
        )
        if self.sigreg_proj is not None:
            jepa_params += list(self.sigreg_proj.parameters())
        params_groups.append(
            {"params": jepa_params, "lr": self.jepa_cfg.lr, "name": "jepa"}
        )

        online_params = list(self.online_encoder.parameters()) + list(
            self.online_quant_conv.parameters()
        )
        params_groups.append(
            {
                "params": online_params,
                "lr": self.jepa_cfg.get("encoder_lr", 1e-5),
                "name": "online_encoder",
            }
        )

        decoder_params = list(self.online_post_quant_conv.parameters()) + list(
            self.online_decoder.parameters()
        )
        params_groups.append(
            {
                "params": decoder_params,
                "lr": self.jepa_cfg.get("decoder_lr", 1e-4),
                "name": "online_decoder",
            }
        )

        optimizer = torch.optim.AdamW(
            params_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(optimizer=optimizer, **self.cfg.lr_scheduler),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return super()._should_include_in_checkpoint(key) or key.startswith(
            "spatial_predictor"
        )

    # -----------------------------------------------------------------
    # I-JEPA spatial prediction visualisation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_ijepa_predictions(
        self, gt_videos: Tensor, namespace: str, step: int
    ) -> None:
        """Save mask pattern + prediction error heatmap as PNG and W&B image."""
        from utils.distributed_utils import is_rank_zero

        if self.trainer.sanity_checking:
            return
        if not (is_rank_zero and self.logger):
            return
        if gt_videos is None:
            return

        video = gt_videos[:1, :1]  # single frame
        online_lat = self._encode_online(video)[:, 0]  # (1, C, H, W)
        target_lat = self._cached_target_latents[:1, :1][:, 0]

        online_p = self._patchify_ijepa(online_lat)
        target_p = self._patchify_ijepa(target_lat)

        ctx_mask, tgt_mask = self.mask_generator(1, online_lat.device)
        pred_p = self.spatial_predictor(online_p, ctx_mask, tgt_mask)

        gh, gw = self.ijepa_grid_h, self.ijepa_grid_w
        mask_grid = tgt_mask[0].float().reshape(gh, gw).cpu().numpy()
        error = (pred_p[0] - target_p[0]).norm(dim=-1).reshape(gh, gw).cpu().numpy()
        error_masked = error * mask_grid
        pred_norms = pred_p[0].norm(dim=-1).reshape(gh, gw).cpu().numpy()
        target_norms = target_p[0].norm(dim=-1).reshape(gh, gw).cpu().numpy()

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            log_dir = self.trainer.log_dir or "."
            save_dir = os.path.join(log_dir, "ijepa_vis", namespace)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"step_{step:06d}.png")

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(mask_grid, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Target mask")
            im1 = axes[1].imshow(pred_norms, cmap="viridis")
            axes[1].set_title("Pred norms")
            fig.colorbar(im1, ax=axes[1], fraction=0.046)
            im2 = axes[2].imshow(target_norms, cmap="viridis")
            axes[2].set_title("Target norms")
            fig.colorbar(im2, ax=axes[2], fraction=0.046)
            im3 = axes[3].imshow(error_masked, cmap="hot")
            axes[3].set_title("Error (targets)")
            fig.colorbar(im3, ax=axes[3], fraction=0.046)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        except Exception:
            save_path = None

        try:
            import wandb

            if save_path and os.path.exists(save_path):
                wandb.log(
                    {f"ijepa_vis/{namespace}/predictions": wandb.Image(save_path)},
                    step=self.global_step,
                    commit=False,
                )
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Hook: visualise I-JEPA during training
    # -----------------------------------------------------------------

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        _, _, _, gt_videos, _ = batch
        if gt_videos is None:
            return
        if (
            self.ijepa_vis_every > 0
            and self.global_step > 0
            and self.global_step % self.ijepa_vis_every == 0
            and self.trainer.world_size == 1
        ):
            self._log_ijepa_predictions(gt_videos, "ijepa_train", self.global_step)
