"""
Spatial JEPA predictor for DFoT.

This variant keeps latent spatial structure in the JEPA branch by predicting
next-frame latent maps as patch tokens, instead of flattening each frame into
a single state vector.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_jepa import DFoTVideoJEPA, ActionEncoder, FeedForward


class TemporalCausalAttention(nn.Module):
    """Causal self-attention over temporal dimension."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.heads)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class SpatialAttention(nn.Module):
    """Non-causal self-attention over spatial patch tokens."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, P, D)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b p (h d) -> b h p d", h=self.heads)
        k = rearrange(k, "b p (h d) -> b h p d", h=self.heads)
        v = rearrange(v, "b p (h d) -> b h p d", h=self.heads)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = rearrange(out, "b h p d -> b p (h d)")
        return self.to_out(out)


class FullSpatiotemporalCausalAttention(nn.Module):
    """Causal attention over flattened (time, patch) sequence."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D), N = T * P, time-major token order
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatialTemporalPredictor(nn.Module):
    """
    Predict next latent maps using spatial patch tokens.

    Input latents are patchified per frame. The model can run either:
    - factorized attention (temporal causal + spatial non-causal), or
    - full non-factorized causal attention over time-major flattened tokens.
    """

    def __init__(
        self,
        latent_channels: int,
        latent_h: int,
        latent_w: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 64,
        patch_size: int = 2,
        factorized_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError(
                f"Latent size ({latent_h}, {latent_w}) must be divisible by patch_size={patch_size}."
            )

        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patch_size = patch_size
        self.factorized_attention = factorized_attention

        self.grid_h = latent_h // patch_size
        self.grid_w = latent_w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.hidden_dim = hidden_dim

        # Conv patch embed/unembed (ViT-style tokenization in latent space)
        self.patch_embed = nn.Conv2d(
            latent_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.patch_unembed = nn.ConvTranspose2d(
            hidden_dim,
            latent_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        self.spatial_pos = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        mlp_dim = int(hidden_dim * mlp_ratio)
        if factorized_attention:
            self.temporal_layers = nn.ModuleList(
                [
                    TemporalCausalAttention(hidden_dim, heads, dim_head, dropout)
                    for _ in range(depth)
                ]
            )
            self.spatial_layers = nn.ModuleList(
                [
                    SpatialAttention(hidden_dim, heads, dim_head, dropout)
                    for _ in range(depth)
                ]
            )
            self.ff_layers = nn.ModuleList(
                [FeedForward(hidden_dim, mlp_dim, dropout) for _ in range(depth)]
            )
        else:
            self.full_layers = nn.ModuleList(
                [
                    FullSpatiotemporalCausalAttention(hidden_dim, heads, dim_head, dropout)
                    for _ in range(depth)
                ]
            )
            self.ff_layers = nn.ModuleList(
                [FeedForward(hidden_dim, mlp_dim, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def _patchify(self, z: Tensor) -> Tensor:
        # z: (B, T, C, H, W) -> (B, T, P, D)
        b, t, c, h, w = z.shape
        z2d = rearrange(z, "b t c h w -> (b t) c h w")
        x2d = self.patch_embed(z2d)  # (B*T, D, gh, gw)
        x = rearrange(x2d, "(b t) d gh gw -> b t (gh gw) d", b=b, t=t)
        return x

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        # tokens: (B, T, P, D) -> (B, T, C, H, W)
        b, t, p, d = tokens.shape
        x2d = rearrange(tokens, "b t (gh gw) d -> (b t) d gh gw", gh=self.grid_h, gw=self.grid_w)
        z2d = self.patch_unembed(x2d)  # (B*T, C, H, W)
        z = rearrange(z2d, "(b t) c h w -> b t c h w", b=b, t=t)
        return z

    def forward(self, latents: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            latents: (B, T, C, H, W)
            actions: (B, T, action_dim)
        Returns:
            pred_latents: (B, T, C, H, W), interpreted as next-state latents.
        """
        b, t = latents.shape[:2]
        if t > self.temporal_pos.shape[1]:
            raise ValueError(
                f"Sequence length {t} exceeds predictor temporal capacity {self.temporal_pos.shape[1]}."
            )

        x = self._patchify(latents)  # (B, T, P, D)

        # Add action context to every patch in the frame.
        act = self.action_proj(actions).unsqueeze(2)  # (B, T, 1, D)
        x = x + act
        x = x + self.spatial_pos.unsqueeze(1) + self.temporal_pos[:, :t].unsqueeze(2)
        x = self.dropout(x)

        if self.factorized_attention:
            for temp_attn, spat_attn, ff in zip(
                self.temporal_layers, self.spatial_layers, self.ff_layers
            ):
                xt = rearrange(x, "b t p d -> (b p) t d")
                xt = xt + temp_attn(xt)
                x = rearrange(xt, "(b p) t d -> b t p d", b=b, p=self.num_patches)

                xs = rearrange(x, "b t p d -> (b t) p d")
                xs = xs + spat_attn(xs)
                x = rearrange(xs, "(b t) p d -> b t p d", b=b, t=t)

                xf = rearrange(x, "b t p d -> (b t p) d")
                xf = xf + ff(xf)
                x = rearrange(xf, "(b t p) d -> b t p d", b=b, t=t, p=self.num_patches)
        else:
            x = rearrange(x, "b t p d -> b (t p) d")
            for full_attn, ff in zip(self.full_layers, self.ff_layers):
                x = x + full_attn(x)
                x = x + ff(x)
            x = rearrange(x, "b (t p) d -> b t p d", t=t, p=self.num_patches)

        x = self.norm(x)
        return self._unpatchify(x)


class DFoTVideoJEPASpatial(DFoTVideoJEPA):
    """DFoT JEPA variant with spatial patch-token predictor."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _build_jepa_model(self):
        jepa_cfg = self.jepa_cfg

        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h

        self.action_encoder = ActionEncoder(
            action_dim=self.external_cond_dim,
            embed_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.action_hidden_dim,
        )

        patch_size = int(jepa_cfg.get("spatial_patch_size", 2))
        factorized = bool(jepa_cfg.get("spatial_factorized_attention", True))

        self.predictor = SpatialTemporalPredictor(
            latent_channels=latent_channels,
            latent_h=latent_h,
            latent_w=latent_w,
            action_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.predictor_hidden_dim,
            depth=jepa_cfg.predictor_depth,
            heads=jepa_cfg.predictor_heads,
            dim_head=jepa_cfg.get("predictor_dim_head", 64),
            mlp_ratio=jepa_cfg.get("predictor_mlp_ratio", 4.0),
            max_seq_len=self.max_tokens + 1,
            patch_size=patch_size,
            factorized_attention=factorized,
            dropout=jepa_cfg.get("predictor_dropout", 0.1),
        )

        # Keep LPIPS/decoder behavior identical to parent implementation.
        from algorithms.vae.common.losses.lpips import LPIPS

        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        rank_zero_print(cyan(f"JEPA spatial predictor patch_size: {patch_size}"))
        rank_zero_print(cyan(f"JEPA spatial predictor factorized_attention: {factorized}"))
        rank_zero_print(cyan(f"JEPA Training mode: {self.jepa_training_mode}"))
        rank_zero_print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every}"))

    def _compute_jepa_loss(
        self,
        online_latents: Tensor,
        actions: Tensor,
        masks: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        JEPA loss using spatial latent predictor.
        """
        do_timing = timings is not None and _tick is not None
        b, t = online_latents.shape[:2]

        if t < 2:
            _zero = torch.tensor(0.0, device=online_latents.device)
            return _zero, {"jepa/pred_loss": _zero, "jepa/mse": _zero, "jepa/cos_sim": _zero}

        target_latents = self._cached_target_latents[:, :t]
        if self.jepa_cfg.get("normalize_jepa_latents", False):
            shape = [1, 1] + list(self.data_mean.shape)
            mean = self.data_mean.reshape(shape)
            std = self.data_std.reshape(shape).clamp_min(self._latent_norm_eps)
            online_latents = (online_latents - mean) / std
            target_latents = (target_latents - mean) / std

        if do_timing:
            t0 = _tick()
        action_embeds = self.action_encoder(actions)
        if do_timing and timings is not None:
            timings["jepa_action_encoder_ms"] = (_tick() - t0) * 1000

        lat_input = online_latents[:, :-1]
        act_input = action_embeds[:, :-1]
        lat_target = target_latents[:, 1:]

        if do_timing:
            t0 = _tick()
        if self.jepa_training_mode == "autoregressive":
            pred_list = []
            current = online_latents[:, 0:1]
            for i in range(t - 1):
                pred_i = self.predictor(current, act_input[:, i : i + 1])
                pred_list.append(pred_i)
                current = pred_i.detach() if i < (t - 2) else pred_i
            lat_pred = torch.cat(pred_list, dim=1)
        else:
            lat_pred = self.predictor(lat_input, act_input)
        if do_timing and timings is not None:
            timings["jepa_predictor_ms"] = (_tick() - t0) * 1000

        if do_timing:
            t0 = _tick()
        transition_masks = masks[:, :-1] & masks[:, 1:]

        pred_loss = F.smooth_l1_loss(lat_pred, lat_target, reduction="none")
        pred_loss = pred_loss.mean(dim=(2, 3, 4))  # (B, T-1)
        if transition_masks.sum() > 0:
            pred_loss = (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
        else:
            pred_loss = pred_loss.mean()
        if do_timing and timings is not None:
            timings["jepa_loss_compute_ms"] = (_tick() - t0) * 1000

        with torch.no_grad():
            if transition_masks.sum() > 0:
                lat_pred_masked = lat_pred[transition_masks]
                lat_target_masked = lat_target[transition_masks]
                mse = F.mse_loss(lat_pred_masked, lat_target_masked)
                cos_sim = F.cosine_similarity(
                    lat_pred_masked.flatten(start_dim=1),
                    lat_target_masked.flatten(start_dim=1),
                    dim=-1,
                ).mean()
            else:
                mse = torch.tensor(0.0, device=online_latents.device)
                cos_sim = torch.tensor(0.0, device=online_latents.device)

        return pred_loss, {
            "jepa/pred_loss": pred_loss,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
        }

    @torch.no_grad()
    def _log_jepa_predictions(
        self, gt_videos: Tensor, actions: Tensor, namespace: str, step: int
    ) -> None:
        """
        Spatial predictor visualization: decode predicted latent maps vs gt_next.
        """
        import os
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if self.trainer.sanity_checking:
            return

        videos = gt_videos[:1]
        actions_batch = actions[:1]
        b, t = videos.shape[:2]
        if t < 2:
            return

        online_latents = self._encode_online(videos)
        action_embeds = self.action_encoder(actions_batch)
        pred_latents = self.predictor(online_latents[:, :-1], action_embeds[:, :-1])
        pred_flat = rearrange(pred_latents, "b t c h w -> (b t) c h w")

        pred_recon = self._decode_online(pred_flat)
        pred_recon = ((pred_recon.clamp(-1, 1) + 1) / 2).reshape(
            b, t - 1, *pred_recon.shape[1:]
        )
        pred_recon = pred_recon.float()
        gt_aligned = videos[:, 1:].float()

        if not is_rank_zero:
            return

        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(log_dir, "jepa_pred_vis", namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")
        frames = torch.stack([pred_recon[0], gt_aligned[0]], dim=1).reshape(
            -1, *videos.shape[2:]
        )
        grid = torchvision.utils.make_grid(frames, nrow=2, padding=2, pad_value=0.5)
        torchvision.utils.save_image(grid, save_path)

        if self.logger:
            log_video(
                [pred_recon],
                gt_aligned,
                step=step,
                namespace=namespace,
                prefix="jepa_pred",
                captions=["predicted | gt"],
                logger=self.logger.experiment,
            )
