"""
Decoupled Predictive-Generative Latent for DFoT.

Splits encoder latent z = enc(img) into two complementary subspaces:
  - zp (predictive):  High-dimensional token representation shaped by
                       temporal JEPA + spatial I-JEPA + SigREG.
                       Acts as conditioning to DiT via adaLN.
                       Shape: (B, Dp, H, W) where Dp >> C (e.g. 128).
  - zg (generative):  Near-identity transform of z, shaped by
                       reconstruction loss (L1 + LPIPS).
                       Acts as the denoising target for DiT.
                       Shape: (B, C, H, W) same as original VAE latent.

Before decoding, zg is modulated by sg(zp) via FiLM (feature-wise
linear modulation) so the decoder benefits from dynamics context
without losing the decoupled training signal.

Both are stop-gradient on the DiT side.  The existing co-training EMA
setup carries over from DFoTVideoJEPA.

Gradient flow:
  JEPA losses (temporal + ijepa + sigreg)  -->  encoder + predictive_head + predictor
  Reconstruction loss (L1 + LPIPS)         -->  encoder + generative_head + fusion + decoder
  Decorrelation loss (cross-covariance)    -->  predictive_head + generative_head
  DFoT diffusion loss                      -->  diffusion_model only
"""

from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import get_scheduler

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_ijepa_sigreg import DFoTVideoIJEPASigREG
from .diffusion.decoupled_diffusion import DecoupledDiffusion


# =============================================================================
# Projection Heads
# =============================================================================


class PredictiveHead(nn.Module):
    """Projects full latent z to a rich predictive token space zp.

    Think of this as a learned patchifier: z (B, C, H, W) is projected
    to zp (B, Dp, H, W) where Dp is a high-dimensional token dim
    (e.g. 128).  The JEPA predictor can operate directly on zp tokens
    without an additional patchification bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        kernel = 3 if downsample else 1
        padding = 1 if downsample else 0
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(z)


class GenerativeHead(nn.Module):
    """Near-identity transform of z to zg.

    Keeps zg at the same channel count as the original VAE latent (C=4)
    so the pretrained decoder can consume it directly.  A light residual
    conv lets training fine-tune what the encoder routes here, but it
    starts as approximate identity so the decoder works out-of-the-box.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, z: Tensor) -> Tensor:
        return z + self.conv(self.norm(z))


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation: modulates zg with dynamics from zp.

    zp is spatially pooled per-position (if dims match) or globally pooled,
    then projected to (scale, shift) vectors that modulate zg channel-wise.
    This gives the decoder access to dynamics context from zp while keeping
    the gradient paths clean (zp is always sg during decoding).
    """

    def __init__(self, zp_channels: int, zg_channels: int):
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(zp_channels, zg_channels * 2),
            nn.SiLU(),
            nn.Linear(zg_channels * 2, zg_channels * 2),
        )
        nn.init.zeros_(self.pool_proj[-1].weight)
        nn.init.zeros_(self.pool_proj[-1].bias)

    def forward(self, zg: Tensor, zp: Tensor) -> Tensor:
        """
        Args:
            zg: (N, Cg, H, W) generative latent
            zp: (N, Dp, H, W) predictive latent (should be detached)
        Returns:
            (N, Cg, H, W) modulated zg
        """
        params = self.pool_proj(zp)  # (N, 2*Cg)
        scale, shift = params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return zg * (1 + scale) + shift


# =============================================================================
# Main Algorithm
# =============================================================================


class DFoTVideoDecoupled(DFoTVideoIJEPASigREG):
    """
    DFoT with decoupled predictive / generative latent subspaces.

    Inherits temporal JEPA + spatial I-JEPA + SigREG from parent.
    Adds:
      - Predictive head (z -> zp, high-dim) and generative head (z -> zg, near-identity)
      - FiLM fusion to modulate zg with sg(zp) before decoding
      - Decorrelation loss between zp and zg
      - zp conditioning in DiT via DecoupledDiffusion
      - zg as denoising target
    """

    def __init__(self, cfg: DictConfig):
        dcfg = cfg.get("decoupled", {})
        self._zp_channels = int(dcfg.get("zp_channels", 32))
        self._zg_channels = int(dcfg.get("zg_channels", 4))
        self._predictive_downsample = bool(dcfg.get("predictive_downsample", False))
        self._decorr_weight = float(dcfg.get("decorrelation_weight", 0.1))
        self._decorr_every = int(dcfg.get("decorrelation_every", 1))
        self._decorr_num_proj = int(dcfg.get("decorrelation_num_proj", 64))
        self._heads_lr = float(dcfg.get("heads_lr", 1e-4))
        self._fusion_weight = float(dcfg.get("fusion_weight", 1.0))
        self._diag_every = int(dcfg.get("diagnostics_every", 100))
        super().__init__(cfg)

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_model(self):
        """Build diffusion model with decoupled backbone, then JEPA components."""
        self.diffusion_model = torch.compile(
            DecoupledDiffusion(
                cfg=self.cfg.diffusion,
                backbone_cfg=self.cfg.backbone,
                x_shape=self.x_shape,
                max_tokens=self.max_tokens,
                external_cond_dim=self.external_cond_dim,
                encoder_cond_channels=self._zp_channels,
            ),
            disable=not self.cfg.compile,
        )

        # Keep base DFoT initialization parity (normalization stats + VAE modules)
        # because this class overrides DFoTVideoJEPA._build_model entirely.
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.is_latent_diffusion and self.is_latent_online:
            self._load_vae()
        else:
            self.vae = None

        self._build_jepa_model()

        self.register_buffer("_running_mean", self.data_mean.clone())
        self.register_buffer("_running_std", self.data_std.clone())
        self._stats_momentum = 0.01

        # Metrics were skipped because we override _build_model without calling super().
        self._build_metrics()

    def _build_jepa_model(self):
        """Build parent JEPA components + projection heads + rebuild for zp dims."""
        super()._build_jepa_model()

        latent_channels = self.x_shape[0]  # C=4

        self.predictive_head = PredictiveHead(
            in_channels=latent_channels,
            out_channels=self._zp_channels,
            downsample=self._predictive_downsample,
        )
        self.target_predictive_head = deepcopy(self.predictive_head)
        for p in self.target_predictive_head.parameters():
            p.requires_grad = False

        self.generative_head = GenerativeHead(channels=latent_channels)
        self.target_generative_head = deepcopy(self.generative_head)
        for p in self.target_generative_head.parameters():
            p.requires_grad = False

        self.film_fusion = FiLMFusion(
            zp_channels=self._zp_channels,
            zg_channels=latent_channels,
        )

        n_ph = sum(p.numel() for p in self.predictive_head.parameters())
        n_gh = sum(p.numel() for p in self.generative_head.parameters())
        n_film = sum(p.numel() for p in self.film_fusion.parameters())
        rank_zero_print(
            cyan(
                f"Decoupled heads: predictive {self._zp_channels}ch ({n_ph/1e3:.1f}K), "
                f"generative {latent_channels}ch ({n_gh/1e3:.1f}K), "
                f"FiLM fusion ({n_film/1e3:.1f}K)"
            )
        )
        rank_zero_print(cyan(f"Decorrelation weight: {self._decorr_weight}"))

        self._build_ijepa_for_zp()
        self._rebuild_sigreg_for_zp()
        self._rebuild_spatial_predictor_for_zp()

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_online_decoupled(self, videos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode with online encoder then split via heads.

        Returns:
            z_full: (B, T, C, H, W)  -- full encoder output
            zp:     (B, T, Dp, Hp, Wp) -- predictive subspace
            zg:     (B, T, C, H, W)  -- generative subspace (same shape as z)
        """
        z_full = self._encode_online(videos)
        B, T = z_full.shape[:2]

        z_flat = rearrange(z_full, "b t c h w -> (b t) c h w")
        zp_flat = self.predictive_head(z_flat)
        zg_flat = self.generative_head(z_flat)

        zp = rearrange(zp_flat, "(b t) c h w -> b t c h w", b=B, t=T)
        zg = rearrange(zg_flat, "(b t) c h w -> b t c h w", b=B, t=T)
        return z_full, zp, zg

    @torch.no_grad()
    def _encode_target_both(self, videos: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode with target encoder, returning sampled latents and mode latents.
        Also caches zp_target and zg_target through the target heads.
        """
        sampled, mode = super()._encode_target_both(videos)

        B, T = mode.shape[:2]
        mode_flat = rearrange(mode, "b t c h w -> (b t) c h w")
        zp_target = self.target_predictive_head(mode_flat)
        zg_target = self.target_generative_head(mode_flat)

        self._cached_target_zp = rearrange(zp_target, "(b t) c h w -> b t c h w", b=B, t=T)
        self._cached_target_zg = rearrange(zg_target, "(b t) c h w -> b t c h w", b=B, t=T)

        return sampled, mode

    # -----------------------------------------------------------------
    # Autoregressive zp prediction for DiT conditioning
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _generate_predicted_zp(
        self,
        zp_context: Tensor,
        action_embeds: Tensor,
        total_length: int,
    ) -> Tensor:
        """Autoregressively predict zp for all frames using the JEPA predictor.

        Args:
            zp_context: (B, Tc, Dp, H, W) encoder zp for context frames
            action_embeds: (B, T, A) action embeddings for the full sequence
            total_length: total number of frames to produce zp for
        Returns:
            zp_full: (B, total_length, Dp, H, W) with context zp kept as-is
                     and future frames predicted autoregressively
        """
        B, Tc = zp_context.shape[:2]
        if total_length <= Tc:
            return zp_context[:, :total_length]

        predicted = [zp_context]
        current = zp_context[:, -1:]  # (B, 1, Dp, H, W)

        for i in range(Tc - 1, total_length - 1):
            act_i = action_embeds[:, i:i + 1]  # (B, 1, A)
            pred_i = self.predictor(current, act_i)  # (B, 1, Dp, H, W)
            predicted.append(pred_i)
            current = pred_i

        return torch.cat(predicted, dim=1)

    def _get_ar_probability(self) -> float:
        """Annealing schedule for autoregressive probability in mixed JEPA training."""
        warmup = self.jepa_cfg.get("ar_warmup_steps", 5000)
        target = self.jepa_cfg.get("ar_target_prob", 0.5)
        return min(target, target * self.global_step / max(warmup, 1))

    # -----------------------------------------------------------------
    # FiLM-fused decode: zg modulated by sg(zp)
    # -----------------------------------------------------------------

    def _decode_online_fused(self, zg: Tensor, zp: Tensor) -> Tensor:
        """Decode zg after modulating it with dynamics info from zp.

        Args:
            zg: (N, C, H, W) generative latent
            zp: (N, Dp, Hp, Wp) predictive latent (will be detached)
        Returns:
            x_recon: (N, 3, H, W) reconstructed pixels in [-1, 1]
        """
        zg_fused = self.film_fusion(zg, zp.detach())
        return self._decode_online(zg_fused)

    # -----------------------------------------------------------------
    # Decorrelation loss (VICReg-style, spatial-aware)
    # -----------------------------------------------------------------

    def _compute_decorrelation_loss(
        self, zp: Tensor, zg: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Cross-covariance penalty between zp and zg on random spatial
        projections to encourage complementary information.
        """
        zp_flat = zp.flatten(0, 1).flatten(1)  # (N, Dp*Hp*Wp)
        zg_flat = zg.flatten(0, 1).flatten(1)  # (N, C*H*W)

        N = zp_flat.shape[0]
        zp_c = zp_flat - zp_flat.mean(dim=0, keepdim=True)
        zg_c = zg_flat - zg_flat.mean(dim=0, keepdim=True)

        zp_std = zp_c.std(dim=0).clamp_min(1e-4)
        zg_std = zg_c.std(dim=0).clamp_min(1e-4)
        zp_n = zp_c / zp_std
        zg_n = zg_c / zg_std

        num_proj = self._decorr_num_proj
        rp_zp = torch.randn(zp_n.shape[1], num_proj, device=zp_n.device, dtype=zp_n.dtype)
        rp_zp = rp_zp / rp_zp.norm(dim=0, keepdim=True)
        rp_zg = torch.randn(zg_n.shape[1], num_proj, device=zg_n.device, dtype=zg_n.dtype)
        rp_zg = rp_zg / rp_zg.norm(dim=0, keepdim=True)

        zp_proj = zp_n @ rp_zp  # (N, num_proj)
        zg_proj = zg_n @ rp_zg  # (N, num_proj)

        cross_cov = (zp_proj.T @ zg_proj) / N  # (num_proj, num_proj)
        loss = cross_cov.pow(2).sum() / (num_proj * num_proj)

        with torch.no_grad():
            max_corr = cross_cov.abs().max()

        return loss, {
            "decoupled/decorr_loss": loss.detach(),
            "decoupled/max_cross_corr": max_corr,
        }

    # -----------------------------------------------------------------
    # Batch preprocessing
    # -----------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor]]:
        gt_videos = batch.get("videos", None)
        actions_raw = batch.get("conds", None)

        if self.is_latent_diffusion and self.is_latent_online:
            xs_full, target_mode = self._encode_target_both(batch["videos"])
            self._cached_target_latents = target_mode

            self._cached_zp_for_dfot = self._cached_target_zp
            xs = self._cached_target_zg
        else:
            xs = batch.get("latents", batch["videos"])
            self._cached_target_latents = None
            self._cached_zp_for_dfot = None

        if self.training:
            self._update_running_latent_stats(xs)

        xs = self._normalize_x(xs)
        conditions = batch.get("conds", None)

        if "masks" in batch:
            masks = batch["masks"]
        else:
            masks = torch.ones(*xs.shape[:2], dtype=torch.bool, device=self.device)

        return xs, conditions, masks, gt_videos, actions_raw

    # -----------------------------------------------------------------
    # Decoder loss: L1 + LPIPS on zg fused with sg(zp)
    # -----------------------------------------------------------------

    def _compute_decoder_loss(
        self,
        videos: Tensor,
        online_zg: Tensor,
        online_zp: Optional[Tensor] = None,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Decoder loss using L1 + LPIPS (matching VAE training).

        zg is fused with sg(zp) via FiLM before decoding, so the decoder
        gets dynamics context while keeping gradient paths clean.
        """
        do_timing = timings is not None and _tick is not None

        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0

        recon_reg = self.jepa_cfg.get("recon_regularizer", False)

        zg_flat = rearrange(online_zg, "b t c h w -> (b t) c h w")
        if not recon_reg:
            zg_flat = zg_flat.detach()

        zp_flat = None
        if online_zp is not None:
            zp_flat = rearrange(online_zp, "b t c h w -> (b t) c h w").detach()

        decoder_bs = self.jepa_cfg.get("decoder_chunk_size", 4)
        use_amp = self.jepa_cfg.get("decoder_lpips_amp", True) and torch.cuda.is_available()
        lpips_every = self.jepa_cfg.get("decoder_lpips_every", 1)
        run_lpips = (lpips_every <= 1) or (self.global_step % lpips_every == 0)

        total = x_norm.shape[0]
        l1_losses: list = []
        lpips_losses: list = []
        t_decoder, t_l1, t_lpips = 0.0, 0.0, 0.0
        for start in range(0, total, decoder_bs):
            x_chunk = x_norm[start: start + decoder_bs]
            zg_chunk = zg_flat[start: start + decoder_bs]
            if do_timing:
                t0 = _tick()
            if zp_flat is not None:
                zp_chunk = zp_flat[start: start + decoder_bs]
                if recon_reg:
                    with torch.amp.autocast("cuda", enabled=False):
                        x_recon = self._decode_online_fused(zg_chunk.float(), zp_chunk.float())
                else:
                    x_recon = self._decode_online_fused(zg_chunk, zp_chunk)
            else:
                if recon_reg:
                    with torch.amp.autocast("cuda", enabled=False):
                        x_recon = self._decode_online(zg_chunk.float())
                else:
                    x_recon = self._decode_online(zg_chunk)
            if do_timing:
                t_decoder += _tick() - t0

            if do_timing:
                t0 = _tick()
            l1_losses.append(F.l1_loss(x_recon.float(), x_chunk.float()))
            if do_timing:
                t_l1 += _tick() - t0

            if run_lpips:
                if do_timing:
                    t0 = _tick()
                if use_amp and not recon_reg:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        lpips_losses.append(self.perceptual_loss(x_recon, x_chunk).mean())
                else:
                    lpips_losses.append(self.perceptual_loss(x_recon.float(), x_chunk.float()).mean())
                if do_timing:
                    t_lpips += _tick() - t0

        if do_timing and timings is not None:
            timings["decoder_forward_ms"] = t_decoder * 1000
            timings["decoder_l1_ms"] = t_l1 * 1000
            if run_lpips:
                timings["decoder_lpips_ms"] = t_lpips * 1000

        l1_loss = torch.stack(l1_losses).mean()
        lpips_loss = (
            torch.stack(lpips_losses).mean()
            if lpips_losses
            else torch.tensor(0.0, device=l1_loss.device)
        )

        rec_w = self.jepa_cfg.get("decoder_mse_weight", 1.0)
        lpips_w = self.jepa_cfg.get("decoder_lpips_weight", 1.0)
        decoder_loss = rec_w * l1_loss
        if run_lpips:
            decoder_loss = decoder_loss + lpips_w * lpips_loss

        log_dict = {
            "decoder/l1_loss": l1_loss.detach(),
            "decoder/lpips_loss": lpips_loss.detach(),
            "decoder/total_loss": decoder_loss.detach(),
            "decoder/lpips_active": torch.tensor(float(run_lpips), device=l1_loss.device),
            "decoder/recon_regularizer": torch.tensor(float(recon_reg), device=l1_loss.device),
        }
        return decoder_loss, log_dict

    # -----------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        xs, conditions, masks, gt_videos, actions_raw = batch
        is_train = self.training

        log_timing = self.jepa_cfg.get("log_step_timing", False)
        timing_freq = self.jepa_cfg.get("log_step_timing_freq", 10)
        do_timing = log_timing and batch_idx % timing_freq == 0

        def _tick() -> float:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        timings: Dict[str, float] = {}
        if do_timing:
            t_step_start = _tick()

        if is_train:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            opt.zero_grad()

        # ========== Phase 1: DFoT Loss with zp conditioning ==========
        if do_timing:
            t0 = _tick()

        noise_levels, masks_dfot = self._get_training_noise_levels(xs, masks)

        dit_zp_source = self.jepa_cfg.get("dit_zp_source", "predicted")
        if dit_zp_source == "predicted" and self._cached_zp_for_dfot is not None and actions_raw is not None:
            zp_context = self._cached_zp_for_dfot[:, :self.n_context_tokens].detach()
            action_embeds_for_zp = self.action_encoder(actions_raw).detach()
            with torch.no_grad():
                zp_cond = self._generate_predicted_zp(
                    zp_context, action_embeds_for_zp, total_length=xs.shape[1],
                )
        else:
            zp_cond = self._cached_zp_for_dfot
            if zp_cond is not None:
                zp_cond = zp_cond.detach()

        xs_pred, dfot_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            encoder_cond=zp_cond,
        )
        dfot_loss = self._reweight_loss(dfot_loss, masks_dfot)
        if do_timing:
            timings["phase1_dfot_forward_ms"] = (_tick() - t0) * 1000

        # ========== Phase 2 & 3: JEPA + Decoder + Decorrelation =========
        jepa_loss = torch.tensor(0.0, device=xs.device)
        decoder_loss = torch.tensor(0.0, device=xs.device)
        decorr_loss = torch.tensor(0.0, device=xs.device)
        _zero = torch.tensor(0.0, device=xs.device)
        jepa_log_dict: Dict[str, Tensor] = {
            "jepa/pred_loss": _zero, "jepa/mse": _zero, "jepa/cos_sim": _zero,
        }
        decoder_log_dict: Dict[str, Tensor] = {
            "decoder/l1_loss": _zero, "decoder/lpips_loss": _zero,
            "decoder/total_loss": _zero, "decoder/lpips_active": _zero,
            "decoder/recon_regularizer": _zero,
        }
        decorr_log_dict: Dict[str, Tensor] = {
            "decoupled/decorr_loss": _zero, "decoupled/max_cross_corr": _zero,
        }

        has_jepa = (
            gt_videos is not None
            and actions_raw is not None
            and self.jepa_loss_weight > 0
        )
        decoder_loss_weight = self.jepa_cfg.get("decoder_loss_weight", 0.0)
        has_decoder = gt_videos is not None and decoder_loss_weight > 0
        if not is_train:
            has_jepa = False
            has_decoder = False

        decoder_loss_every = self.jepa_cfg.get("decoder_loss_every", 1)
        recon_reg = self.jepa_cfg.get("recon_regularizer", False)
        if has_decoder and not recon_reg and decoder_loss_every > 1:
            if self.global_step % decoder_loss_every != 0:
                has_decoder = False

        run_decorr = (
            is_train
            and self._decorr_weight > 0
            and (self._decorr_every <= 1 or self.global_step % self._decorr_every == 0)
        )

        if has_jepa or has_decoder or run_decorr:
            if do_timing:
                t0 = _tick()

            z_full, online_zp, online_zg = self._encode_online_decoupled(gt_videos)

            if do_timing:
                timings["phase2_online_encoder_ms"] = (_tick() - t0) * 1000

            original_cached = self._cached_target_latents
            if self._cached_target_zp is not None:
                self._cached_target_latents = self._cached_target_zp

            if has_decoder:
                if do_timing:
                    t0 = _tick()
                decoder_loss, decoder_log_dict = self._compute_decoder_loss(
                    gt_videos, online_zg, online_zp,
                    timings=timings if do_timing else None,
                    _tick=_tick if do_timing else None,
                )
                if do_timing:
                    timings["phase2_decoder_loss_ms"] = (_tick() - t0) * 1000

            if has_jepa:
                if do_timing:
                    t0 = _tick()
                jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                    online_zp, actions_raw, masks,
                    timings=timings if do_timing else None,
                    _tick=_tick if do_timing else None,
                )
                if do_timing:
                    timings["phase2_jepa_loss_ms"] = (_tick() - t0) * 1000

            if run_decorr and has_jepa:
                decorr_loss, decorr_log_dict = self._compute_decorrelation_loss(
                    online_zp, online_zg
                )

            self._cached_target_latents = original_cached

        phase2_loss = (
            self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
            + self._decorr_weight * decorr_loss
        )
        phase2_needs_backward = is_train and phase2_loss.requires_grad

        # ========== Two-phase backward ==========
        if is_train:
            use_ddp = self.trainer.world_size > 1
            if do_timing:
                t0 = _tick()
            if use_ddp and phase2_needs_backward:
                with self.trainer.strategy.model.no_sync():
                    self.manual_backward(self.dfot_loss_weight * dfot_loss)
            else:
                self.manual_backward(self.dfot_loss_weight * dfot_loss)
            if do_timing:
                timings["phase1_dfot_backward_ms"] = (_tick() - t0) * 1000

        if phase2_needs_backward:
            if do_timing:
                t0 = _tick()
            self.manual_backward(phase2_loss)
            if do_timing:
                timings["phase2_backward_ms"] = (_tick() - t0) * 1000

        # ========== Per-group gradient clipping + optimizer step ==========
        if is_train:
            if do_timing:
                t0 = _tick()
            clip_val = self.trainer.gradient_clip_val
            if not clip_val:
                clip_val = 1.0

            raw_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)

            if scaler is not None:
                scaler.unscale_(raw_opt)

            if (
                self.cfg.logging.grad_norm_freq
                and self.global_step % self.cfg.logging.grad_norm_freq == 0
            ):
                from lightning.pytorch.utilities import grad_norm as _grad_norm
                for prefix, module in [
                    ("diffusion_model", self.diffusion_model),
                    ("predictor", self.predictor),
                    ("online_encoder", self.online_encoder),
                    ("predictive_head", self.predictive_head),
                    ("generative_head", self.generative_head),
                    ("film_fusion", self.film_fusion),
                ]:
                    norms = _grad_norm(module, norm_type=2)
                    self.log_dict({f"grad_norm/{prefix}/{k}": v for k, v in norms.items()})

            for group in raw_opt.param_groups:
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, clip_val)
            if do_timing:
                timings["optimizer_clip_grad_ms"] = (_tick() - t0) * 1000

            if do_timing:
                t0 = _tick()
            optim_progress = self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress
            optim_progress.increment_ready()
            if scaler is not None:
                scaler.step(raw_opt)
                scaler.update()
            else:
                raw_opt.step()
            optim_progress.increment_completed()
            if do_timing:
                timings["optimizer_step_ms"] = (_tick() - t0) * 1000
            if do_timing:
                t0 = _tick()
            sch.step()
            if do_timing:
                timings["optimizer_scheduler_ms"] = (_tick() - t0) * 1000

        if do_timing and timings:
            timings["step_total_ms"] = (_tick() - t_step_start) * 1000
            for key, val in timings.items():
                self.log(f"timing/{key}", val, on_step=True, sync_dist=is_train)

        # =============== Logging ===============
        total_loss = (
            self.dfot_loss_weight * dfot_loss
            + self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
            + self._decorr_weight * decorr_loss
        )
        if is_train and batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/dfot_loss", dfot_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/jepa_loss", jepa_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/decoder_loss", decoder_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/decorr_loss", decorr_loss.detach(), on_step=True, sync_dist=True)
            for key, value in {**jepa_log_dict, **decoder_log_dict, **decorr_log_dict}.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": total_loss.detach(),
            "dfot_loss": dfot_loss.detach(),
            "jepa_loss": jepa_loss.detach(),
            "xs_pred": xs_pred,
            "xs": xs,
        }

    # -----------------------------------------------------------------
    # EMA update: include projection heads + FiLM fusion
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target(self) -> None:
        super()._ema_update_target()

        decay = self.ema_decay
        for p_online, p_target in zip(
            self.predictive_head.parameters(),
            self.target_predictive_head.parameters(),
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        for p_online, p_target in zip(
            self.generative_head.parameters(),
            self.target_generative_head.parameters(),
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        params_groups = []

        params_groups.append({
            "params": list(self.diffusion_model.parameters()),
            "lr": self.cfg.lr,
            "name": "diffusion",
        })

        jepa_params = (
            list(self.predictor.parameters())
            + list(self.action_encoder.parameters())
            + list(self.spatial_predictor.parameters())
            + list(self.predictive_head.parameters())
        )
        if self.sigreg_proj is not None:
            jepa_params += list(self.sigreg_proj.parameters())
        params_groups.append({
            "params": jepa_params,
            "lr": self.jepa_cfg.lr,
            "name": "jepa",
        })

        online_params = (
            list(self.online_encoder.parameters())
            + list(self.online_quant_conv.parameters())
        )
        params_groups.append({
            "params": online_params,
            "lr": self.jepa_cfg.get("encoder_lr", 1e-5),
            "name": "online_encoder",
        })

        decoder_params = (
            list(self.online_post_quant_conv.parameters())
            + list(self.online_decoder.parameters())
            + list(self.generative_head.parameters())
            + list(self.film_fusion.parameters())
        )
        params_groups.append({
            "params": decoder_params,
            "lr": self.jepa_cfg.get("decoder_lr", 1e-4),
            "name": "online_decoder",
        })

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
    # Sampling
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _get_predicted_zp_for_inference(
        self,
        context_zg: Tensor,
        conditions: Optional[Tensor],
        total_length: int,
    ) -> Tensor:
        """Encode context frames to zp, then predict the rest autoregressively.

        Args:
            context_zg: (B, Tc, C, H, W) normalized zg context frames
            conditions: (B, T, ...) raw action conditions for the full sequence
            total_length: total frames to produce zp for
        Returns:
            zp_pred: (B, total_length, Dp, H, W) predicted zp
        """
        B, Tc = context_zg.shape[:2]
        ctx_unnorm = self._unnormalize_x(context_zg)
        ctx_flat = rearrange(ctx_unnorm, "b t c h w -> (b t) c h w")

        zp_context = self.predictive_head(ctx_flat)
        zp_context = rearrange(zp_context, "(b t) c h w -> b t c h w", b=B, t=Tc)

        if total_length <= Tc or conditions is None:
            if total_length <= Tc:
                return zp_context[:, :total_length]
            pad = zp_context[:, -1:].expand(-1, total_length - Tc, -1, -1, -1)
            return torch.cat([zp_context, pad], dim=1)

        action_embeds = self.action_encoder(conditions)
        return self._generate_predicted_zp(zp_context, action_embeds, total_length)

    def _predict_sequence(
        self,
        context,
        length=None,
        conditions=None,
        guidance_fn=None,
        reconstruction_guidance=0.0,
        history_guidance=None,
        sliding_context_len=None,
        return_all=False,
    ):
        """Override to use JEPA-predicted zp for sliding-window positions."""
        full_zp = self._cached_zp_for_dfot

        if full_zp is None or length is None or length <= self.max_tokens:
            if full_zp is not None and conditions is not None:
                zp_context = full_zp[:, :context.shape[1]].detach()
                action_embeds = self.action_encoder(conditions).detach()
                target_len = length if length is not None else self.max_tokens
                with torch.no_grad():
                    predicted_zp = self._generate_predicted_zp(
                        zp_context, action_embeds, total_length=target_len,
                    )
                self._cached_zp_for_dfot = predicted_zp
                result = super()._predict_sequence(
                    context=context, length=length, conditions=conditions,
                    guidance_fn=guidance_fn,
                    reconstruction_guidance=reconstruction_guidance,
                    history_guidance=history_guidance,
                    sliding_context_len=sliding_context_len,
                    return_all=return_all,
                )
                self._cached_zp_for_dfot = full_zp
                return result
            return super()._predict_sequence(
                context=context, length=length, conditions=conditions,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                history_guidance=history_guidance,
                sliding_context_len=sliding_context_len,
                return_all=return_all,
            )

        if sliding_context_len is None:
            sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len = context.shape[:2]
        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens
        x_shape = self.x_shape

        zp_context = full_zp[:, :gt_len].detach()
        action_embeds = self.action_encoder(conditions).detach() if conditions is not None else None
        with torch.no_grad():
            if action_embeds is not None:
                full_predicted_zp = self._generate_predicted_zp(
                    zp_context, action_embeds, total_length=length,
                )
            else:
                full_predicted_zp = full_zp

        curr_token = gt_len
        xs_pred = context
        record = None
        from tqdm import tqdm
        pbar = tqdm(
            total=self.sampling_timesteps
            * (
                1
                + (length - sliding_context_len - 1)
                // (self.max_tokens - sliding_context_len)
            ),
            initial=0,
            desc="Predicting with DFoT (decoupled)",
            leave=False,
        )
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported with sliding window.")
            c = min(sliding_context_len, curr_token)
            h = min(length - curr_token, self.max_tokens - c)
            h = min(h, chunk_size) if chunk_size > 0 else h
            l = c + h

            pad = torch.zeros((batch_size, h, *x_shape))
            ctx = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1)
            generated_len = curr_token - max(curr_token - c, gt_len)
            ctx_mask = torch.ones((batch_size, c), dtype=torch.long)
            if generated_len > 0:
                ctx_mask[:, -generated_len:] = 2
            pad_mask = torch.zeros((batch_size, h), dtype=torch.long)
            ctx_mask = torch.cat([ctx_mask, pad_mask.long()], 1).to(ctx.device)

            cond_len = l if self.use_causal_mask else self.max_tokens
            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c: curr_token - c + cond_len]

            window_start = curr_token - c
            window_end = curr_token - c + l
            T_zp = full_predicted_zp.shape[1]
            zp_start = min(window_start, T_zp)
            zp_end = min(window_end, T_zp)
            if zp_start < T_zp:
                zp_slice = full_predicted_zp[:, zp_start:zp_end]
                need = l - zp_slice.shape[1]
                if need > 0:
                    zp_pad = zp_slice[:, -1:].expand(-1, need, -1, -1, -1)
                    zp_slice = torch.cat([zp_slice, zp_pad], dim=1)
            else:
                zp_slice = full_predicted_zp[:, -1:].expand(-1, l, -1, -1, -1)

            self._cached_zp_for_dfot = zp_slice

            new_pred, record = self._sample_sequence(
                batch_size,
                length=l,
                context=ctx,
                context_mask=ctx_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                history_guidance=history_guidance,
                return_all=return_all,
                pbar=pbar,
            )
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
        pbar.close()

        self._cached_zp_for_dfot = full_zp
        return xs_pred, record

    def _sample_sequence(
        self, batch_size, length=None, context=None, context_mask=None,
        conditions=None, guidance_fn=None, reconstruction_guidance=0.0,
        history_guidance=None, return_all=False, pbar=None,
    ):
        encoder_cond = None
        if self._cached_zp_for_dfot is not None:
            enc = self._cached_zp_for_dfot.detach()
            horizon = (length if length is not None else self.max_tokens)
            if not self.use_causal_mask:
                horizon = self.max_tokens

            T_enc = enc.shape[1]
            if T_enc >= horizon:
                encoder_cond = enc[:, :horizon]
            else:
                pad = enc[:, -1:].expand(-1, horizon - T_enc, -1, -1, -1)
                encoder_cond = torch.cat([enc, pad], dim=1)

        self._current_encoder_cond = encoder_cond

        result = super()._sample_sequence(
            batch_size=batch_size, length=length, context=context,
            context_mask=context_mask, conditions=conditions,
            guidance_fn=guidance_fn,
            reconstruction_guidance=reconstruction_guidance,
            history_guidance=history_guidance,
            return_all=return_all, pbar=pbar,
        )
        self._current_encoder_cond = None
        return result

    def _do_sample_step(
        self,
        xs_pred,
        from_noise_levels,
        to_noise_levels,
        conditions,
        conditions_mask,
        guidance_fn,
        nfe,
    ):
        enc = self._current_encoder_cond
        if enc is not None and nfe > 1:
            enc = repeat(enc, "b ... -> (b nfe) ...", nfe=nfe)

        return self.diffusion_model.sample_step(
            xs_pred,
            from_noise_levels,
            to_noise_levels,
            self._process_conditions(
                (
                    repeat(
                        conditions,
                        "b ... -> (b nfe) ...",
                        nfe=nfe,
                    ).clone()
                    if conditions is not None
                    else None
                ),
                from_noise_levels,
            ),
            conditions_mask,
            guidance_fn=guidance_fn,
            encoder_cond=enc,
        )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        xs, conditions, masks, gt_videos = batch
        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

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

        if (
            is_rank_zero and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            n = min(self.logging.max_num_videos - self.num_logged_videos, gt_videos_vis.shape[0])
            log_video(
                recons[:n].float(), gt_videos_vis[:n].float(),
                step=self.global_step, namespace="denoising_vis",
                logger=self.logger.experiment, indent=self.num_logged_videos,
                captions="denoised | gt",
            )
        if self.trainer.world_size > 1:
            torch.distributed.barrier()

    # -----------------------------------------------------------------
    # Decoder visualization
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_decoder_reconstructions(
        self, gt_videos: Tensor, namespace: str, step: int
    ) -> None:
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if self.trainer.sanity_checking:
            return

        videos = gt_videos[:1]
        T = videos.shape[1]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0

        posterior = self.vae.encode(x_norm)
        z_target = posterior.mode()
        zg_target = self.target_generative_head(z_target)
        zp_target = self.target_predictive_head(z_target)
        zg_fused_target = self.film_fusion(zg_target, zp_target)
        recon_target = self.vae.decode(zg_fused_target)
        recon_target = ((recon_target.clamp(-1, 1) + 1) / 2).reshape(1, T, *recon_target.shape[1:])

        h = self.online_encoder(x_norm)
        moments = self.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        zg_online = self.generative_head(mean)
        zp_online = self.predictive_head(mean)
        recon_online = self._decode_online_fused(zg_online, zp_online)
        recon_online = ((recon_online.clamp(-1, 1) + 1) / 2).reshape(1, T, *recon_online.shape[1:])

        recon_target = recon_target.float()
        recon_online = recon_online.float()
        videos = videos.float()

        if not is_rank_zero:
            return

        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(log_dir, "decoder_vis", namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")
        frames = torch.stack(
            [recon_target[0], recon_online[0], videos[0]], dim=1
        ).reshape(-1, *videos.shape[2:])
        grid = torchvision.utils.make_grid(frames, nrow=3, padding=2, pad_value=0.5)
        torchvision.utils.save_image(grid, save_path)

        if self.logger:
            log_video(
                [recon_target, recon_online], videos,
                step=step, namespace=namespace, prefix="decoder_recon",
                captions=["target | online | gt"], logger=self.logger.experiment,
            )

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return super()._should_include_in_checkpoint(key) or any(
            key.startswith(p) for p in (
                "predictive_head", "generative_head",
                "target_predictive_head", "target_generative_head",
                "film_fusion",
            )
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["decoupled_cfg"] = {
            "zp_channels": self._zp_channels,
            "zg_channels": self._zg_channels,
            "predictive_downsample": self._predictive_downsample,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        ckpt_state = checkpoint.get("state_dict", {})
        head_keys = [
            k for k in self.state_dict().keys()
            if any(k.startswith(p) for p in (
                "predictive_head", "generative_head",
                "target_predictive_head", "target_generative_head",
                "film_fusion",
            ))
        ]
        loaded = [k for k in head_keys if k in ckpt_state]
        new = [k for k in head_keys if k not in ckpt_state]
        if loaded:
            rank_zero_print(cyan(f"Loaded head weights: {len(loaded)} parameters"))
        if new:
            rank_zero_print(cyan(f"Randomly initialized head weights: {len(new)} parameters"))

    # -----------------------------------------------------------------
    # I-JEPA grid override for zp spatial dims
    # -----------------------------------------------------------------

    def _build_ijepa_for_zp(self):
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h

        if self._predictive_downsample:
            zp_h, zp_w = latent_h // 2, latent_w // 2
        else:
            zp_h, zp_w = latent_h, latent_w

        ps = self.ijepa_patch_size
        if zp_h % ps != 0 or zp_w % ps != 0:
            raise ValueError(
                f"zp spatial size ({zp_h}, {zp_w}) not divisible by ijepa_patch_size={ps}"
            )

        self.ijepa_grid_h = zp_h // ps
        self.ijepa_grid_w = zp_w // ps
        num_patches = self.ijepa_grid_h * self.ijepa_grid_w
        self.ijepa_patch_dim = self._zp_channels * ps * ps

        from .dfot_video_ijepa_sigreg import BlockMaskGenerator, SpatialIJEPAPredictor

        self.mask_generator = BlockMaskGenerator(
            grid_h=self.ijepa_grid_h, grid_w=self.ijepa_grid_w,
            num_targets=self.ijepa_num_target_blocks,
            target_scale_range=self.ijepa_target_scale,
            target_aspect_ratio_range=self.ijepa_target_aspect,
            min_context_ratio=self.ijepa_min_context,
        )
        self.spatial_predictor = SpatialIJEPAPredictor(
            patch_dim=self.ijepa_patch_dim, hidden_dim=self.ijepa_hidden_dim,
            depth=self.ijepa_depth, heads=self.ijepa_heads,
            num_patches=num_patches, dropout=self.ijepa_dropout,
        )
        rank_zero_print(
            cyan(f"I-JEPA (zp): grid={self.ijepa_grid_h}x{self.ijepa_grid_w}, "
                 f"patch_dim={self.ijepa_patch_dim}")
        )

    def _rebuild_sigreg_for_zp(self):
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        if self._predictive_downsample:
            zp_h, zp_w = latent_h // 2, latent_w // 2
        else:
            zp_h, zp_w = latent_h, latent_w

        zp_state_dim = self._zp_channels * zp_h * zp_w
        self.state_dim = zp_state_dim

        if self.sigreg_proj_dim > 0:
            self.sigreg_proj = nn.Sequential(
                nn.Linear(zp_state_dim, self.sigreg_proj_dim),
                nn.LayerNorm(self.sigreg_proj_dim),
                nn.GELU(),
                nn.Linear(self.sigreg_proj_dim, self.sigreg_proj_dim),
            )
            rank_zero_print(
                cyan(f"SigREG projection rebuilt for zp: {zp_state_dim} -> {self.sigreg_proj_dim}")
            )

    def _rebuild_spatial_predictor_for_zp(self):
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        if self._predictive_downsample:
            zp_h, zp_w = latent_h // 2, latent_w // 2
        else:
            zp_h, zp_w = latent_h, latent_w

        from .dfot_video_jepa_spatial import SpatialTemporalPredictor
        patch_size = int(self.jepa_cfg.get("spatial_patch_size", 2))
        factorized = bool(self.jepa_cfg.get("spatial_factorized_attention", True))

        self.predictor = SpatialTemporalPredictor(
            latent_channels=self._zp_channels,
            latent_h=zp_h, latent_w=zp_w,
            action_dim=self.jepa_cfg.action_embed_dim,
            hidden_dim=self.jepa_cfg.predictor_hidden_dim,
            depth=self.jepa_cfg.predictor_depth,
            heads=self.jepa_cfg.predictor_heads,
            dim_head=self.jepa_cfg.get("predictor_dim_head", 64),
            mlp_ratio=self.jepa_cfg.get("predictor_mlp_ratio", 4.0),
            max_seq_len=self.max_tokens + 1,
            patch_size=patch_size,
            factorized_attention=factorized,
            dropout=self.jepa_cfg.get("predictor_dropout", 0.1),
        )
        n_pred = sum(p.numel() for p in self.predictor.parameters())
        rank_zero_print(
            cyan(f"Spatial predictor rebuilt for zp: {self._zp_channels}ch, "
                 f"{zp_h}x{zp_w}, params={n_pred/1e6:.1f}M")
        )

    # -----------------------------------------------------------------
    # JEPA loss: fully overridden for zp
    # -----------------------------------------------------------------

    def _compute_jepa_loss(
        self, online_latents: Tensor, actions: Tensor, masks: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        do_timing = timings is not None and _tick is not None
        b, t = online_latents.shape[:2]
        if t < 2:
            zero = torch.tensor(0.0, device=online_latents.device)
            return zero, {
                "jepa/pred_loss": zero, "jepa/sigreg_loss_encoder": zero,
                "jepa/sigreg_loss_predictor": zero, "jepa/sigreg_loss": zero,
                "jepa/mse": zero, "jepa/cos_sim": zero,
            }

        sigreg_source = online_latents
        target_latents = self._cached_target_latents[:, :t]

        if self.jepa_cfg.get("normalize_jepa_latents", False):
            zp_mean = online_latents.mean(dim=(0, 1, 3, 4), keepdim=True)
            zp_std = online_latents.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(self._latent_norm_eps)
            online_latents = (online_latents - zp_mean) / zp_std
            target_latents = (target_latents - zp_mean) / zp_std

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

        use_ar = False
        if self.jepa_training_mode == "autoregressive":
            use_ar = True
        elif self.jepa_training_mode == "mixed":
            ar_prob = self._get_ar_probability()
            use_ar = random.random() < ar_prob

        if use_ar:
            pred_list = []
            current = online_latents[:, 0:1].detach()
            for i in range(t - 1):
                pred_i = self.predictor(current, act_input[:, i: i + 1])
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
        pred_loss = pred_loss.mean(dim=tuple(range(2, pred_loss.ndim)))
        if transition_masks.sum() > 0:
            pred_loss = (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
        else:
            pred_loss = pred_loss.mean()
        if do_timing and timings is not None:
            timings["jepa_loss_compute_ms"] = (_tick() - t0) * 1000

        states_flat = sigreg_source.reshape(b * t, -1)
        with torch.amp.autocast("cuda", enabled=False):
            sigreg_input = (
                states_flat.float()
                if self.sigreg_proj is None
                else self.sigreg_proj(states_flat.float())
            )
            sigreg_loss_encoder = self.sigreg(sigreg_input)

        sigreg_loss_predictor = torch.tensor(0.0, device=online_latents.device)
        sigreg_loss_total = sigreg_loss_encoder

        total_loss = (
            self.prediction_loss_weight * pred_loss
            + self.sigreg_loss_weight * sigreg_loss_total
        )

        ijepa_loss, ijepa_log = self._compute_ijepa_loss(
            sigreg_source, self._cached_target_latents[:, :t], masks,
        )
        total_loss = total_loss + self.ijepa_loss_weight * ijepa_loss

        with torch.no_grad():
            if transition_masks.sum() > 0:
                lat_pred_m = lat_pred[transition_masks]
                lat_target_m = lat_target[transition_masks]
                mse = F.mse_loss(lat_pred_m, lat_target_m)
                cos_sim = F.cosine_similarity(
                    lat_pred_m.flatten(1), lat_target_m.flatten(1), dim=-1
                ).mean()
            else:
                mse = torch.tensor(0.0, device=online_latents.device)
                cos_sim = torch.tensor(0.0, device=online_latents.device)

        _dev = online_latents.device
        log_dict = {
            "jepa/pred_loss": pred_loss, "jepa/sigreg_loss_encoder": sigreg_loss_encoder,
            "jepa/sigreg_loss_predictor": sigreg_loss_predictor,
            "jepa/sigreg_loss": sigreg_loss_total, "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
            "jepa/weighted_pred_loss": self.prediction_loss_weight * pred_loss,
            "jepa/weighted_sigreg_loss": self.sigreg_loss_weight * sigreg_loss_total,
            "jepa/ar_probability": torch.tensor(self._get_ar_probability(), device=_dev),
            "jepa/used_autoregressive": torch.tensor(float(use_ar), device=_dev),
        }
        log_dict.update(ijepa_log)
        log_dict["ijepa/weighted_spatial_loss"] = self.ijepa_loss_weight * ijepa_loss

        return total_loss, log_dict

    # -----------------------------------------------------------------
    # Diagnostics: log everything needed to verify decoupling works
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_decoupled_diagnostics(
        self, gt_videos: Tensor, actions_raw: Tensor, namespace: str = "training"
    ) -> None:
        """Comprehensive diagnostics to verify the decoupled pipeline."""
        if gt_videos is None:
            return

        videos = gt_videos[:1]
        B, T = videos.shape[:2]
        if T < 2:
            return

        z_full, online_zp, online_zg = self._encode_online_decoupled(videos)

        # --- 1. Shape & norm diagnostics ---
        self.log("diag/z_full_norm", z_full.norm(dim=(2, 3, 4)).mean(), on_step=True, sync_dist=True)
        self.log("diag/zp_norm", online_zp.norm(dim=(2, 3, 4)).mean(), on_step=True, sync_dist=True)
        self.log("diag/zg_norm", online_zg.norm(dim=(2, 3, 4)).mean(), on_step=True, sync_dist=True)
        self.log("diag/zp_channels", float(online_zp.shape[2]), on_step=True, sync_dist=True)
        self.log("diag/zg_channels", float(online_zg.shape[2]), on_step=True, sync_dist=True)

        # --- 2. zg vs z distance (should be small since near-identity head) ---
        zg_z_diff = (online_zg - z_full).abs().mean()
        self.log("diag/zg_z_l1_diff", zg_z_diff, on_step=True, sync_dist=True)

        zg_flat = online_zg.flatten(2)
        z_flat = z_full.flatten(2)
        zg_z_cos = F.cosine_similarity(zg_flat, z_flat, dim=-1).mean()
        self.log("diag/zg_z_cosine_sim", zg_z_cos, on_step=True, sync_dist=True)

        # --- 3. FiLM fusion effect ---
        zg_2d = rearrange(online_zg, "b t c h w -> (b t) c h w")
        zp_2d = rearrange(online_zp, "b t c h w -> (b t) c h w")
        zg_fused = self.film_fusion(zg_2d, zp_2d.detach())
        film_diff = (zg_fused - zg_2d).abs().mean()
        self.log("diag/film_diff_l1", film_diff, on_step=True, sync_dist=True)

        film_params = self.film_fusion.pool_proj(zp_2d.detach())
        scale, shift = film_params.chunk(2, dim=1)
        self.log("diag/film_scale_mean", scale.mean(), on_step=True, sync_dist=True)
        self.log("diag/film_scale_std", scale.std(), on_step=True, sync_dist=True)
        self.log("diag/film_shift_mean", shift.mean(), on_step=True, sync_dist=True)
        self.log("diag/film_shift_std", shift.std(), on_step=True, sync_dist=True)

        # --- 4. Recon quality: with FiLM vs without FiLM ---
        x_flat = rearrange(videos[:1, :1], "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0
        zg_single = zg_2d[:1]
        zp_single = zp_2d[:1]

        recon_with_film = self._decode_online_fused(zg_single, zp_single)
        recon_without_film = self._decode_online(zg_single)

        l1_with = F.l1_loss(recon_with_film.float(), x_norm.float())
        l1_without = F.l1_loss(recon_without_film.float(), x_norm.float())
        self.log("diag/recon_l1_with_film", l1_with, on_step=True, sync_dist=True)
        self.log("diag/recon_l1_without_film", l1_without, on_step=True, sync_dist=True)
        self.log("diag/film_improvement", l1_without - l1_with, on_step=True, sync_dist=True)

        # --- 5. Head weight norms (track learning) ---
        ph_norm = sum(p.norm().item() ** 2 for p in self.predictive_head.parameters()) ** 0.5
        gh_norm = sum(p.norm().item() ** 2 for p in self.generative_head.parameters()) ** 0.5
        film_norm = sum(p.norm().item() ** 2 for p in self.film_fusion.parameters()) ** 0.5
        self.log("diag/predictive_head_weight_norm", ph_norm, on_step=True, sync_dist=True)
        self.log("diag/generative_head_weight_norm", gh_norm, on_step=True, sync_dist=True)
        self.log("diag/film_fusion_weight_norm", film_norm, on_step=True, sync_dist=True)

        # --- 6. Channel variance (collapse detection) ---
        zp_ch_var = online_zp.var(dim=(0, 1, 3, 4)).mean()
        zg_ch_var = online_zg.var(dim=(0, 1, 3, 4)).mean()
        self.log("diag/zp_channel_variance", zp_ch_var, on_step=True, sync_dist=True)
        self.log("diag/zg_channel_variance", zg_ch_var, on_step=True, sync_dist=True)

        # --- 7. Batch variance (another collapse indicator) ---
        if B > 1:
            zp_batch_var = online_zp.var(dim=0).mean()
            zg_batch_var = online_zg.var(dim=0).mean()
            self.log("diag/zp_batch_variance", zp_batch_var, on_step=True, sync_dist=True)
            self.log("diag/zg_batch_variance", zg_batch_var, on_step=True, sync_dist=True)

    # -----------------------------------------------------------------
    # SigREG embedding graphs: use zp instead of raw encoder output
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_sigreg_embedding_graphs(
        self, gt_videos: Tensor, actions: Tensor, namespace: str = "training"
    ) -> None:
        if self.trainer.sanity_checking:
            return
        if gt_videos is None or actions is None:
            return

        num_videos = min(self.sigreg_graph_num_videos, gt_videos.shape[0])
        if num_videos <= 0:
            return
        videos = gt_videos[:num_videos]
        acts = actions[:num_videos]
        _, t = videos.shape[:2]
        if t < 2:
            return

        _, online_zp, _ = self._encode_online_decoupled(videos)
        action_embeds = self.action_encoder(acts)
        pred_latents = self.predictor(online_zp[:, :-1], action_embeds[:, :-1])

        emb_online = online_zp[:, :-1]
        emb_pred = pred_latents
        self._log_embedding_stats(emb_online, namespace, "online_latents")
        self._log_embedding_stats(emb_pred, namespace, "pred_latents")

        if self.sigreg_proj is not None:
            bo, to = emb_online.shape[:2]
            bp, tp = emb_pred.shape[:2]
            emb_online_proj = self.sigreg_proj(
                emb_online.reshape(-1, self.state_dim)
            ).reshape(bo, to, self.sigreg_proj_dim, 1, 1)
            emb_pred_proj = self.sigreg_proj(
                emb_pred.reshape(-1, self.state_dim)
            ).reshape(bp, tp, self.sigreg_proj_dim, 1, 1)
            self._log_embedding_stats(emb_online_proj, namespace, "online_proj", max_channels=8)
            self._log_embedding_stats(emb_pred_proj, namespace, "pred_proj", max_channels=8)

    # -----------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        from .dfot_video import DFoTVideo
        DFoTVideo.on_train_batch_end(self, outputs, batch, batch_idx)

        if (self.global_step + 1) % self.ema_update_every == 0:
            self._ema_update_target()
            self._refresh_data_stats()

        _, _, _, gt_videos, actions_raw = batch

        if (
            self._diag_every > 0
            and self.global_step > 0
            and self.global_step % self._diag_every == 0
            and gt_videos is not None
        ):
            self._log_decoupled_diagnostics(gt_videos, actions_raw)

        decoder_vis_every = self.jepa_cfg.get("decoder_vis_every", 50)
        if (
            self.trainer.world_size == 1
            and self.jepa_cfg.get("decoder_vis_enabled", True)
            and self.global_step % decoder_vis_every == 0
        ):
            if gt_videos is not None:
                self._log_decoder_reconstructions(
                    gt_videos, namespace="decoder_vis_train", step=self.global_step
                )
