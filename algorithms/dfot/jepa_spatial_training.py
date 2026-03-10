"""
Standalone Spatial JEPA Training (no DiT co-training).

Inherits the spatial patch-token JEPA architecture from DFoTVideoJEPASpatial
but removes the DFoT diffusion backbone entirely, allowing focused iteration
on the spatial JEPA recipe (encoder, SpatialTemporalPredictor, decoder probe).

Same diagnostics and gradient paths as JEPATraining, but uses the spatial
predictor that preserves latent spatial structure via patch tokens.
"""

import os
from typing import Optional, Any, Dict, Tuple
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange
from transformers import get_scheduler

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_jepa_spatial import DFoTVideoJEPASpatial


class JEPASpatialTraining(DFoTVideoJEPASpatial):
    """
    Standalone spatial JEPA training without DiT.

    Inherits all spatial JEPA machinery (dual encoder, SpatialTemporalPredictor,
    EMA, decoder probe, loss functions, visualisation helpers) from
    DFoTVideoJEPASpatial and strips out everything related to the diffusion
    backbone.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Parent (DFoTVideoJEPASpatial -> DFoTVideoJEPA) sets automatic_optimization = False
        # for its two-phase backward.  We only have one loss so automatic is fine.
        self.automatic_optimization = True

        self._ref_image: Optional[Tensor] = None
        self._ref_mean: Optional[Tensor] = None
        self._ref_logvar: Optional[Tensor] = None

    # -----------------------------------------------------------------
    # Model building  (skip diffusion backbone)
    # -----------------------------------------------------------------

    def _build_model(self):
        """Build VAE + spatial JEPA components; skip diffusion backbone entirely."""
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        if self.is_latent_diffusion and self.is_latent_online:
            self._load_vae()

        self.diffusion_model = None

        self._build_jepa_model()

        self.register_buffer("_running_mean", self.data_mean.clone())
        self.register_buffer("_running_std", self.data_std.clone())
        self._stats_momentum = 0.01

    # -----------------------------------------------------------------
    # Batch preprocessing
    # -----------------------------------------------------------------

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Simplified preprocessing for JEPA-only training.
        Returns (gt_videos, actions_raw, masks).
        """
        gt_videos = batch["videos"]
        actions_raw = batch.get("conds", None)

        target_latents = self._encode_target_mode(gt_videos)
        self._cached_target_latents = target_latents

        if self.training:
            self._update_running_latent_stats(target_latents)

        if "masks" in batch:
            masks = batch["masks"]
        else:
            masks = torch.ones(
                target_latents.shape[:2], dtype=torch.bool, device=self.device
            )

        return gt_videos, actions_raw, masks

    @torch.no_grad()
    def _encode_target_mode(self, videos):
        """Encode with frozen target encoder -> deterministic (mode) latents."""
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        vae_bs = self.cfg.vae.batch_size
        chunks = []
        for start in range(0, x_normalized.shape[0], vae_bs):
            posterior = self.vae.encode(x_normalized[start : start + vae_bs])
            chunks.append(posterior.mode())
        mode = torch.cat(chunks, dim=0)
        return mode.reshape(B, T, *mode.shape[1:])

    # -----------------------------------------------------------------
    # Optimizers  (3 groups -- no diffusion)
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        params_groups = [
            {
                "params": (
                    list(self.predictor.parameters())
                    + list(self.action_encoder.parameters())
                ),
                "lr": self.jepa_cfg.lr,
                "name": "jepa",
            },
            {
                "params": (
                    list(self.online_encoder.parameters())
                    + list(self.online_quant_conv.parameters())
                ),
                "lr": self.jepa_cfg.get("encoder_lr", 1e-5),
                "name": "online_encoder",
            },
            {
                "params": (
                    list(self.online_post_quant_conv.parameters())
                    + list(self.online_decoder.parameters())
                ),
                "lr": self.jepa_cfg.get("decoder_lr", 1e-4),
                "name": "online_decoder",
            },
        ]

        optimizer = torch.optim.AdamW(
            params_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer, **self.cfg.lr_scheduler
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training"):
        gt_videos, actions_raw, masks = batch

        online_latents = self._encode_online(gt_videos)

        # ---- Decoder loss ----
        _zero = torch.tensor(0.0, device=self.device)
        decoder_loss = _zero
        decoder_log = {
            "decoder/mse_loss": _zero,
            "decoder/lpips_loss": _zero,
            "decoder/total_loss": _zero,
        }

        decoder_loss_weight = self.jepa_cfg.get("decoder_loss_weight", 0.0)
        has_decoder = gt_videos is not None and decoder_loss_weight > 0
        decoder_loss_every = self.jepa_cfg.get("decoder_loss_every", 1)
        recon_reg = self.jepa_cfg.get("recon_regularizer", False)
        if has_decoder and not recon_reg and decoder_loss_every > 1:
            if self.global_step % decoder_loss_every != 0:
                has_decoder = False

        if has_decoder:
            decoder_loss, decoder_log = self._compute_decoder_loss(
                gt_videos, online_latents
            )

        # ---- JEPA loss (spatial predictor from parent) ----
        jepa_loss, jepa_log = self._compute_jepa_loss(
            online_latents, actions_raw, masks
        )

        # ---- Total loss (Lightning handles backward / optim / scheduler) ----
        total_loss = (
            self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
        )

        # ---- Logging ----
        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss", total_loss.detach(),
                on_step=True, sync_dist=True,
            )
            self.log(
                f"{namespace}/jepa_loss", jepa_loss.detach(),
                on_step=True, sync_dist=True,
            )
            self.log(
                f"{namespace}/decoder_loss", decoder_loss.detach(),
                on_step=True, sync_dist=True,
            )
            for key, value in jepa_log.items():
                self.log(
                    f"{namespace}/{key}", value,
                    on_step=True, sync_dist=True,
                )
            for key, value in decoder_log.items():
                self.log(
                    f"{namespace}/{key}", value,
                    on_step=True, sync_dist=True,
                )

        # ---- Diagnostics ----
        self._maybe_log_drift(gt_videos)
        self._log_collapse_metrics(online_latents)

        return total_loss

    # -----------------------------------------------------------------
    # Post-batch hooks
    # -----------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        if (
            self.cfg.logging.grad_norm_freq
            and self.global_step % self.cfg.logging.grad_norm_freq == 0
        ):
            from lightning.pytorch.utilities import grad_norm

            norms = grad_norm(self.predictor, norm_type=2)
            self.log_dict(
                {f"grad_norm/predictor/{k}": v for k, v in norms.items()}
            )

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update + dual-mode visualisation."""
        if (self.global_step + 1) % self.ema_update_every == 0:
            self._ema_update_target()
            self._refresh_data_stats()

        vis_every = self.jepa_cfg.get("decoder_vis_every", 500)
        if not (
            self.jepa_cfg.get("decoder_vis_enabled", True)
            and self.global_step % vis_every == 0
            and self.global_step > 0
        ):
            return
        gt_videos, actions_raw, masks = batch
        if gt_videos is None:
            return

        self._log_decoder_reconstructions(
            gt_videos,
            namespace="decoder_vis_train",
            step=self.global_step,
        )
        self._log_jepa_predictions(
            gt_videos, actions_raw,
            namespace="jepa_pred_tf_train",
            step=self.global_step,
        )
        self._log_jepa_predictions_autoregressive(
            gt_videos, actions_raw,
            namespace="jepa_pred_ar_train",
            step=self.global_step,
        )

    # -----------------------------------------------------------------
    # Encoder drift monitoring
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _encode_online_posterior(self, images):
        """
        Full posterior (mean, logvar) from the online encoder.
        Used for KL-based drift tracking.
        """
        x = 2.0 * images - 1.0
        h = self.online_encoder(x)
        moments = self.online_quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    @torch.no_grad()
    def _maybe_log_drift(self, gt_videos):
        """
        Track encoder drift via KL divergence from step 0.

        On the first call a reference image is frozen and its encoder
        posterior stored.  Every ``drift_log_every`` steps the same image
        is re-encoded and KL + L2 distance are logged.
        """
        drift_every = self.jepa_cfg.get("drift_log_every", 100)
        if self.global_step % drift_every != 0:
            return

        if self._ref_image is None:
            self._ref_image = gt_videos[0, 0:1].clone()
            mean, logvar = self._encode_online_posterior(self._ref_image)
            self._ref_mean = mean.clone()
            self._ref_logvar = logvar.clone()
            return

        mean_k, logvar_k = self._encode_online_posterior(self._ref_image)

        kl = 0.5 * (
            self._ref_logvar - logvar_k
            + (logvar_k.exp() + (mean_k - self._ref_mean) ** 2)
            / self._ref_logvar.exp()
            - 1
        ).mean()

        l2_dist = (mean_k - self._ref_mean).pow(2).mean().sqrt()

        self.log(
            "diagnostics/encoder_drift_kl", kl,
            on_step=True, sync_dist=True,
        )
        self.log(
            "diagnostics/encoder_drift_l2", l2_dist,
            on_step=True, sync_dist=True,
        )

    # -----------------------------------------------------------------
    # Collapse detection
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_collapse_metrics(self, online_latents):
        """
        Monitor representation collapse indicators:
          - batch_variance: var across batch, averaged.  -> 0 means collapse.
          - channel_std:    per-channel std averaged.
          - feature_norm:   mean L2 norm of latent vectors.
        """
        collapse_every = self.jepa_cfg.get("collapse_log_every", 50)
        if self.global_step % collapse_every != 0:
            return

        batch_var = online_latents.var(dim=0).mean()
        channel_std = online_latents.std(dim=(0, 1, 3, 4)).mean()
        feature_norm = online_latents.flatten(2).norm(dim=-1).mean()

        self.log(
            "diagnostics/batch_variance", batch_var,
            on_step=True, sync_dist=True,
        )
        self.log(
            "diagnostics/channel_std", channel_std,
            on_step=True, sync_dist=True,
        )
        self.log(
            "diagnostics/feature_norm", feature_norm,
            on_step=True, sync_dist=True,
        )

    # -----------------------------------------------------------------
    # Autoregressive visualisation (spatial predictor interface)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_jepa_predictions_autoregressive(
        self, gt_videos, actions, namespace, step,
    ):
        """
        True autoregressive rollout for spatial predictor: only s_0 and a_0..a_{T-2}
        are given; the predictor feeds its own predictions back at every step.

        Spatial predictor expects (B, T, C, H, W) latents and (B, T, action_dim)
        actions. Layout: PNG grid with (T-1) rows x 2 cols = [predicted | gt].
        """
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if self.trainer.sanity_checking:
            return

        videos = gt_videos[:1]
        actions_batch = actions[:1]
        B, T = videos.shape[:2]

        if T < 2:
            return

        online_latents = self._encode_online(videos)
        action_embeds = self.action_encoder(actions_batch)

        # Spatial predictor: (B, T, C, H, W) latents, (B, T, action_embed_dim) actions
        current = online_latents[:, 0:1]
        pred_list = []
        for t in range(T - 1):
            pred_i = self.predictor(current, action_embeds[:, t : t + 1])
            pred_list.append(pred_i)
            current = pred_i.detach() if t < (T - 2) else pred_i

        pred_latents = torch.cat(pred_list, dim=1)
        pred_flat = rearrange(pred_latents, "b t c h w -> (b t) c h w")

        pred_recon = self._decode_online(pred_flat)
        pred_recon = (pred_recon.clamp(-1, 1) + 1) / 2
        pred_recon = pred_recon.reshape(
            1, T - 1, *pred_recon.shape[1:]
        ).float()

        gt_aligned = videos[:, 1:].float()

        if not is_rank_zero:
            return

        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(
            log_dir, "jepa_pred_vis_ar", namespace
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"step_{step:06d}.png"
        )
        frames = torch.stack(
            [pred_recon[0], gt_aligned[0]], dim=1
        ).reshape(-1, *videos.shape[2:])
        grid = torchvision.utils.make_grid(
            frames, nrow=2, padding=2, pad_value=0.5
        )
        torchvision.utils.save_image(grid, save_path)

        if self.logger:
            log_video(
                [pred_recon],
                gt_aligned,
                step=step,
                namespace=namespace,
                prefix="jepa_pred_ar",
                captions=["predicted (AR) | gt"],
                logger=self.logger.experiment,
            )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        """Validation: JEPA + decoder losses and visualisations."""
        gt_videos, actions_raw, masks = batch

        online_latents = self._encode_online(gt_videos)

        jepa_loss, jepa_log = self._compute_jepa_loss(
            online_latents, actions_raw, masks
        )

        decoder_loss_weight = self.jepa_cfg.get(
            "decoder_loss_weight", 0.0
        )
        decoder_loss = torch.tensor(0.0, device=self.device)
        if decoder_loss_weight > 0:
            decoder_loss, _ = self._compute_decoder_loss(
                gt_videos, online_latents
            )

        total_loss = (
            self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
        )

        self.log(
            f"{namespace}/loss", total_loss,
            on_epoch=True, sync_dist=True,
        )
        self.log(
            f"{namespace}/jepa_loss", jepa_loss,
            on_epoch=True, sync_dist=True,
        )
        self.log(
            f"{namespace}/decoder_loss", decoder_loss,
            on_epoch=True, sync_dist=True,
        )
        for key, value in jepa_log.items():
            self.log(
                f"{namespace}/{key}", value,
                on_epoch=True, sync_dist=True,
            )

        if batch_idx == 0:
            self._log_decoder_reconstructions(
                gt_videos,
                namespace="decoder_vis_val",
                step=self.global_step,
            )
            self._log_jepa_predictions(
                gt_videos, actions_raw,
                namespace="jepa_pred_tf_val",
                step=self.global_step,
            )
            self._log_jepa_predictions_autoregressive(
                gt_videos, actions_raw,
                namespace="jepa_pred_ar_val",
                step=self.global_step,
            )

    def on_validation_epoch_end(self, namespace="validation"):
        """No video-generation metrics to compute."""
        self.generator = None
        self.num_logged_videos = 0

    # -----------------------------------------------------------------
    # Checkpointing  (lightweight)
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key):
        if not self.jepa_cfg.get("save_weights", False):
            return False
        return (
            key.startswith("action_encoder")
            or key.startswith("predictor")
            or key.startswith("online_encoder")
            or key.startswith("online_quant_conv")
            or key.startswith("online_post_quant_conv")
            or key.startswith("online_decoder")
            or key.startswith("vae.encoder.")
            or key.startswith("vae.quant_conv.")
            or key.startswith("vae.post_quant_conv.")
            or key.startswith("vae.decoder.")
            or key.startswith("_running_")
        )

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]
        checkpoint["jepa_cfg"] = self.jepa_cfg

    def on_load_checkpoint(self, checkpoint):
        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            checkpoint.pop("loops", None)
            checkpoint.pop("callbacks", None)
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0

        expected_groups = 3
        ckpt_schedulers = checkpoint.get("lr_schedulers", None)
        if isinstance(ckpt_schedulers, list) and ckpt_schedulers:
            first = ckpt_schedulers[0]
            if isinstance(first, dict):
                sstate = first.get("state_dict", first)
                if isinstance(sstate, dict):
                    blrs = sstate.get("base_lrs", None)
                    if (
                        isinstance(blrs, list)
                        and len(blrs) != expected_groups
                    ):
                        rank_zero_print(
                            cyan(
                                "Ignoring incompatible lr_schedulers "
                                f"(ckpt={len(blrs)}, "
                                f"expected={expected_groups})."
                            )
                        )
                        checkpoint["lr_schedulers"] = []

        new_state_dict = {}
        ckpt_state = checkpoint.get("state_dict", {})
        loaded, skipped = 0, 0
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in ckpt_state
            ):
                new_state_dict[key] = ckpt_state[key]
                loaded += 1
            else:
                new_state_dict[key] = value
                if key in ckpt_state:
                    skipped += 1
        checkpoint["state_dict"] = new_state_dict

        if loaded:
            rank_zero_print(
                cyan(f"Loaded {loaded} keys from checkpoint")
            )
        if skipped:
            rank_zero_print(
                cyan(f"Skipped {skipped} checkpoint keys")
            )
