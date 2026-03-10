"""
Spatial JEPA + SigREG variant for DFoT.

This variant starts from DFoTVideoJEPASpatial and adds:
- SigREG regularization on online-encoder latent embeddings.
- Optional SigREG projection MLP.
- SigREG on predictor outputs to stabilize latent geometry.

Intended defaults are provided via config:
- smaller ViT predictor depth (2)
- larger latent patch size (4)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from transformers import get_scheduler

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_jepa_spatial import DFoTVideoJEPASpatial

try:
    import lejepa
    USE_LEJEPA = True
except ImportError:
    from .sigreg import SigREG
    USE_LEJEPA = False


class DFoTVideoJEPASpatialSigREG(DFoTVideoJEPASpatial):
    """DFoT spatial JEPA variant with SigREG latent regularization."""

    def __init__(self, cfg: DictConfig):
        self.prediction_loss_weight = cfg.jepa.get("prediction_loss_weight", 1.0)
        self.sigreg_loss_weight = cfg.jepa.get("sigreg_loss_weight", 1.0)
        self.sigreg_num_slices = int(cfg.jepa.get("sigreg_num_slices", 1024))
        self.sigreg_proj_dim = int(cfg.jepa.get("sigreg_proj_dim", 256))
        self.sigreg_graph_every = int(cfg.jepa.get("sigreg_graph_every", 200))
        self.sigreg_graph_num_videos = int(cfg.jepa.get("sigreg_graph_num_videos", 1))
        self.sigreg_graph_max_points = int(cfg.jepa.get("sigreg_graph_max_points", 20000))
        super().__init__(cfg)
        self._ref_image: Optional[Tensor] = None
        self._ref_mean: Optional[Tensor] = None
        self._ref_logvar: Optional[Tensor] = None

    def _build_jepa_model(self):
        super()._build_jepa_model()

        latent_channels = self.x_shape[0]
        latent_h = self.x_shape[1]
        latent_w = self.x_shape[2] if len(self.x_shape) > 2 else latent_h
        self.state_dim = latent_channels * latent_h * latent_w

        if USE_LEJEPA:
            try:
                univariate_test = lejepa.univariate.EppsPulley()
                self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
                    univariate_test=univariate_test,
                    num_slices=self.sigreg_num_slices,
                )
                rank_zero_print(cyan("Using official LeJEPA SigREG implementation"))
            except Exception as e:
                rank_zero_print(cyan(f"LeJEPA init failed ({e}), using custom SigREG"))
                self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
        else:
            self.sigreg = SigREG(num_slices=self.sigreg_num_slices)
            rank_zero_print(cyan("Using custom SigREG implementation (LeJEPA not installed)"))

        if self.sigreg_proj_dim > 0:
            self.sigreg_proj = nn.Sequential(
                nn.Linear(self.state_dim, self.sigreg_proj_dim),
                nn.LayerNorm(self.sigreg_proj_dim),
                nn.GELU(),
                nn.Linear(self.sigreg_proj_dim, self.sigreg_proj_dim),
            )
            rank_zero_print(
                cyan(f"JEPA SigREG projection: {self.state_dim} -> {self.sigreg_proj_dim}")
            )
        else:
            self.sigreg_proj = None

        rank_zero_print(cyan(f"JEPA SigREG weight: {self.sigreg_loss_weight}"))
        rank_zero_print(cyan(f"JEPA SigREG slices: {self.sigreg_num_slices}"))
        rank_zero_print(cyan(f"JEPA prediction loss weight: {self.prediction_loss_weight}"))

    def configure_optimizers(self):
        """Same as parent, but include SigREG projection parameters in JEPA group."""
        params_groups = []
        params_groups.append(
            {
                "params": list(self.diffusion_model.parameters()),
                "lr": self.cfg.lr,
                "name": "diffusion",
            }
        )

        jepa_params = list(self.predictor.parameters()) + list(self.action_encoder.parameters())
        if self.sigreg_proj is not None:
            jepa_params += list(self.sigreg_proj.parameters())
        params_groups.append({"params": jepa_params, "lr": self.jepa_cfg.lr, "name": "jepa"})

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

    def _compute_jepa_loss(
        self,
        online_latents: Tensor,
        actions: Tensor,
        masks: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        do_timing = timings is not None and _tick is not None
        b, t = online_latents.shape[:2]
        if t < 2:
            zero = torch.tensor(0.0, device=online_latents.device)
            return zero, {
                "jepa/pred_loss": zero,
                "jepa/sigreg_loss_encoder": zero,
                "jepa/sigreg_loss_predictor": zero,
                "jepa/sigreg_loss": zero,
                "jepa/mse": zero,
                "jepa/cos_sim": zero,
            }

        sigreg_source_online = online_latents
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

        # SigREG on encoder embeddings (pre-normalization latents).
        # Run in float32 to prevent float16 overflow in the statistical test.
        states_flat = sigreg_source_online.reshape(b * t, -1)
        with torch.amp.autocast("cuda", enabled=False):
            sigreg_input_enc = (
                states_flat.float()
                if self.sigreg_proj is None
                else self.sigreg_proj(states_flat.float())
            )
            sigreg_loss_encoder = self.sigreg(sigreg_input_enc)

        sigreg_loss_predictor = torch.tensor(0.0, device=online_latents.device)
        sigreg_loss_total = sigreg_loss_encoder

        total_loss = (
            self.prediction_loss_weight * pred_loss
            + self.sigreg_loss_weight * sigreg_loss_total
        )

        with torch.no_grad():
            states_pred = lat_pred.reshape(b, t - 1, -1)
            states_target = lat_target.reshape(b, t - 1, -1)
            states_input = online_latents[:, :-1].reshape(b, t - 1, -1)

            states_pred_norm = F.normalize(states_pred, dim=-1)
            states_target_norm = F.normalize(states_target, dim=-1)
            copy_pred_norm = F.normalize(states_input, dim=-1)

            copy_loss = F.mse_loss(
                copy_pred_norm, states_target_norm, reduction="none"
            ).mean(dim=-1)
            if transition_masks.sum() > 0:
                copy_loss = (copy_loss * transition_masks.float()).sum() / transition_masks.sum()
            else:
                copy_loss = copy_loss.mean()

            pred_std_across_batch = states_pred.std(dim=0).mean()
            encoder_norm_mean = states_flat.norm(dim=-1).mean()
            pred_norm_mean = states_pred.norm(dim=-1).mean()
            target_norm_mean = states_target.norm(dim=-1).mean()

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

        return total_loss, {
            "jepa/pred_loss": pred_loss,
            "jepa/copy_baseline_loss": copy_loss,
            "jepa/pred_vs_copy_ratio": pred_loss / (copy_loss + 1e-8),
            "jepa/sigreg_loss_encoder": sigreg_loss_encoder,
            "jepa/sigreg_loss_predictor": sigreg_loss_predictor,
            "jepa/sigreg_loss": sigreg_loss_total,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
            "jepa/encoder_norm_mean": encoder_norm_mean,
            "jepa/pred_norm_mean": pred_norm_mean,
            "jepa/target_norm_mean": target_norm_mean,
            "jepa/pred_std_across_batch": pred_std_across_batch,
            "jepa/weighted_pred_loss": self.prediction_loss_weight * pred_loss,
            "jepa/weighted_sigreg_loss": self.sigreg_loss_weight * sigreg_loss_total,
        }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return super()._should_include_in_checkpoint(key) or key.startswith("sigreg_proj")

    def _log_embedding_stats(
        self, embeddings: Tensor, namespace: str, prefix: str, max_channels: int = 4
    ) -> None:
        """
        Log lightweight scalar stats only (no plotting / IO).
        Expects embeddings shaped (B, T, C, H, W) or (B, T, D, 1, 1).
        """
        if embeddings.numel() == 0:
            return
        emb = embeddings.float()
        self.log(
            f"sigreg/{namespace}/{prefix}_mean",
            emb.mean(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            f"sigreg/{namespace}/{prefix}_var",
            emb.var(unbiased=False),
            on_step=True,
            sync_dist=True,
        )
        ch_mean = emb.mean(dim=(0, 1, 3, 4))
        ch_var = emb.var(dim=(0, 1, 3, 4), unbiased=False)
        num_ch = min(max_channels, int(ch_mean.numel()))
        for i in range(num_ch):
            self.log(
                f"sigreg/{namespace}/{prefix}_ch{i}_mean",
                ch_mean[i],
                on_step=True,
                sync_dist=True,
            )
            self.log(
                f"sigreg/{namespace}/{prefix}_ch{i}_var",
                ch_var[i],
                on_step=True,
                sync_dist=True,
            )

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

        online_latents = self._encode_online(videos)
        action_embeds = self.action_encoder(acts)
        pred_latents = self.predictor(online_latents[:, :-1], action_embeds[:, :-1])

        emb_online = online_latents[:, :-1]  # (B, T-1, C, H, W)
        emb_pred = pred_latents              # (B, T-1, C, H, W)
        self._log_embedding_stats(emb_online, namespace, "online_latents")
        self._log_embedding_stats(emb_pred, namespace, "pred_latents")

        if self.sigreg_proj is not None:
            bo, to = emb_online.shape[:2]
            bp, tp = emb_pred.shape[:2]
            emb_online_proj = self.sigreg_proj(emb_online.reshape(-1, self.state_dim)).reshape(
                bo, to, self.sigreg_proj_dim, 1, 1
            )
            emb_pred_proj = self.sigreg_proj(emb_pred.reshape(-1, self.state_dim)).reshape(
                bp, tp, self.sigreg_proj_dim, 1, 1
            )
            self._log_embedding_stats(emb_online_proj, namespace, "online_proj", max_channels=8)
            self._log_embedding_stats(emb_pred_proj, namespace, "pred_proj", max_channels=8)

    @torch.no_grad()
    def _encode_online_posterior(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        x = 2.0 * images - 1.0
        h = self.online_encoder(x)
        moments = self.online_quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    @torch.no_grad()
    def _maybe_log_drift(self, gt_videos: Tensor) -> None:
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
            self._ref_logvar
            - logvar_k
            + (logvar_k.exp() + (mean_k - self._ref_mean) ** 2) / self._ref_logvar.exp()
            - 1
        ).mean()
        l2_dist = (mean_k - self._ref_mean).pow(2).mean().sqrt()
        self.log("diagnostics/encoder_drift_kl", kl, on_step=True, sync_dist=True)
        self.log("diagnostics/encoder_drift_l2", l2_dist, on_step=True, sync_dist=True)

    @torch.no_grad()
    def _log_collapse_metrics(self, online_latents: Tensor) -> None:
        collapse_every = self.jepa_cfg.get("collapse_log_every", 50)
        if self.global_step % collapse_every != 0:
            return
        batch_var = online_latents.var(dim=0).mean()
        channel_std = online_latents.std(dim=(0, 1, 3, 4)).mean()
        feature_norm = online_latents.flatten(2).norm(dim=-1).mean()
        self.log("diagnostics/batch_variance", batch_var, on_step=True, sync_dist=True)
        self.log("diagnostics/channel_std", channel_std, on_step=True, sync_dist=True)
        self.log("diagnostics/feature_norm", feature_norm, on_step=True, sync_dist=True)

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        _, _, _, gt_videos, actions_raw = batch
        if gt_videos is not None:
            online_latents = self._encode_online(gt_videos)
            self._maybe_log_drift(gt_videos)
            self._log_collapse_metrics(online_latents)

            if self.sigreg_graph_every > 0 and self.global_step % self.sigreg_graph_every == 0:
                self._log_sigreg_embedding_graphs(
                    gt_videos, actions_raw, namespace="training"
                )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        super().validation_step(batch, batch_idx, namespace=namespace)
        if batch_idx != 0:
            return
        if self.sigreg_graph_every <= 0:
            return
        xs, conditions, masks, gt_videos, actions_raw = batch
        del xs, conditions, masks
        if gt_videos is None or actions_raw is None:
            return
        self._log_sigreg_embedding_graphs(gt_videos, actions_raw, namespace=namespace)
