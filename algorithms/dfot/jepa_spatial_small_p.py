"""
Standalone Spatial JEPA Training with a smaller predictor.

Identical to JEPASpatialTraining except the predictor uses reduced depth
(default 2 instead of 4), enforcing more of the predictive burden on the
encoder. Use jepa.predictor_depth_small in config to override (default: 2).
"""

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .dfot_video_jepa_spatial import ActionEncoder, SpatialTemporalPredictor
from .jepa_spatial_training import JEPASpatialTraining


class JEPASpatialSmallP(JEPASpatialTraining):
    """
    Same as JEPASpatialTraining but with a smaller predictor (reduced depth).

    Uses predictor_depth_small (default: 2) instead of predictor_depth,
    shifting predictive capacity to the encoder.
    """

    def _build_jepa_model(self):
        """Build JEPA model with smaller predictor depth."""
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
        depth = int(jepa_cfg.get("predictor_depth_small", 2))

        self.predictor = SpatialTemporalPredictor(
            latent_channels=latent_channels,
            latent_h=latent_h,
            latent_w=latent_w,
            action_dim=jepa_cfg.action_embed_dim,
            hidden_dim=jepa_cfg.predictor_hidden_dim,
            depth=depth,
            heads=jepa_cfg.predictor_heads,
            dim_head=jepa_cfg.get("predictor_dim_head", 64),
            mlp_ratio=jepa_cfg.get("predictor_mlp_ratio", 4.0),
            max_seq_len=self.max_tokens + 1,
            patch_size=patch_size,
            factorized_attention=factorized,
            dropout=jepa_cfg.get("predictor_dropout", 0.1),
        )

        from algorithms.vae.common.losses.lpips import LPIPS

        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        rank_zero_print(cyan(f"JEPA spatial small-predictor depth: {depth}"))
        rank_zero_print(cyan(f"JEPA spatial predictor patch_size: {patch_size}"))
        rank_zero_print(cyan(f"JEPA spatial predictor factorized_attention: {factorized}"))
        rank_zero_print(cyan(f"JEPA Training mode: {self.jepa_training_mode}"))
        rank_zero_print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every}"))
