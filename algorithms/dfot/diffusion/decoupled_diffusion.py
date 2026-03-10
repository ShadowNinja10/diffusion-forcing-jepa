"""
Continuous-time diffusion wrapper that passes encoder conditioning (zp) through
to the backbone alongside the standard external conditioning.

Subclasses ContinuousDiffusion and overrides only the methods that need the
extra ``encoder_cond`` argument so that all scheduling / loss-weight logic
is inherited unchanged.
"""

from typing import Optional, Callable

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from ..backbones import DiT3DDecoupled
from .continuous_diffusion import ContinuousDiffusion
from .discrete_diffusion import ModelPrediction


class DecoupledDiffusion(ContinuousDiffusion):
    """ContinuousDiffusion extended with per-patch encoder conditioning (zp)."""

    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        encoder_cond_channels: int = 0,
    ):
        self._encoder_cond_channels = encoder_cond_channels
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens, external_cond_dim)

    def _build_model(self):
        self.model = DiT3DDecoupled(
            cfg=self.backbone_cfg,
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
            use_causal_mask=self.use_causal_mask,
            encoder_cond_channels=self._encoder_cond_channels,
        )

    # ------------------------------------------------------------------
    # model_predictions: used during *sampling* (k is discrete integer)
    # ------------------------------------------------------------------

    def model_predictions(
        self, x, k, external_cond=None, external_cond_mask=None, encoder_cond=None
    ):
        # During sampling k is a long integer index; look up the logsnr buffer.
        model_output = self.model(
            x,
            self.precond_scale * self.logsnr[k],
            external_cond,
            external_cond_mask,
            encoder_cond,
        )

        if self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)
        elif self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        return ModelPrediction(pred_noise, x_start, model_output)

    # ------------------------------------------------------------------
    # forward: used during *training* (k is a float in [0, 1])
    # Mirrors ContinuousDiffusion.forward but threads encoder_cond through.
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        encoder_cond: Optional[torch.Tensor] = None,
    ):
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        v_pred = self.model(
            x_t, self.precond_scale * logsnr, external_cond, None, encoder_cond
        )
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    # ------------------------------------------------------------------
    # Sampling helpers: pass encoder_cond through
    # ------------------------------------------------------------------

    def p_mean_variance(
        self, x, k, external_cond=None, external_cond_mask=None, encoder_cond=None
    ):
        model_pred = self.model_predictions(
            x=x,
            k=k,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            encoder_cond=encoder_cond,
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        encoder_cond: Optional[torch.Tensor] = None,
    ):
        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
                encoder_cond=encoder_cond,
            )

        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."
        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            guidance_fn=guidance_fn,
            encoder_cond=encoder_cond,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        encoder_cond: Optional[torch.Tensor] = None,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implemented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            encoder_cond=encoder_cond,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        encoder_cond: Optional[torch.Tensor] = None,
    ):
        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    encoder_cond=encoder_cond,
                )
                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )
                grad = -torch.autograd.grad(guidance_loss, x)[0]
                grad = torch.nan_to_num(grad, nan=0.0)
                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,
                    self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise),
                    model_pred.pred_x_start,
                )
        else:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                encoder_cond=encoder_cond,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(self.add_shape_channels(mask), x, x_pred)
        return x_pred
