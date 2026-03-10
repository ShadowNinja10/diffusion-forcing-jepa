"""
DiT3D variant that accepts per-patch encoder conditioning (zp) alongside
the standard noise-level and external-condition embeddings.

The encoder conditioning is patchified via a Conv2d with the same patch_size
as the main input, projected to hidden_size, and added to the adaLN
conditioning tensor so that every DiT block receives per-patch zp information.
"""

from typing import Optional

import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat

from .dit3d import DiT3D


class DiT3DDecoupled(DiT3D):
    """DiT3D with additional per-patch encoder conditioning for decoupled JEPA."""

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask: bool = True,
        encoder_cond_channels: int = 0,
    ):
        super().__init__(cfg, x_shape, max_tokens, external_cond_dim, use_causal_mask)

        self.encoder_cond_channels = encoder_cond_channels
        hidden_size = cfg.hidden_size

        if encoder_cond_channels > 0:
            channels, resolution, *_ = x_shape
            self.encoder_cond_proj = nn.Sequential(
                nn.Conv2d(
                    encoder_cond_channels,
                    hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    bias=True,
                ),
                nn.SiLU(),
            )
            self._init_encoder_cond_proj()
        else:
            self.encoder_cond_proj = None

    def _init_encoder_cond_proj(self) -> None:
        if self.encoder_cond_proj is None:
            return
        for m in self.encoder_cond_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        encoder_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, C, H, W) noised latents (zg-space)
            noise_levels:   (B, T) discrete noise levels
            external_cond:  (B, T, cond_dim) action conditioning
            external_cond_mask: optional mask for external_cond
            encoder_cond:   (B, T, Cp, H, W) stop-grad predictive latents (zp)
        """
        input_batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        emb = self.noise_level_pos_embedding(noise_levels)

        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        if encoder_cond is not None and self.encoder_cond_proj is not None:
            ec = rearrange(encoder_cond, "b t c h w -> (b t) c h w")
            ec_emb = self.encoder_cond_proj(ec)  # (B*T, hidden, gh, gw)
            ec_emb = rearrange(ec_emb, "(b t) c gh gw -> b (t gh gw) c", b=input_batch_size)
            emb = emb + ec_emb

        x = self.dit_base(x, emb)
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )
        x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)
        return x
