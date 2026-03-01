"""
Dual-Encoder JEPA for DFoT (BYOL-Style).

Architecture:
- **Target VAE** (EMA-updated): Provides stable latents for DFoT diffusion
  and prediction targets for JEPA.  Both encoder and decoder are updated
  via EMA from their online counterparts.
- **Online encoder** (trainable): A copy of the VAE encoder that receives
  JEPA gradients.  Learns to produce latents that are inherently predictive
  of future states.
- **Online decoder** (trainable): Adapts to the online encoder; used for
  collapse diagnostics and EMA-synced into the target VAE decoder.
- **ViT Predictor**: Operates on *clean* latent states from the online
  encoder and encoded actions; predicts future clean latent states
  produced by the *target* encoder (BYOL asymmetry).
- **EMA sync**: The online encoder+decoder is slowly blended into the
  target VAE (like BYOL / DINO) so DFoT gradually benefits from improved
  representations without sudden distribution shifts.
- **Running stats**: data_mean / data_std are refreshed after each EMA
  update to track the target encoder's evolving output distribution.

Gradient paths:
  DFoT loss     -->  diffusion backbone only
  JEPA loss     -->  online encoder + predictor + action encoder
  Decoder loss  -->  online decoder only (encoder detached)
"""

import time
from copy import deepcopy
from typing import Optional, Any, Dict, Tuple, Callable
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat
from transformers import get_scheduler

from algorithms.vae import ImageVAE
from algorithms.vae.common.losses.lpips import LPIPS
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
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
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
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        # FlashAttention-2: ~2-4x faster, O(1) memory vs O(T^2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ViTPredictor(nn.Module):
    """
    Causal ViT predictor that operates on *clean* latent states and
    encoded actions to predict the next clean latent state.
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
        self.hidden_dim = hidden_dim

        # Input projections
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection -> next state
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            states:  (B, T, state_dim)  -- flattened clean latent states
            actions: (B, T, action_dim) -- encoded action embeddings
        Returns:
            pred_next_states: (B, T, state_dim) -- prediction for s_{t+1}
        """
        B, T, _ = states.shape

        state_emb = self.state_proj(states)
        action_emb = self.action_proj(actions)
        combined = torch.cat([state_emb, action_emb], dim=-1)
        x = self.combine_proj(combined)

        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output_proj(x)


class ActionEncoder(nn.Module):
    """Encodes raw actions into embeddings."""

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
    DFoT with dual-encoder JEPA (BYOL-style).

    - Target encoder (self.vae) is frozen and provides stable latents for DFoT.
    - Online encoder (self.online_encoder + self.online_quant_conv) is trainable
      and produces clean latents for the JEPA predictor.
    - EMA periodically blends online -> target so DFoT gradually benefits.
    """

    def __init__(self, cfg: DictConfig):
        self.jepa_cfg = cfg.jepa
        self.jepa_loss_weight = cfg.jepa.loss_weight
        self.dfot_loss_weight = cfg.jepa.get("dfot_loss_weight", 1.0)
        self.jepa_training_mode = cfg.jepa.get("training_mode", "teacher_forcing")
        self._latent_norm_eps = float(cfg.jepa.get("latent_norm_eps", 1e-4))

        # EMA config
        self.ema_decay = cfg.jepa.get("ema_decay", 0.999)
        self.ema_update_every = cfg.jepa.get("ema_update_every", 100)

        # Force online latent mode
        assert cfg.latent.enable, "JEPA training requires latent diffusion"
        assert cfg.latent.type == "online", "JEPA training requires online latent processing"

        super().__init__(cfg)

        # Manual optimization: DFoT and JEPA+Decoder use separate backward
        # passes so the diffusion graph is freed before JEPA runs, cutting
        # peak GPU memory roughly in half.
        self.automatic_optimization = False

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_model(self):
        """Build DFoT model, then build JEPA components."""
        super()._build_model()
        self._build_jepa_model()

        # Running statistics for tracking target encoder output distribution.
        # Initialized from the pretrained VAE stats, updated after each EMA step.
        # Use register_buffer so Lightning auto-moves these to the correct device.
        self.register_buffer("_running_mean", self.data_mean.clone())
        self.register_buffer("_running_std", self.data_std.clone())
        self._stats_momentum = 0.01  # EMA momentum for running stats

    def _build_jepa_model(self):
        """Build ViT predictor and action encoder."""
        jepa_cfg = self.jepa_cfg

        # State dim = flattened latent
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

        # Perceptual loss for decoder reconstruction diagnostic
        self.perceptual_loss = LPIPS().eval()
        # All LPIPS params are frozen internally; mark explicitly for clarity
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        rank_zero_print(cyan(f"JEPA State dim: {self.state_dim}"))
        rank_zero_print(cyan(f"JEPA Predictor hidden dim: {jepa_cfg.predictor_hidden_dim}"))
        rank_zero_print(cyan(f"JEPA Training mode: {self.jepa_training_mode}"))
        rank_zero_print(cyan(f"JEPA EMA decay: {self.ema_decay}, update every: {self.ema_update_every}"))
        rank_zero_print(cyan(f"DFoT loss weight: {self.dfot_loss_weight}"))
        decoder_loss_every = jepa_cfg.get("decoder_loss_every", 1)
        recon_reg = jepa_cfg.get("recon_regularizer", False)
        if not recon_reg and decoder_loss_every > 1:
            rank_zero_print(cyan(f"Decoder loss every {decoder_loss_every} steps (recon_reg=false, diagnostic only)"))

    # -----------------------------------------------------------------
    # VAE loading -- dual encoder
    # -----------------------------------------------------------------

    def _load_vae(self) -> None:
        """
        Load VAE and create the dual-encoder setup:
        - self.vae = full VAE (target encoder -- frozen)
        - self.online_encoder = trainable copy of encoder
        - self.online_quant_conv = trainable copy of quant_conv
        """
        assert not self.is_latent_video_vae, "JEPA currently only supports ImageVAE"

        self.vae = ImageVAE.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)

        # Freeze entire target VAE
        freeze_model(self.vae)
        rank_zero_print(cyan("Target VAE is fully frozen"))

        # Create trainable online encoder (deep copy of encoder + quant_conv)
        self.online_encoder = deepcopy(self.vae.encoder)
        self.online_quant_conv = deepcopy(self.vae.quant_conv)

        # Create trainable online decoder (deep copy of post_quant_conv + decoder)
        self.online_post_quant_conv = deepcopy(self.vae.post_quant_conv)
        self.online_decoder = deepcopy(self.vae.decoder)

        # Ensure online copies are trainable
        for m in (self.online_encoder, self.online_quant_conv,
                  self.online_post_quant_conv, self.online_decoder):
            for p in m.parameters():
                p.requires_grad = True

        n_online_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in (self.online_encoder, self.online_quant_conv,
                      self.online_post_quant_conv, self.online_decoder)
        )
        rank_zero_print(
            cyan(f"Online encoder+decoder created: {n_online_params / 1e6:.1f}M trainable params")
        )

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_online(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the trainable online encoder.
        Uses mode() for deterministic, clean latents.

        When recon_regularizer is enabled, the backward pass flows through
        the encoder; run in float32 to avoid fp16 overflow in GroupNorm /
        attention layers.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        recon_reg = self.jepa_cfg.get("recon_regularizer", False)

        vae_bs = self.cfg.vae.batch_size
        total = x_normalized.shape[0]
        latent_chunks: list = []
        for start in range(0, total, vae_bs):
            x_chunk = x_normalized[start : start + vae_bs]
            if recon_reg:
                with torch.amp.autocast("cuda", enabled=False):
                    h = self.online_encoder(x_chunk.float())
                    moments = self.online_quant_conv(h)
            else:
                h = self.online_encoder(x_chunk)
                moments = self.online_quant_conv(h)
            mean, _ = torch.chunk(moments, 2, dim=1)
            latent_chunks.append(mean)
        latents_flat = torch.cat(latent_chunks, dim=0)

        return latents_flat.reshape(B, T, *latents_flat.shape[1:])

    @torch.no_grad()
    def _encode_target_both(self, videos: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode raw videos with the frozen target encoder **once**, returning
        both sampled latents (for DFoT) and mode latents (for JEPA targets).

        This avoids running the target encoder twice per training step.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            sampled: (B, T, C, H, W)  -- stochastic, used by DFoT
            mode:    (B, T, C, H, W)  -- deterministic, used as JEPA targets
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        vae_bs = self.cfg.vae.batch_size
        total = x_normalized.shape[0]
        sample_chunks: list = []
        mode_chunks: list = []
        for start in range(0, total, vae_bs):
            x_chunk = x_normalized[start : start + vae_bs]
            posterior = self.vae.encode(x_chunk)
            sample_chunks.append(posterior.sample())
            mode_chunks.append(posterior.mode())

        sampled = torch.cat(sample_chunks, dim=0).reshape(B, T, -1, *sample_chunks[0].shape[2:])
        mode = torch.cat(mode_chunks, dim=0).reshape(B, T, -1, *mode_chunks[0].shape[2:])
        return sampled, mode

    def _decode_online(self, z: Tensor) -> Tensor:
        """
        Decode latents with the trainable online decoder.

        Args:
            z: (N, C, H, W) latent tensor
        Returns:
            x_recon: (N, 3, H, W) reconstructed pixels in [-1, 1]
        """
        z = self.online_post_quant_conv(z)
        return self.online_decoder(z)

    def _compute_decoder_loss(
        self,
        videos: Tensor,
        online_latents: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Reconstruction loss through the online decoder, reusing pre-computed
        online encoder latents (shared with the JEPA forward pass).

        When ``recon_regularizer`` is **False** (default):
            Latents are detached so only the decoder receives gradients.

        When ``recon_regularizer`` is **True**:
            Gradients flow through the encoder too, acting as a
            reconstruction-based regularizer that prevents collapse.

        Args:
            videos:          (B, T, 3, H, W) raw frames in [0, 1]
            online_latents:  (B, T, C, H, W) pre-computed latents from online encoder
            timings:         optional dict to populate with decoder_forward_ms, decoder_mse_ms, decoder_lpips_ms
            _tick:           optional sync+perf_counter callable for timing
        Returns:
            decoder_loss: scalar
            log_dict: dict of per-component losses
        """
        do_timing = timings is not None and _tick is not None

        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0  # [-1, 1]

        recon_reg = self.jepa_cfg.get("recon_regularizer", False)

        z_flat = rearrange(online_latents, "b t c h w -> (b t) c h w")
        if not recon_reg:
            z_flat = z_flat.detach()

        decoder_bs = self.jepa_cfg.get("decoder_chunk_size", 4)
        use_amp = self.jepa_cfg.get("decoder_lpips_amp", True) and torch.cuda.is_available()
        lpips_every = self.jepa_cfg.get("decoder_lpips_every", 1)
        run_lpips = (lpips_every <= 1) or (self.global_step % lpips_every == 0)

        total = x_norm.shape[0]
        mse_losses: list = []
        lpips_losses: list = []
        t_decoder, t_mse, t_lpips = 0.0, 0.0, 0.0
        for start in range(0, total, decoder_bs):
            x_chunk = x_norm[start : start + decoder_bs]
            z_chunk = z_flat[start : start + decoder_bs]
            if do_timing:
                t0 = _tick()
            if recon_reg:
                with torch.amp.autocast("cuda", enabled=False):
                    x_recon = self._decode_online(z_chunk.float())
            else:
                x_recon = self._decode_online(z_chunk)
            if do_timing:
                t_decoder += _tick() - t0
            if do_timing:
                t0 = _tick()
            mse_losses.append(F.mse_loss(x_recon.float(), x_chunk.float()))
            if do_timing:
                t_mse += _tick() - t0
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
            timings["decoder_mse_ms"] = t_mse * 1000
            if run_lpips:
                timings["decoder_lpips_ms"] = t_lpips * 1000

        mse_loss = torch.stack(mse_losses).mean()
        lpips_loss = (
            torch.stack(lpips_losses).mean()
            if lpips_losses
            else torch.tensor(0.0, device=mse_loss.device)
        )

        mse_w = self.jepa_cfg.get("decoder_mse_weight", 1.0)
        lpips_w = self.jepa_cfg.get("decoder_lpips_weight", 1.0)
        decoder_loss = mse_w * mse_loss
        if run_lpips:
            decoder_loss = decoder_loss + lpips_w * lpips_loss

        log_dict = {
            "decoder/mse_loss": mse_loss.detach(),
            "decoder/lpips_loss": lpips_loss.detach(),
            "decoder/total_loss": decoder_loss.detach(),
            "decoder/lpips_active": torch.tensor(float(run_lpips), device=mse_loss.device),
            "decoder/recon_regularizer": torch.tensor(float(recon_reg), device=mse_loss.device),
        }
        return decoder_loss, log_dict

    @torch.no_grad()
    def _log_decoder_reconstructions(
        self, gt_videos: Tensor, namespace: str, step: int
    ) -> None:
        """
        Save side-by-side reconstruction two ways:
          1. PNG grid on disk: {trainer.log_dir}/decoder_vis/{namespace}/step_{step:06d}.png
             Each row = one timestep, three columns = [target_enc | online_enc | gt].
             Works even without wandb (useful for CPU/offline testing).
          2. Wandb video (when a logger is attached).

        Forward passes run on ALL ranks to avoid NCCL collective desync in DDP;
        only rank 0 performs the actual I/O (PNG save, wandb log).
        """
        import os
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if self.trainer.sanity_checking:
            return

        

        videos = gt_videos[:1]  # (1, T, 3, H, W) in [0, 1]
        T = videos.shape[1]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0  # [-1, 1]

        # Target encoder (frozen) + frozen VAE decoder
        posterior = self.vae.encode(x_norm)
        z_target = posterior.mode()
        recon_target = self.vae.decode(z_target)  # [-1, 1]
        recon_target = ((recon_target.clamp(-1, 1) + 1) / 2).reshape(1, T, *recon_target.shape[1:])

        # Online encoder (trainable) + online decoder (trainable)
        h = self.online_encoder(x_norm)
        moments = self.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        recon_online = self._decode_online(mean)  # [-1, 1]
        recon_online = ((recon_online.clamp(-1, 1) + 1) / 2).reshape(1, T, *recon_online.shape[1:])

        # Cast to float32: numpy / torchvision don't support bfloat16
        recon_target = recon_target.float()
        recon_online = recon_online.float()
        videos = videos.float()

        if not is_rank_zero:
            return
        

        # --- 1. Disk: PNG grid (T rows × 3 cols = target | online | gt) ---
        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(log_dir, "decoder_vis", namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")
        frames = torch.stack(
            [recon_target[0], recon_online[0], videos[0]], dim=1
        ).reshape(-1, *videos.shape[2:])  # (3T, C, H, W)
        grid = torchvision.utils.make_grid(frames, nrow=3, padding=2, pad_value=0.5)
        torchvision.utils.save_image(grid, save_path)

        # --- 2. Wandb: video (when logger is attached) ---
        if self.logger:
            log_video(
                [recon_target, recon_online],
                videos,             # GT already in [0, 1]
                step=step,
                namespace=namespace,
                prefix="decoder_recon",
                captions=["target | online | gt"],
                logger=self.logger.experiment,
            )

    @torch.no_grad()
    def _log_jepa_predictions(
        self, gt_videos: Tensor, actions: Tensor, namespace: str, step: int
    ) -> None:
        """
        Decode JEPA-predicted embeddings to pixel space and log alongside GT.

        The predictor takes (online_state_t, action_t) -> predicted_state_{t+1}.
        We decode predicted states with the online decoder so that the quality
        should start from gibberish and improve as training progresses.

        Layout: PNG grid with T-1 rows × 2 cols = [predicted | gt].
        Also logs a wandb video when a logger is attached.

        Forward passes run on ALL ranks to avoid NCCL collective desync in DDP;
        only rank 0 performs the actual I/O (PNG save, wandb log).
        """
        import os
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if self.trainer.sanity_checking:
            return

        videos = gt_videos[:1]  # (1, T, 3, H, W) in [0, 1]
        actions_batch = actions[:1]  # (1, T, action_dim)
        B, T = videos.shape[:2]

        if T < 2:
            return

        online_latents = self._encode_online(videos)  # (1, T, C, H, W)
        latent_shape = online_latents.shape[2:]  # (C, H, W)

        online_states = online_latents.reshape(B, T, -1)  # (1, T, state_dim)
        action_embeds = self.action_encoder(actions_batch)  # (1, T, action_embed_dim)

        states_input = online_states[:, :-1]        # (1, T-1, state_dim)
        action_input = action_embeds[:, :-1]        # (1, T-1, action_embed_dim)
        states_pred = self.predictor(states_input, action_input)  # (1, T-1, state_dim)

        pred_latents = states_pred.reshape(B * (T - 1), *latent_shape)  # (T-1, C, H, W)
        pred_recon = self._decode_online(pred_latents)  # (T-1, 3, H, W) in [-1, 1]
        pred_recon = ((pred_recon.clamp(-1, 1) + 1) / 2)  # [0, 1]
        pred_recon = pred_recon.reshape(1, T - 1, *pred_recon.shape[1:]).float()

        gt_aligned = videos[:, 1:].float()  # (1, T-1, 3, H, W)

        if not is_rank_zero:
            return

        # --- 1. Disk: PNG grid ((T-1) rows × 2 cols = predicted | gt) ---
        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(log_dir, "jepa_pred_vis", namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")
        frames = torch.stack(
            [pred_recon[0], gt_aligned[0]], dim=1
        ).reshape(-1, *videos.shape[2:])  # (2*(T-1), C, H, W)
        grid = torchvision.utils.make_grid(frames, nrow=2, padding=2, pad_value=0.5)
        torchvision.utils.save_image(grid, save_path)

        # --- 2. Wandb: video (when logger is attached) ---
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

    # -----------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target(self) -> None:
        """
        Exponential moving average update: blend online encoder AND decoder
        weights into the target VAE.  Called every ``ema_update_every`` steps.

        This keeps the target encoder-decoder pair consistent so that
        _decode() (used during sampling/metrics) stays matched with the
        evolving target encoder.
        """
        decay = self.ema_decay

        # Encoder
        for p_online, p_target in zip(
            self.online_encoder.parameters(), self.vae.encoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        # quant_conv (encoder side)
        for p_online, p_target in zip(
            self.online_quant_conv.parameters(), self.vae.quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        # post_quant_conv (decoder side)
        for p_online, p_target in zip(
            self.online_post_quant_conv.parameters(), self.vae.post_quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        # Decoder
        for p_online, p_target in zip(
            self.online_decoder.parameters(), self.vae.decoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    @torch.no_grad()
    def _update_running_latent_stats(self, raw_latents: Tensor) -> None:
        """
        Update running mean/std from a batch of raw (un-normalized) target
        encoder latents.  Uses exponential moving average so the estimate
        stays fresh as the target encoder drifts via EMA.
        """
        # raw_latents: (B, T, C, H, W)  -- compute per-channel stats
        # Collapse all dims except C to match data_mean shape
        mean = raw_latents.mean(dim=(0, 1, 3, 4))  # (C,)
        std = raw_latents.std(dim=(0, 1, 3, 4))  # (C,)

        # Reshape to match data_mean buffer shape: (C, 1, 1)
        target_shape = self.data_mean.shape
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).reshape(target_shape)
        std = torch.nan_to_num(std, nan=self._latent_norm_eps, posinf=1.0, neginf=self._latent_norm_eps)
        std = std.clamp_min(self._latent_norm_eps).reshape(target_shape)

        m = self._stats_momentum
        self._running_mean.mul_(1 - m).add_(mean, alpha=m)
        self._running_std.mul_(1 - m).add_(std, alpha=m)

    @torch.no_grad()
    def _refresh_data_stats(self) -> None:
        """Copy running stats into the normalization buffers."""
        self.data_mean.copy_(torch.nan_to_num(self._running_mean, nan=0.0, posinf=0.0, neginf=0.0))
        safe_std = torch.nan_to_num(
            self._running_std, nan=self._latent_norm_eps, posinf=1.0, neginf=self._latent_norm_eps
        ).clamp_min(self._latent_norm_eps)
        self.data_std.copy_(safe_std)

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        """Three param groups: diffusion, JEPA head, online encoder."""
        params_groups = []

        # 1. Diffusion model (backbone -- trained by DFoT loss only)
        params_groups.append({
            "params": list(self.diffusion_model.parameters()),
            "lr": self.cfg.lr,
            "name": "diffusion",
        })

        # 2. JEPA predictor + action encoder
        jepa_params = (
            list(self.predictor.parameters()) +
            list(self.action_encoder.parameters())
        )
        params_groups.append({
            "params": jepa_params,
            "lr": self.jepa_cfg.lr,
            "name": "jepa",
        })

        # 3. Online encoder (lower LR for stability)
        online_params = (
            list(self.online_encoder.parameters()) +
            list(self.online_quant_conv.parameters())
        )
        params_groups.append({
            "params": online_params,
            "lr": self.jepa_cfg.get("encoder_lr", 1e-5),
            "name": "online_encoder",
        })

        # 4. Online decoder (diagnostic reconstruction probe)
        decoder_params = (
            list(self.online_post_quant_conv.parameters()) +
            list(self.online_decoder.parameters())
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

    # -----------------------------------------------------------------
    # Batch preprocessing
    # -----------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Returns 5-element tuple:
            xs:          (B, T, C, H, W) normalized latents from *target* encoder for DFoT
            conditions:  (B, T, cond_dim)
            masks:       (B, T)
            gt_videos:   (B, T, 3, H, W) raw videos for JEPA
            actions_raw: (B, T, action_dim) raw actions for JEPA
        """
        gt_videos = batch.get("videos", None)
        actions_raw = batch.get("conds", None)

        # DFoT path: encode with frozen target VAE.
        # Encode ONCE and get both sampled (for DFoT) and mode (for JEPA targets)
        # to avoid running the target encoder twice per step.
        if self.is_latent_diffusion and self.is_latent_online:
            xs, target_mode = self._encode_target_both(batch["videos"])
            # Cache mode latents for _compute_jepa_loss
            self._cached_target_latents = target_mode
        else:
            xs = batch.get("latents", batch["videos"])
            self._cached_target_latents = None

        # Track running stats of raw latents (before normalization)
        # so we can refresh data_mean/data_std after EMA updates.
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
    # JEPA Loss
    # -----------------------------------------------------------------

    def _compute_jepa_loss(
        self,
        online_latents: Tensor,
        actions: Tensor,
        masks: Tensor,
        timings: Optional[Dict[str, float]] = None,
        _tick: Optional[Callable[[], float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute JEPA prediction loss (BYOL-style asymmetric).

        Pipeline:
        1. Use pre-computed **online** encoder latents as predictor inputs
        2. Use cached **target** encoder latents as prediction targets
        3. Encode actions
        4. Predictor: (online_states[:-1], actions[:-1]) -> pred_states
        5. Loss = smooth_l1(pred_states, target_states[1:])

        Args:
            online_latents: (B, T, C, H, W) pre-computed from online encoder
            actions:        (B, T, action_dim) raw actions
            masks:          (B, T) validity masks
            timings:        optional dict for jepa_action_encoder_ms, jepa_predictor_ms, jepa_loss_compute_ms
            _tick:          optional sync+perf_counter callable for timing
        """
        do_timing = timings is not None and _tick is not None
        B, T = online_latents.shape[:2]

        if T < 2:
            _zero = torch.tensor(0.0, device=online_latents.device)
            return _zero, {
                "jepa/pred_loss": _zero,
                "jepa/mse": _zero,
                "jepa/cos_sim": _zero,
            }

        # 1. Online latents already computed (grad flows through encoder)

        # 2. Target latents from cache; slice to T for safety
        target_latents = self._cached_target_latents[:, :T]  # (B, T, C, H, W)

        # Optionally normalize latents so the predictor sees unit-scale inputs
        # and smooth_l1_loss operates with a consistent beta transition point.
        if self.jepa_cfg.get("normalize_jepa_latents", False):
            shape = [1, 1] + list(self.data_mean.shape)  # (1, 1, C, 1, 1)
            mean = self.data_mean.reshape(shape)
            std = self.data_std.reshape(shape).clamp_min(self._latent_norm_eps)
            online_latents = (online_latents - mean) / std
            target_latents = (target_latents - mean) / std

        online_states = online_latents.reshape(B, T, -1)  # (B, T, state_dim)
        target_states = target_latents.reshape(B, T, -1)  # (B, T, state_dim)

        # 3. Encode actions
        if do_timing:
            t0 = _tick()
        action_embeds = self.action_encoder(actions)  # (B, T, action_embed_dim)
        if do_timing and timings is not None:
            timings["jepa_action_encoder_ms"] = (_tick() - t0) * 1000

        # 4. Inputs (online) and targets (target encoder, already no-grad)
        states_input = online_states[:, :-1]          # (B, T-1, state_dim)
        action_embeds_input = action_embeds[:, :-1]   # (B, T-1, action_embed_dim)
        states_target = target_states[:, 1:]          # (B, T-1, state_dim) -- from target encoder

        # 5. Predict
        if do_timing:
            t0 = _tick()
        if self.jepa_training_mode == "autoregressive":
            pred_list = []
            current = online_states[:, 0:1]
            for t in range(T - 1):
                act_t = action_embeds_input[:, t : t + 1]
                pred_t = self.predictor(current, act_t)
                pred_list.append(pred_t)
                # Detach to prevent long backprop chains
                current = pred_t.detach() if t < T - 2 else pred_t
            states_pred = torch.cat(pred_list, dim=1)
        else:
            # Teacher forcing (default)
            states_pred = self.predictor(states_input, action_embeds_input)
        if do_timing and timings is not None:
            timings["jepa_predictor_ms"] = (_tick() - t0) * 1000

        # 6. Loss
        if do_timing:
            t0 = _tick()
        transition_masks = masks[:, :-1] & masks[:, 1:]

        pred_loss = F.smooth_l1_loss(
            states_pred, states_target, reduction="none"
        ).mean(dim=-1)  # (B, T-1)

        if transition_masks.sum() > 0:
            pred_loss = (
                (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
            )
        else:
            pred_loss = pred_loss.mean()
        if do_timing and timings is not None:
            timings["jepa_loss_compute_ms"] = (_tick() - t0) * 1000

        # Metrics
        with torch.no_grad():
            if transition_masks.sum() > 0:
                mse = F.mse_loss(
                    states_pred[transition_masks],
                    states_target[transition_masks],
                )
                cos_sim = F.cosine_similarity(
                    states_pred[transition_masks],
                    states_target[transition_masks],
                    dim=-1,
                ).mean()
            else:
                mse = torch.tensor(0.0, device=online_latents.device)
                cos_sim = torch.tensor(0.0, device=online_latents.device)

        log_dict = {
            "jepa/pred_loss": pred_loss,
            "jepa/mse": mse,
            "jepa/cos_sim": cos_sim,
        }
        return pred_loss, log_dict

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """
        Manual optimization with two-phase backward:
          Phase 1  -- DFoT loss  (diffusion backbone only)
          Phase 2  -- JEPA + Decoder loss  (shared online encoder pass)

        The DFoT computation graph is freed before JEPA runs, so peak
        GPU memory ≈ max(DFoT_graph, JEPA+Decoder_graph) instead of
        sum(all three).
        """
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

        # ========== Phase 1: DFoT Loss (diffusion backbone only) ==========
        if do_timing:
            t0 = _tick()
        noise_levels, masks_dfot = self._get_training_noise_levels(xs, masks)
        xs_pred, dfot_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        dfot_loss = self._reweight_loss(dfot_loss, masks_dfot)
        if do_timing:
            timings["phase1_dfot_forward_ms"] = (_tick() - t0) * 1000

        # ========== Phase 2: JEPA + Decoder (shared encoder pass) =========
        jepa_loss = torch.tensor(0.0, device=xs.device)
        decoder_loss = torch.tensor(0.0, device=xs.device)
        _zero = torch.tensor(0.0, device=xs.device)
        jepa_log_dict: Dict[str, Tensor] = {
            "jepa/pred_loss": _zero,
            "jepa/mse": _zero,
            "jepa/cos_sim": _zero,
        }
        decoder_log_dict: Dict[str, Tensor] = {
            "decoder/mse_loss": _zero,
            "decoder/lpips_loss": _zero,
            "decoder/total_loss": _zero,
            "decoder/lpips_active": _zero,
            "decoder/recon_regularizer": _zero,
        }

        has_jepa = (
            gt_videos is not None
            and actions_raw is not None
            and self.jepa_loss_weight > 0
        )
        decoder_loss_weight = self.jepa_cfg.get("decoder_loss_weight", 0.0)
        has_decoder = gt_videos is not None and decoder_loss_weight > 0
        # Validation denoising calls into training_step under no_grad; skip
        # heavy JEPA/decoder losses there to avoid OOM during sanity/val.
        if not is_train:
            has_jepa = False
            has_decoder = False

        decoder_loss_every = self.jepa_cfg.get("decoder_loss_every", 1)
        recon_reg = self.jepa_cfg.get("recon_regularizer", False)
        if has_decoder and not recon_reg and decoder_loss_every > 1:
            if self.global_step % decoder_loss_every != 0:
                has_decoder = False

        if has_jepa or has_decoder:
            if do_timing:
                t0 = _tick()
            online_latents = self._encode_online(gt_videos)
            if do_timing:
                timings["phase2_online_encoder_ms"] = (_tick() - t0) * 1000

            if has_decoder:
                if do_timing:
                    t0 = _tick()
                decoder_loss, decoder_log_dict = self._compute_decoder_loss(
                    gt_videos,
                    online_latents,
                    timings=timings if do_timing else None,
                    _tick=_tick if do_timing else None,
                    
                )
                if do_timing:
                    timings["phase2_decoder_loss_ms"] = (_tick() - t0) * 1000

            if has_jepa:
                if do_timing:
                    t0 = _tick()
                jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                    online_latents,
                    actions_raw,
                    masks,
                    timings=timings if do_timing else None,
                    _tick=_tick if do_timing else None,
                )
                if do_timing:
                    timings["phase2_jepa_loss_ms"] = (_tick() - t0) * 1000

        # Phase-2 backward must exclude DFoT loss to avoid backwarding through
        # the already-freed DFoT graph.
        phase2_loss = (
            self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
        )
        phase2_needs_backward = is_train and phase2_loss.requires_grad

        # ========== Two-phase backward with DDP-safe gradient sync ==========
        # With find_unused_parameters=True, a naive two-phase backward causes
        # DDP to mark Phase 2 params as "unused" after Phase 1 backward,
        # triggering premature allreduce with zero gradients. The actual Phase 2
        # gradients then arrive late, creating non-deterministic NCCL op counts
        # across ranks (sequence number mismatch → timeout).
        #
        # Fix: wrap Phase 1 backward in no_sync() so DDP defers ALL gradient
        # reduction until Phase 2 backward, where find_unused_parameters picks
        # up Phase 1 gradients from param.grad and reduces them properly.
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

        # ========== Gradient clipping + optimizer step ====================
        if is_train:
            if do_timing:
                t0 = _tick()
            clip_val = self.trainer.gradient_clip_val
            if not clip_val:
                clip_val = 1.0
            self.clip_gradients(
                opt,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=(
                    self.trainer.gradient_clip_algorithm or "norm"
                ),
            )
            if do_timing:
                timings["optimizer_clip_grad_ms"] = (_tick() - t0) * 1000
            if do_timing:
                t0 = _tick()
            opt.step()
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
            # Sort keys for consistent output; print in two lines for readability
            keys_ordered = [
                "phase1_dfot_forward_ms", "phase1_dfot_backward_ms",
                "phase2_online_encoder_ms", "phase2_jepa_loss_ms",
                "jepa_action_encoder_ms", "jepa_predictor_ms", "jepa_loss_compute_ms",
                "phase2_decoder_loss_ms", "decoder_forward_ms", "decoder_mse_ms", "decoder_lpips_ms",
                "phase2_decoder_backward_ms", "phase2_jepa_backward_ms",
                "optimizer_clip_grad_ms", "optimizer_step_ms", "optimizer_scheduler_ms",
                "step_total_ms",
            ]
            present = [k for k in keys_ordered if k in timings]
            extra = [k for k in sorted(timings.keys()) if k not in present]
            all_keys = present + extra
            line = " | ".join(f"{k}: {timings[k]:.0f}ms" for k in all_keys)
            rank_zero_print(cyan(f"[Timing step {self.global_step}]"))
            rank_zero_print(cyan(line))

        # =============== Logging ===============
        total_loss = (
            self.dfot_loss_weight * dfot_loss
            + self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
        )
        if is_train and batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/dfot_loss", dfot_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/jepa_loss", jepa_loss.detach(), on_step=True, sync_dist=True)
            self.log(f"{namespace}/decoder_loss", decoder_loss.detach(), on_step=True, sync_dist=True)
            for key, value in jepa_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)
            for key, value in decoder_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": total_loss.detach(),
            "dfot_loss": dfot_loss.detach(),
            "jepa_loss": jepa_loss.detach(),
            "xs_pred": xs_pred,
            "xs": xs,
        }

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EMA update + decoder visualization after DDP sync is complete."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (self.global_step + 1) % self.ema_update_every == 0:
            self._ema_update_target()
            self._refresh_data_stats()

        decoder_vis_every = self.jepa_cfg.get("decoder_vis_every", 50)
        if (
            self.trainer.world_size == 1
            and self.jepa_cfg.get("decoder_vis_enabled", True)
            and self.global_step % decoder_vis_every == 0
        ):
            _, _, _, gt_videos, actions_raw = batch
            if gt_videos is not None:
                self._log_decoder_reconstructions(
                    gt_videos, namespace="decoder_vis_train", step=self.global_step
                )
                self._log_jepa_predictions(
                    gt_videos, actions_raw, namespace="jepa_pred_vis_train", step=self.global_step
                )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step -- convert 5-element batch for parent methods.

        Mirrors the working reference: call _eval_denoising, sample videos,
        update metrics, log videos.  No rank-asymmetric work here -- all
        sync points come from gather_data inside the parent helpers.
        """
        torch.cuda.empty_cache()

        xs, conditions, masks, gt_videos, actions_raw = batch
        parent_batch = (xs, conditions, masks, gt_videos)

        if self.trainer.state.fn == "FIT":
            self._eval_denoising_jepa(parent_batch, batch_idx, namespace=namespace)
            torch.cuda.empty_cache()

        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):
            all_videos = self._sample_all_videos(parent_batch, batch_idx, namespace)
            if all_videos is not None:
                self._update_metrics(all_videos)
                self._log_videos(all_videos, namespace)
                del all_videos
                torch.cuda.empty_cache()

    def on_validation_epoch_end(self, namespace="validation") -> None:
        torch.cuda.empty_cache()
        super().on_validation_epoch_end(namespace)

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate denoising -- adapted for the 5-element batch.

        Uses gather_data (barrier + allgather) like the parent _eval_denoising
        so all ranks stay synchronized.
        """
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
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            num_videos_to_log = min(
                self.logging.max_num_videos - self.num_logged_videos,
                gt_videos_vis.shape[0],
            )
            log_video(
                recons[:num_videos_to_log].float(),
                gt_videos_vis[:num_videos_to_log].float(),
                step=self.global_step,
                namespace="denoising_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                captions="denoised | gt",
            )

        # Barrier so all ranks proceed to _sample_all_videos together
        if self.trainer.world_size > 1:
            torch.distributed.barrier()

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key: str) -> bool:
        """Include JEPA components, online encoder, EMA-updated target VAE, and running stats."""
        base_include = super()._should_include_in_checkpoint(key)
        jepa_include = (
            key.startswith("action_encoder")
            or key.startswith("predictor")
            or key.startswith("online_encoder")
            or key.startswith("online_quant_conv")
            or key.startswith("online_post_quant_conv")
            or key.startswith("online_decoder")
            # EMA-updated target VAE: must be saved so the target encoder/decoder
            # can be restored to their drifted state on checkpoint resume.
            or key.startswith("vae.encoder.")
            or key.startswith("vae.quant_conv.")
            or key.startswith("vae.post_quant_conv.")
            or key.startswith("vae.decoder.")
            # Running normalization stats (registered buffers)
            or key.startswith("_running_")
        )
        return base_include or jepa_include

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["jepa_cfg"] = self.jepa_cfg

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        # Guard against scheduler state incompatibility (DFoT ckpt has 1
        # group, JEPA has 4).
        expected_param_groups = 4
        ckpt_schedulers = checkpoint.get("lr_schedulers", None)
        if isinstance(ckpt_schedulers, list) and len(ckpt_schedulers) > 0:
            first_sched = ckpt_schedulers[0]
            if isinstance(first_sched, dict):
                sched_state = first_sched.get("state_dict", first_sched)
                if isinstance(sched_state, dict):
                    base_lrs = sched_state.get("base_lrs", None)
                    if (
                        isinstance(base_lrs, list)
                        and len(base_lrs) != expected_param_groups
                    ):
                        rank_zero_print(
                            cyan(
                                "Ignoring incompatible lr_schedulers state from "
                                f"checkpoint (ckpt groups={len(base_lrs)}, "
                                f"expected={expected_param_groups})."
                            )
                        )
                        checkpoint["lr_schedulers"] = []

        # Report which JEPA keys were loaded vs randomly initialized
        loaded_keys = set(checkpoint.get("state_dict", {}).keys())
        jepa_keys = [
            k
            for k in self.state_dict().keys()
            if k.startswith("action_encoder")
            or k.startswith("predictor")
            or k.startswith("online_encoder")
            or k.startswith("online_quant_conv")
            or k.startswith("online_post_quant_conv")
            or k.startswith("online_decoder")
        ]
        loaded_jepa = [k for k in jepa_keys if k in loaded_keys]
        new_jepa = [k for k in jepa_keys if k not in loaded_keys]

        if loaded_jepa:
            rank_zero_print(cyan(f"Loaded JEPA weights: {len(loaded_jepa)} parameters"))
        if new_jepa:
            rank_zero_print(
                cyan(f"Randomly initialized JEPA weights: {len(new_jepa)} parameters")
            )
            rank_zero_print(
                cyan("  (This is expected when finetuning from pre-trained DFoT)")
            )

        ckpt_state = checkpoint.get("state_dict", {})
        vae_in_ckpt = any(k.startswith("vae.encoder.") for k in ckpt_state)
        if vae_in_ckpt:
            rank_zero_print(cyan("Target VAE weights will be restored from checkpoint (EMA-updated state)."))
        else:
            rank_zero_print(cyan(f"VAE will be loaded from: {self.cfg.vae.pretrained_path}"))
