"""
Dual-Encoder JEPA for DFoT (BYOL-Style).

Architecture:
- **Target encoder** (frozen): Provides stable latents for DFoT diffusion.
  The data_mean / data_std normalization stays valid.
- **Online encoder** (trainable): A copy of the VAE encoder that receives
  JEPA gradients.  Learns to produce latents that are inherently predictive
  of future states.
- **ViT Predictor**: Operates on *clean* latent states from the online
  encoder and encoded actions; predicts future clean latent states.
- **EMA sync**: The online encoder is slowly blended into the target
  encoder (like BYOL / DINO) so DFoT gradually benefits from improved
  representations without sudden distribution shifts.

Gradient paths:
  DFoT loss  -->  diffusion backbone only
  JEPA loss  -->  online encoder + predictor + action encoder
"""

from copy import deepcopy
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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        dots = dots.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
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
        self.jepa_training_mode = cfg.jepa.get("training_mode", "teacher_forcing")

        # EMA config
        self.ema_decay = cfg.jepa.get("ema_decay", 0.999)
        self.ema_update_every = cfg.jepa.get("ema_update_every", 100)

        # Force online latent mode
        assert cfg.latent.enable, "JEPA training requires latent diffusion"
        assert cfg.latent.type == "online", "JEPA training requires online latent processing"

        super().__init__(cfg)

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_model(self):
        """Build DFoT model, then build JEPA components."""
        super()._build_model()
        self._build_jepa_model()

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

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        # Online encoder forward
        h = self.online_encoder(x_normalized)
        moments = self.online_quant_conv(h)
        # Split into mean and logvar, take mode (= mean)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latents_flat = mean  # mode() of DiagonalGaussian = mean

        return latents_flat.view(B, T, *latents_flat.shape[1:])

    def _encode_target(self, videos: Tensor) -> Tensor:
        """
        Encode raw videos with the frozen target encoder.
        Uses mode() for deterministic, clean latents.

        Args:
            videos: (B, T, 3, H, W) in [0, 1]
        Returns:
            latents: (B, T, C, H, W)
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_normalized = 2.0 * x_flat - 1.0

        with torch.no_grad():
            posterior = self.vae.encode(x_normalized)
            latents_flat = posterior.mode()

        return latents_flat.view(B, T, *latents_flat.shape[1:])

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
        self, videos: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Diagnostic: verify online encoder hasn't collapsed by measuring how
        well a trainable decoder can reconstruct pixels from its latents.

        Pipeline:
        1. Encode frames with online encoder  (gradients flow through *decoder*)
        2. DETACH the latent z               (no gradient to encoder)
        3. Decode with trainable decoder     (gradients flow through decoder only)
        4. Loss = mse_weight * MSE + lpips_weight * LPIPS

        Args:
            videos: (B, T, 3, H, W) raw frames in [0, 1]
        Returns:
            decoder_loss: scalar
            log_dict: dict of per-component losses
        """
        B, T = videos.shape[:2]
        x_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        x_norm = 2.0 * x_flat - 1.0  # [-1, 1]

        # Encode (no grad to encoder via detach below)
        h = self.online_encoder(x_norm)
        moments = self.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        z = mean.detach()  # <-- gradient wall: encoder receives no signal

        # Decode
        x_recon = self._decode_online(z)  # (B*T, 3, H, W) in [-1, 1]

        mse_loss = F.mse_loss(x_recon, x_norm)
        lpips_loss = self.perceptual_loss(x_recon, x_norm).mean()

        mse_w = self.jepa_cfg.get("decoder_mse_weight", 1.0)
        lpips_w = self.jepa_cfg.get("decoder_lpips_weight", 1.0)
        decoder_loss = mse_w * mse_loss + lpips_w * lpips_loss

        log_dict = {
            "decoder/mse_loss": mse_loss.detach(),
            "decoder/lpips_loss": lpips_loss.detach(),
            "decoder/total_loss": decoder_loss.detach(),
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

        Only runs on rank 0 and skips sanity-checking epochs.
        """
        import os
        import torchvision
        from utils.distributed_utils import is_rank_zero
        from utils.logging_utils import log_video

        if not is_rank_zero:
            return
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
        recon_target = ((recon_target.clamp(-1, 1) + 1) / 2).view(1, T, *recon_target.shape[1:])

        # Online encoder (trainable) + online decoder (trainable)
        h = self.online_encoder(x_norm)
        moments = self.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        recon_online = self._decode_online(mean)  # [-1, 1]
        recon_online = ((recon_online.clamp(-1, 1) + 1) / 2).view(1, T, *recon_online.shape[1:])

        # Cast to float32: numpy / torchvision don't support bfloat16
        recon_target = recon_target.float()
        recon_online = recon_online.float()
        videos = videos.float()

        # --- 1. Disk: PNG grid (T rows × 3 cols = target | online | gt) ---
        log_dir = self.trainer.log_dir or "."
        save_dir = os.path.join(log_dir, "decoder_vis", namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")
        # Interleave: (T, 3_sources, C, H, W) -> (3T, C, H, W), make_grid with nrow=3
        frames = torch.stack(
            [recon_target[0], recon_online[0], videos[0]], dim=1
        ).view(-1, *videos.shape[2:])  # (3T, C, H, W)
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

    # -----------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target_encoder(self) -> None:
        """
        Exponential moving average update: blend online encoder weights
        into the target encoder.  Called every ``ema_update_every`` steps.
        """
        decay = self.ema_decay
        for p_online, p_target in zip(
            self.online_encoder.parameters(), self.vae.encoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

        for p_online, p_target in zip(
            self.online_quant_conv.parameters(), self.vae.quant_conv.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

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

        # DFoT path: encode with frozen target VAE (via parent's _encode)
        if self.is_latent_diffusion and self.is_latent_online:
            xs = self._encode(batch["videos"])
        else:
            xs = batch.get("latents", batch["videos"])

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
        videos: Tensor,
        actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute JEPA prediction loss on clean latents from the online encoder.

        Pipeline:
        1. Encode all frames with the online encoder -> clean latents
        2. Flatten to state vectors
        3. Encode actions
        4. Predictor: (states[:-1], actions[:-1]) -> pred_states
        5. Loss = smooth_l1(pred_states, states[1:].detach())
        """
        B, T = videos.shape[:2]

        if T < 2:
            return torch.tensor(0.0, device=videos.device), {}

        # 1. Encode with online encoder (gradients flow through)
        online_latents = self._encode_online(videos)  # (B, T, C, H, W)

        # 2. Flatten
        states = online_latents.view(B, T, -1)  # (B, T, state_dim)

        # 3. Encode actions
        action_embeds = self.action_encoder(actions)  # (B, T, action_embed_dim)

        # 4. Inputs and targets
        states_input = states[:, :-1]             # (B, T-1, state_dim)
        action_embeds_input = action_embeds[:, :-1]  # (B, T-1, action_embed_dim)
        states_target = states[:, 1:].detach()    # (B, T-1, state_dim) -- detached!

        # 5. Predict
        if self.jepa_training_mode == "autoregressive":
            pred_list = []
            current = states[:, 0:1]
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

        # 6. Loss
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
                mse = torch.tensor(0.0, device=videos.device)
                cos_sim = torch.tensor(0.0, device=videos.device)

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
        """Training step: DFoT loss + JEPA loss, then EMA update."""
        xs, conditions, masks, gt_videos, actions_raw = batch

        # =============== DFoT Loss (uses target encoder latents) ===============
        noise_levels, masks_dfot = self._get_training_noise_levels(xs, masks)
        xs_pred, dfot_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        dfot_loss = self._reweight_loss(dfot_loss, masks_dfot)

        # =============== JEPA Loss (uses online encoder latents) ===============
        jepa_loss = torch.tensor(0.0, device=xs.device)
        jepa_log_dict: Dict[str, Tensor] = {}

        if (
            gt_videos is not None
            and actions_raw is not None
            and self.jepa_loss_weight > 0
        ):
            jepa_loss, jepa_log_dict = self._compute_jepa_loss(
                gt_videos, actions_raw, masks,
            )

        # =============== Decoder Reconstruction Loss (collapse diagnostic) ===============
        decoder_loss = torch.tensor(0.0, device=xs.device)
        decoder_log_dict: Dict[str, Tensor] = {}

        decoder_loss_weight = self.jepa_cfg.get("decoder_loss_weight", 0.0)
        if gt_videos is not None and decoder_loss_weight > 0:
            decoder_loss, decoder_log_dict = self._compute_decoder_loss(gt_videos)

        # =============== Combined Loss ===============
        total_loss = (
            dfot_loss
            + self.jepa_loss_weight * jepa_loss
            + decoder_loss_weight * decoder_loss
        )

        # =============== Logging ===============
        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/dfot_loss", dfot_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/jepa_loss", jepa_loss, on_step=True, sync_dist=True)
            self.log(f"{namespace}/decoder_loss", decoder_loss, on_step=True, sync_dist=True)
            for key, value in jepa_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)
            for key, value in decoder_log_dict.items():
                self.log(f"{namespace}/{key}", value, on_step=True, sync_dist=True)

        # =============== Decoder Visualization ===============
        decoder_vis_every = self.jepa_cfg.get("decoder_vis_every", 50)
        if gt_videos is not None and self.global_step % decoder_vis_every == 0:
            self._log_decoder_reconstructions(
                gt_videos, namespace="decoder_vis_train", step=self.global_step
            )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": total_loss,
            "dfot_loss": dfot_loss,
            "jepa_loss": jepa_loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EMA update of target encoder after each training step (if due)."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (self.global_step + 1) % self.ema_update_every == 0:
            self._ema_update_target_encoder()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step -- convert 5-element batch for parent methods."""
        xs, conditions, masks, gt_videos, actions_raw = batch
        parent_batch = (xs, conditions, masks, gt_videos)

        if self.trainer.state.fn == "FIT":
            self._eval_denoising_jepa(parent_batch, batch_idx, namespace=namespace)

        # Decoder reconstruction visualization (once per validation epoch)
        if gt_videos is not None and batch_idx == 0:
            self._log_decoder_reconstructions(
                gt_videos, namespace="decoder_vis_val", step=self.global_step
            )

        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):
            all_videos = self._sample_all_videos(parent_batch, batch_idx, namespace)
            if all_videos is not None:
                self._update_metrics(all_videos)
                self._log_videos(all_videos, namespace)

    def _eval_denoising_jepa(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate denoising -- adapted for the 5-element batch."""
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

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _should_include_in_checkpoint(self, key: str) -> bool:
        """Include JEPA components and online encoder in checkpoint."""
        base_include = super()._should_include_in_checkpoint(key)
        jepa_include = (
            key.startswith("action_encoder")
            or key.startswith("predictor")
            or key.startswith("online_encoder")
            or key.startswith("online_quant_conv")
            or key.startswith("online_post_quant_conv")
            or key.startswith("online_decoder")
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

        rank_zero_print(cyan(f"VAE will be loaded from: {self.cfg.vae.pretrained_path}"))
