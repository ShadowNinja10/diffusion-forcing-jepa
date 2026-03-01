#!/usr/bin/env python3
"""
Render decoder/JEPA visualizations for a folder of checkpoints and log to W&B.

Supports both dfot_video_jepa (plain) and dfot_video_jepa_spatial architectures.

Example (plain JEPA):
  python log_decoder_vis_ckpts.py \
    --checkpoint-dir /path/to/checkpoints \
    --wandb-entity jsc9903-new-york-university \
    --wandb-project dfot-jepa \
    --wandb-run-name jepa-retro-vis \
    --num-videos 10

Example (spatial JEPA):
  python log_decoder_vis_ckpts.py \
    --checkpoint-dir /scratch/jsc9903/RLFM/diffusion-forcing-jepa/outputs/2026-02-28/07-37-47/checkpoints \
    --algorithm dfot_video_jepa_spatial \
    --wandb-entity jsc9903-new-york-university \
    --wandb-project dfot-jepa \
    --wandb-run-name jepa-spatial-retro-vis \
    --num-videos 10

"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import wandb
from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from experiments import build_experiment
from utils.hydra_utils import unwrap_shortcuts


def _torch_26_weights_only_compat() -> None:
    try:
        safe_types = []
        from omegaconf.base import ContainerMetadata

        safe_types.append(ContainerMetadata)
        from omegaconf import DictConfig, ListConfig

        safe_types.extend([DictConfig, ListConfig])
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            add_safe_globals(safe_types)
    except Exception:
        return


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x


def _parse_step(path: Path) -> int:
    m = re.search(r"step=(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _parse_epoch(path: Path) -> int:
    m = re.search(r"epoch=(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _videos_to_uint8(video: torch.Tensor) -> np.ndarray:
    # video: (T, C, H, W) in [0, 1]
    return (video.clamp(0, 1).mul(255).byte().cpu().numpy())


def _concat_panel(videos: list[torch.Tensor]) -> np.ndarray:
    # videos: list[(T, C, H, W)] in [0, 1]
    panel = torch.cat(videos, dim=-1)  # concat width
    return _videos_to_uint8(panel)


def _annotate_video_uint8(video: np.ndarray, label: str) -> np.ndarray:
    # video: (T, C, H, W) uint8
    out = np.empty_like(video)
    for t in range(video.shape[0]):
        frame = np.transpose(video[t], (1, 2, 0))  # (H, W, C)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        box_w = min(img.width - 1, max(90, 8 * len(label) + 12))
        draw.rectangle([(0, 0), (box_w, 20)], fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255))
        out[t] = np.transpose(np.asarray(img), (2, 0, 1))
    return out


def _concat_labeled_panel(videos: list[torch.Tensor], labels: list[str]) -> np.ndarray:
    if len(videos) != len(labels):
        raise ValueError("videos and labels must have the same length")
    chunks = []
    for video, label in zip(videos, labels):
        v_uint8 = _videos_to_uint8(video)
        chunks.append(_annotate_video_uint8(v_uint8, label))
    return np.concatenate(chunks, axis=3)  # concat width


def _parse_video_indices(indices_arg: str) -> list[int]:
    if not indices_arg.strip():
        return []
    indices = []
    for item in indices_arg.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            raise ValueError(f"video index must be >= 0, got {idx}")
        indices.append(idx)
    if not indices:
        raise ValueError("No valid values parsed from --video-indices")
    return sorted(set(indices))


def _collect_samples(loader, device: torch.device, required_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    videos_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    for raw_batch in loader:
        raw_batch = _to_device(raw_batch, device)
        videos = raw_batch.get("videos", None)
        actions = raw_batch.get("conds", None)
        if videos is None or actions is None:
            raise RuntimeError("Batch must contain 'videos' and 'conds' for JEPA decoder visualization.")
        for i in range(videos.shape[0]):
            videos_list.append(videos[i].detach())
            actions_list.append(actions[i].detach())
            if len(videos_list) >= required_count:
                return torch.stack(videos_list, dim=0), torch.stack(actions_list, dim=0)
    raise RuntimeError(
        f"Requested {required_count} sample(s), but dataloader only provided {len(videos_list)} sample(s)."
    )


@torch.no_grad()
def _compute_decoder_recons(
    model, gt_videos: torch.Tensor, vae_chunk_size: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # gt_videos: (B, T, 3, H, W) in [0, 1]
    # Process in chunks to avoid OOM with many videos/frames.
    b, t = gt_videos.shape[:2]
    x_flat = rearrange(gt_videos, "b t c h w -> (b t) c h w")
    x_norm = 2.0 * x_flat - 1.0

    vae_bs = vae_chunk_size or getattr(model.cfg.vae, "batch_size", 4)
    vae_bs = max(1, min(vae_bs, 16))
    total = x_norm.shape[0]

    target_chunks: list[torch.Tensor] = []
    online_chunks: list[torch.Tensor] = []
    for start in range(0, total, vae_bs):
        chunk = x_norm[start : start + vae_bs]
        posterior = model.vae.encode(chunk)
        z_target = posterior.mode()
        recon_target = model.vae.decode(z_target)
        target_chunks.append((recon_target.clamp(-1, 1) + 1) / 2)

        h = model.online_encoder(chunk)
        moments = model.online_quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        recon_online = model._decode_online(mean)
        online_chunks.append((recon_online.clamp(-1, 1) + 1) / 2)

    recon_target = torch.cat(target_chunks, dim=0).reshape(b, t, *target_chunks[0].shape[1:]).float()
    recon_online = torch.cat(online_chunks, dim=0).reshape(b, t, *online_chunks[0].shape[1:]).float()
    return recon_target, recon_online


def _is_spatial_jepa(model) -> bool:
    """True if model uses SpatialTemporalPredictor (dfot_video_jepa_spatial)."""
    return hasattr(model.predictor, "_patchify")


@torch.no_grad()
def _compute_jepa_pred_recons(
    model, gt_videos: torch.Tensor, actions: torch.Tensor, decoder_chunk_size: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns pred_recon, gt_aligned in [0, 1], shape (B, T-1, 3, H, W)
    b, t = gt_videos.shape[:2]
    if t < 2:
        raise ValueError("Need at least 2 frames for JEPA prediction visualization.")

    online_latents = model._encode_online(gt_videos)
    latent_shape = online_latents.shape[2:]
    action_embeds = model.action_encoder(actions)

    if _is_spatial_jepa(model):
        # Spatial predictor: (B, T-1, C, H, W) latents + actions -> (B, T-1, C, H, W)
        pred_latents = model.predictor(online_latents[:, :-1], action_embeds[:, :-1])
    else:
        # Plain JEPA: flattened states (B, T-1, state_dim) -> (B, T-1, state_dim)
        online_states = online_latents.reshape(b, t, -1)
        states_pred = model.predictor(online_states[:, :-1], action_embeds[:, :-1])
        pred_latents = states_pred.reshape(b * (t - 1), *latent_shape)

    pred_flat = pred_latents.reshape(b * (t - 1), *latent_shape)
    # Decode in chunks to avoid OOM
    dec_bs = decoder_chunk_size or (
        model.jepa_cfg.get("decoder_chunk_size", 4) if hasattr(model, "jepa_cfg") else 4
    )
    dec_bs = max(1, min(dec_bs, 16))
    total = pred_flat.shape[0]
    recon_chunks: list[torch.Tensor] = []
    for start in range(0, total, dec_bs):
        chunk = pred_flat[start : start + dec_bs]
        recon = model._decode_online(chunk)
        recon_chunks.append((recon.clamp(-1, 1) + 1) / 2)
    pred_recon = torch.cat(recon_chunks, dim=0).reshape(b, t - 1, *recon_chunks[0].shape[1:]).float()
    gt_aligned = gt_videos[:, 1:].float()
    return pred_recon, gt_aligned


def _build_cfg(project_root: Path, extra_overrides: list[str], algorithm: str = "dfot_video_jepa"):
    cfg_dir = project_root / "configurations"
    cfg_dir_str = str(cfg_dir)
    dataset_experiment = (
        "minecraft_video_generation_jepa_spatial"
        if algorithm == "dfot_video_jepa_spatial"
        else "minecraft_video_generation_jepa"
    )
    overrides = [
        "experiment=video_generation",
        "dataset=minecraft",
        f"algorithm={algorithm}",
        f"dataset_experiment={dataset_experiment}",
        "@DiT/B",
        "@diffusion/continuous",
        "algorithm.checkpoint.strict=false",
        "experiment.training.batch_size=1",
        "experiment.validation.batch_size=1",
        "algorithm.vae.batch_size=1",
        "dataset.max_frames=12",
        "dataset.context_length=6",
        "dataset.n_frames=12",
        "dataset.num_eval_videos=50",
        "experiment.validation.limit_batch=1.0",
        "experiment.find_unused_parameters=true",
        "algorithm.jepa.recon_regularizer=true",
        "wandb.mode=disabled",
    ]
    overrides.extend(extra_overrides)

    argv = ["script.py", *overrides]
    hydra_overrides = unwrap_shortcuts(argv, config_path=cfg_dir_str, config_name="config")[1:]

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="config", overrides=hydra_overrides)

    def _pick(name: str, default: str) -> str:
        for item in overrides:
            if item.startswith(f"{name}="):
                return item.split("=", 1)[1]
        return default

    with open_dict(cfg):
        cfg.experiment._name = _pick("experiment", "video_generation")
        cfg.dataset._name = _pick("dataset", "minecraft")
        cfg.algorithm._name = _pick("algorithm", "dfot_video_jepa")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wandb-entity", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-run-name", type=str, default="decoder-vis-retro")
    parser.add_argument("--wandb-tags", nargs="*", default=["retro-decoder-vis"])
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument(
        "--video-indices",
        type=str,
        default="",
        help="Comma-separated global sample indices from the loader, e.g. 0,5,11. Overrides --num-videos.",
    )
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "training"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dfot_video_jepa",
        choices=["dfot_video_jepa", "dfot_video_jepa_spatial"],
        help="Algorithm architecture (plain JEPA vs spatial patch-token JEPA)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vae-chunk-size",
        type=int,
        default=0,
        help="Max frames per VAE encode/decode chunk (0=use config). Reduce if OOM.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("decoder_vis"))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional hydra override, repeatable. Example: --override algorithm.jepa.decoder_lr=5e-5",
    )
    args = parser.parse_args()

    _torch_26_weights_only_compat()

    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    ckpts = sorted(args.checkpoint_dir.glob("*.ckpt"), key=_parse_step)
    if args.max_checkpoints > 0:
        ckpts = ckpts[: args.max_checkpoints]
    if not ckpts:
        raise RuntimeError(f"No .ckpt files found in {args.checkpoint_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent
    cfg = _build_cfg(project_root, args.override, algorithm=args.algorithm)

    exp = build_experiment(cfg, logger=None, ckpt_path=None)
    model = exp._build_algo().to(device).eval()
    dm = exp.data_module
    loader = dm.val_dataloader() if args.split == "validation" else dm.train_dataloader()

    selected_indices = _parse_video_indices(args.video_indices)
    required_count = (max(selected_indices) + 1) if selected_indices else args.num_videos
    all_videos, all_actions = _collect_samples(loader, device=device, required_count=required_count)
    if selected_indices:
        gt_videos = all_videos[selected_indices]
        actions = all_actions[selected_indices]
        sample_ids = selected_indices
    else:
        gt_videos = all_videos[: args.num_videos]
        actions = all_actions[: args.num_videos]
        sample_ids = list(range(args.num_videos))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config={
            "checkpoint_dir": str(args.checkpoint_dir),
            "algorithm": args.algorithm,
            "num_videos": args.num_videos,
            "video_indices": sample_ids,
            "split": args.split,
            "device": str(device),
            "overrides": args.override,
        },
        reinit=True,
    )

    try:
        for idx, ckpt_path in enumerate(ckpts, start=1):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.on_load_checkpoint(ckpt)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            model.eval()

            step = int(ckpt.get("global_step", _parse_step(ckpt_path)))
            epoch = int(ckpt.get("epoch", _parse_epoch(ckpt_path)))

            vae_chunk = args.vae_chunk_size or 0
            recon_target, recon_online = _compute_decoder_recons(
                model, gt_videos, vae_chunk_size=vae_chunk
            )
            pred_recon, gt_aligned = _compute_jepa_pred_recons(
                model, gt_videos, actions, decoder_chunk_size=vae_chunk
            )

            log_dict: dict[str, Any] = {
                "ckpt/index": idx,
                "ckpt/epoch": epoch,
                "ckpt/step": step,
            }

            for v in range(recon_target.shape[0]):
                sample_id = sample_ids[v]
                decoder_panel = _concat_labeled_panel(
                    [recon_target[v], recon_online[v], gt_videos[v]],
                    ["target_dec", "online_dec", "gt"],
                )
                pred_panel = _concat_labeled_panel(
                    [pred_recon[v], gt_aligned[v]],
                    ["jepa_pred", "gt_next"],
                )

                # Save local GIF snapshots for quick spot-checking.
                dec_path = out_dir / f"step_{step:07d}_sample{sample_id}_decoder.gif"
                pred_path = out_dir / f"step_{step:07d}_sample{sample_id}_pred.gif"
                imageio.mimsave(dec_path, [np.transpose(f, (1, 2, 0)) for f in decoder_panel], fps=8)
                imageio.mimsave(pred_path, [np.transpose(f, (1, 2, 0)) for f in pred_panel], fps=8)

                log_dict[f"retro/decoder_recon_sample_{sample_id}"] = wandb.Video(
                    decoder_panel,
                    fps=8,
                    format="gif",
                    caption=f"{ckpt_path.name} | sample={sample_id} | target_dec | online_dec | gt",
                )
                log_dict[f"retro/jepa_pred_sample_{sample_id}"] = wandb.Video(
                    pred_panel,
                    fps=8,
                    format="gif",
                    caption=f"{ckpt_path.name} | sample={sample_id} | jepa_pred | gt_next",
                )

            wandb.log(log_dict, step=step)
            print(f"[{idx}/{len(ckpts)}] logged {ckpt_path.name} at step={step}, epoch={epoch}")

            del ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        run.finish()


if __name__ == "__main__":
    main()
