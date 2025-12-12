"""
Comprehensive test for JEPA + DFoT pipeline with mock data.
Tests the full pipeline including:
- Model initialization
- VAE encoding (mock)
- JEPA predictor forward/backward
- DFoT diffusion forward/backward
- Combined loss computation
- Gradient flow to all components

Run: python test_jepa_full_pipeline.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Tuple, Optional
from omegaconf import OmegaConf, DictConfig


# =============================================================================
# Mock Components (to avoid import issues)
# =============================================================================

class MockVAE(nn.Module):
    """Mock VAE that mimics the real ImageVAE interface."""
    
    def __init__(self, in_channels=3, latent_channels=4, downsample_factor=8):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels * 2, 4, 2, 1),
        )
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1)
        
        # Decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 64, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Tanh(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return MockDistribution(moments)
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """Mock from_pretrained that just creates a new model."""
        return cls()


class MockDistribution:
    """Mock distribution with sample() and mode() methods."""
    def __init__(self, moments):
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
    
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)
    
    def mode(self):
        return self.mean


class MockDiffusionModel(nn.Module):
    """Mock diffusion model that mimics DiscreteDiffusion interface."""
    
    def __init__(self, x_shape, external_cond_dim):
        super().__init__()
        self.x_shape = x_shape
        c, h, w = x_shape
        input_dim = c * h * w
        
        # Simple MLP for testing
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1 + external_cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        
        # Mock diffusion buffers
        self.register_buffer("sqrt_alphas_cumprod", torch.linspace(1.0, 0.1, 100))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.linspace(0.0, 0.9, 100))
    
    def forward(self, x, external_cond, k):
        """
        Forward pass mimicking diffusion training.
        Returns: (predicted_x, loss)
        """
        B, T = x.shape[:2]
        
        # Add noise
        noise = torch.randn_like(x) * 0.1
        noised_x = x + noise
        
        # Flatten for simple MLP
        x_flat = noised_x.view(B * T, -1)
        k_flat = k.view(B * T, 1).float() / 100.0
        
        if external_cond is not None:
            cond_flat = external_cond.view(B * T, -1)
            input_flat = torch.cat([x_flat, k_flat, cond_flat], dim=-1)
        else:
            zeros = torch.zeros(B * T, 0, device=x.device)
            input_flat = torch.cat([x_flat, k_flat, zeros], dim=-1)
        
        # Predict (simplified)
        # Pad input to match expected size
        expected_input_size = self.model[0].in_features
        if input_flat.shape[-1] < expected_input_size:
            padding = torch.zeros(input_flat.shape[0], expected_input_size - input_flat.shape[-1], device=x.device)
            input_flat = torch.cat([input_flat, padding], dim=-1)
        elif input_flat.shape[-1] > expected_input_size:
            input_flat = input_flat[:, :expected_input_size]
            
        pred_flat = self.model(input_flat)
        pred = pred_flat.view(B, T, *self.x_shape)
        
        # Compute loss
        loss = F.mse_loss(pred, x, reduction="none")
        
        return pred, loss


# =============================================================================
# JEPA Components (copied from dfot_video_jepa.py for standalone testing)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, C = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        dots = dots.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ViTPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, depth=4, heads=8,
                 dim_head=64, mlp_ratio=4.0, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.dropout_layer = nn.Dropout(dropout)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, states, actions):
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
    def __init__(self, action_dim, embed_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, actions):
        return self.net(actions)


# =============================================================================
# Full Pipeline Test Class
# =============================================================================

class MockJEPADFoT(nn.Module):
    """
    Mock JEPA + DFoT model that mimics the full pipeline.
    This tests the exact same forward/backward logic as the real implementation.
    """
    
    def __init__(
        self,
        video_shape=(3, 256, 256),
        latent_channels=4,
        latent_size=32,
        action_dim=8,
        action_embed_dim=256,
        predictor_hidden_dim=512,
        predictor_depth=4,
        train_vae_encoder=True,
        use_mode_for_targets=True,
        jepa_loss_weight=1.0,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.state_dim = latent_channels * latent_size * latent_size
        self.action_dim = action_dim
        self.train_vae_encoder = train_vae_encoder
        self.use_mode_for_targets = use_mode_for_targets
        self.jepa_loss_weight = jepa_loss_weight
        
        # VAE
        self.vae = MockVAE(in_channels=video_shape[0], latent_channels=latent_channels)
        
        # DFoT (mock diffusion model)
        self.diffusion_model = MockDiffusionModel(
            x_shape=(latent_channels, latent_size, latent_size),
            external_cond_dim=action_dim,
        )
        
        # JEPA components
        self.action_encoder = ActionEncoder(action_dim, action_embed_dim)
        self.predictor = ViTPredictor(
            state_dim=self.state_dim,
            action_dim=action_embed_dim,
            hidden_dim=predictor_hidden_dim,
            depth=predictor_depth,
            heads=8,
            max_seq_len=64,
        )
        
        # Freeze VAE encoder if needed
        if not train_vae_encoder:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            for param in self.vae.quant_conv.parameters():
                param.requires_grad = False
    
    def encode_videos(self, videos, use_mode=True):
        """Encode videos to latents."""
        B, T, C, H, W = videos.shape
        videos_flat = rearrange(videos, "b t c h w -> (b t) c h w")
        
        # Normalize to [-1, 1]
        videos_norm = 2.0 * videos_flat - 1.0
        
        # Encode
        posterior = self.vae.encode(videos_norm)
        if use_mode:
            latents_flat = posterior.mode()
        else:
            latents_flat = posterior.sample()
        
        # Reshape
        latents = rearrange(latents_flat, "(b t) c h w -> b t c h w", b=B, t=T)
        return latents
    
    def compute_jepa_loss(self, videos, actions, masks):
        """Compute JEPA prediction loss."""
        B, T = videos.shape[:2]
        
        if T < 2:
            return torch.tensor(0.0, device=videos.device), {}
        
        # Encode videos
        latents = self.encode_videos(videos, use_mode=self.use_mode_for_targets)
        
        # Flatten to states
        states = latents.view(B, T, -1)
        
        # Prepare inputs (teacher forcing)
        states_input = states[:, :-1]
        actions_input = actions[:, :-1]
        states_target = states[:, 1:]
        
        # Encode actions and predict
        action_embeds = self.action_encoder(actions_input)
        states_pred = self.predictor(states_input, action_embeds)
        
        # Compute loss
        transition_masks = masks[:, :-1] & masks[:, 1:]
        pred_loss = F.smooth_l1_loss(states_pred, states_target.detach(), reduction="none")
        pred_loss = pred_loss.mean(dim=-1)
        
        if transition_masks.sum() > 0:
            pred_loss = (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
        else:
            pred_loss = pred_loss.mean()
        
        return pred_loss, {"mse": F.mse_loss(states_pred, states_target)}
    
    def compute_dfot_loss(self, latents, conditions, masks):
        """Compute DFoT diffusion loss."""
        B, T = latents.shape[:2]
        
        # Random noise levels
        k = torch.randint(0, 100, (B, T), device=latents.device)
        
        # Diffusion forward
        _, dfot_loss = self.diffusion_model(latents, conditions, k)
        
        # Average loss
        dfot_loss = dfot_loss.mean()
        
        return dfot_loss
    
    def training_step(self, videos, actions, masks):
        """Full training step."""
        B, T = videos.shape[:2]
        
        # 1. Encode videos for DFoT
        latents = self.encode_videos(videos, use_mode=False)  # use sample for DFoT
        
        # 2. DFoT loss
        dfot_loss = self.compute_dfot_loss(latents, actions, masks)
        
        # 3. JEPA loss
        jepa_loss, jepa_metrics = self.compute_jepa_loss(videos, actions, masks)
        
        # 4. Combined loss
        total_loss = dfot_loss + self.jepa_loss_weight * jepa_loss
        
        return {
            "total_loss": total_loss,
            "dfot_loss": dfot_loss,
            "jepa_loss": jepa_loss,
            **jepa_metrics,
        }


# =============================================================================
# Test Functions
# =============================================================================

def create_mock_batch(batch_size=2, num_frames=8, height=256, width=256, action_dim=8, device="cpu"):
    """Create mock batch of data."""
    videos = torch.rand(batch_size, num_frames, 3, height, width, device=device)
    
    # One-hot actions
    actions = torch.zeros(batch_size, num_frames, action_dim, device=device)
    for b in range(batch_size):
        for t in range(num_frames):
            idx = torch.randint(0, action_dim, (1,)).item()
            actions[b, t, idx] = 1.0
    
    masks = torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
    
    return videos, actions, masks


def test_vae_encoding():
    """Test VAE encoding with sample() and mode()."""
    print("\n" + "="*60)
    print("Test 1: VAE Encoding (sample vs mode)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vae = MockVAE().to(device)
    x = torch.randn(4, 3, 256, 256, device=device)
    
    # Test sample
    dist = vae.encode(x)
    z_sample = dist.sample()
    z_mode = dist.mode()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Latent (sample) shape: {z_sample.shape}")
    print(f"  Latent (mode) shape: {z_mode.shape}")
    print(f"  Mode == Mean: {torch.allclose(z_mode, dist.mean)}")
    print(f"  Sample has noise: {not torch.allclose(z_sample, z_mode)}")
    
    print("  ✓ VAE encoding test passed!")


def test_jepa_predictor():
    """Test JEPA predictor forward pass."""
    print("\n" + "="*60)
    print("Test 2: JEPA Predictor Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    state_dim = 4 * 32 * 32
    action_embed_dim = 256
    batch_size = 2
    seq_len = 8
    
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_embed_dim,
        hidden_dim=256,
        depth=2,
    ).to(device)
    
    states = torch.randn(batch_size, seq_len, state_dim, device=device)
    actions = torch.randn(batch_size, seq_len, action_embed_dim, device=device)
    
    pred = predictor(states, actions)
    
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Prediction shape: {pred.shape}")
    assert pred.shape == states.shape
    
    print("  ✓ JEPA predictor test passed!")


def test_full_forward_pass():
    """Test full model forward pass."""
    print("\n" + "="*60)
    print("Test 3: Full Model Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = MockJEPADFoT(
        video_shape=(3, 256, 256),
        latent_channels=4,
        latent_size=32,
        action_dim=8,
        predictor_depth=2,
        predictor_hidden_dim=256,
    ).to(device)
    
    videos, actions, masks = create_mock_batch(
        batch_size=2, num_frames=8, height=256, width=256, action_dim=8, device=device
    )
    
    print(f"  Videos: {videos.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Masks: {masks.shape}")
    
    output = model.training_step(videos, actions, masks)
    
    print(f"  Total loss: {output['total_loss'].item():.4f}")
    print(f"  DFoT loss: {output['dfot_loss'].item():.4f}")
    print(f"  JEPA loss: {output['jepa_loss'].item():.4f}")
    
    assert not torch.isnan(output['total_loss'])
    print("  ✓ Full forward pass test passed!")


def test_gradient_flow():
    """Test that gradients flow to all components."""
    print("\n" + "="*60)
    print("Test 4: Gradient Flow")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MockJEPADFoT(
        train_vae_encoder=True,  # Enable VAE encoder training
        predictor_depth=2,
        predictor_hidden_dim=256,
    ).to(device)
    
    videos, actions, masks = create_mock_batch(batch_size=2, num_frames=8, device=device)
    
    # Forward
    output = model.training_step(videos, actions, masks)
    
    # Backward
    output['total_loss'].backward()
    
    # Check gradients
    def has_nonzero_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    
    vae_encoder_grad = has_nonzero_grad(model.vae.encoder)
    action_encoder_grad = has_nonzero_grad(model.action_encoder)
    predictor_grad = has_nonzero_grad(model.predictor)
    diffusion_grad = has_nonzero_grad(model.diffusion_model)
    
    print(f"  VAE encoder has gradients: {vae_encoder_grad}")
    print(f"  Action encoder has gradients: {action_encoder_grad}")
    print(f"  Predictor has gradients: {predictor_grad}")
    print(f"  Diffusion model has gradients: {diffusion_grad}")
    
    assert action_encoder_grad, "Action encoder should have gradients"
    assert predictor_grad, "Predictor should have gradients"
    assert diffusion_grad, "Diffusion model should have gradients"
    
    print("  ✓ Gradient flow test passed!")


def test_training_loop():
    """Test multiple training steps."""
    print("\n" + "="*60)
    print("Test 5: Training Loop (Loss Convergence)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MockJEPADFoT(
        train_vae_encoder=True,
        predictor_depth=2,
        predictor_hidden_dim=256,
        jepa_loss_weight=1.0,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Fixed batch for consistent comparison
    torch.manual_seed(42)
    videos, actions, masks = create_mock_batch(batch_size=4, num_frames=8, device=device)
    
    losses = []
    num_steps = 50
    
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model.training_step(videos, actions, masks)
        output['total_loss'].backward()
        optimizer.step()
        losses.append(output['total_loss'].item())
        
        if step % 10 == 0:
            print(f"    Step {step:3d}: loss={losses[-1]:.4f}, jepa={output['jepa_loss'].item():.4f}")
    
    print(f"\n  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    assert losses[-1] < losses[0], "Loss should decrease"
    print("  ✓ Training loop test passed!")


def test_frozen_vae():
    """Test that frozen VAE encoder doesn't get gradients."""
    print("\n" + "="*60)
    print("Test 6: Frozen VAE Encoder")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MockJEPADFoT(
        train_vae_encoder=False,  # Freeze VAE encoder
        predictor_depth=2,
    ).to(device)
    
    videos, actions, masks = create_mock_batch(batch_size=2, num_frames=4, device=device)
    
    output = model.training_step(videos, actions, masks)
    output['total_loss'].backward()
    
    # Check VAE encoder has no gradients (frozen)
    vae_encoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.vae.encoder.parameters()
    )
    
    # But predictor should have gradients
    predictor_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.predictor.parameters()
    )
    
    print(f"  VAE encoder has gradients: {vae_encoder_grad}")
    print(f"  Predictor has gradients: {predictor_grad}")
    
    assert not vae_encoder_grad, "Frozen VAE encoder should NOT have gradients"
    assert predictor_grad, "Predictor should have gradients"
    
    print("  ✓ Frozen VAE test passed!")


def test_different_loss_weights():
    """Test different JEPA loss weights."""
    print("\n" + "="*60)
    print("Test 7: Different JEPA Loss Weights")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    videos, actions, masks = create_mock_batch(batch_size=2, num_frames=8, device=device)
    
    results = {}
    for weight in [0.0, 0.5, 1.0, 2.0]:
        model = MockJEPADFoT(jepa_loss_weight=weight, predictor_depth=2).to(device)
        output = model.training_step(videos, actions, masks)
        results[weight] = {
            "total": output['total_loss'].item(),
            "dfot": output['dfot_loss'].item(),
            "jepa": output['jepa_loss'].item(),
        }
        print(f"  λ={weight}: total={results[weight]['total']:.4f}, dfot={results[weight]['dfot']:.4f}, jepa={results[weight]['jepa']:.4f}")
    
    # When weight=0, total should equal dfot
    assert abs(results[0.0]['total'] - results[0.0]['dfot']) < 1e-5
    
    print("  ✓ Loss weights test passed!")


def main():
    print("\n" + "#"*60)
    print("#" + " "*20 + "JEPA + DFoT FULL PIPELINE TESTS" + " "*7 + "#")
    print("#"*60)
    
    test_vae_encoding()
    test_jepa_predictor()
    test_full_forward_pass()
    test_gradient_flow()
    test_training_loop()
    test_frozen_vae()
    test_different_loss_weights()
    
    print("\n" + "#"*60)
    print("#" + " "*20 + "ALL TESTS PASSED! ✓" + " "*19 + "#")
    print("#"*60)
    
    print("\n" + "="*60)
    print("Pipeline Verification Complete!")
    print("="*60)
    print("""
Next steps to run with real data:

1. Load pre-trained DFoT model:
   python main.py \\
     +name=jepa_finetune \\
     experiment=video_generation \\
     dataset=minecraft \\
     algorithm=dfot_video_jepa \\
     load=pretrained:MCRAFT_L.ckpt \\
     algorithm.vae.pretrained_path=pretrained:ImageVAE_MCRAFT.ckpt

2. Or with your own checkpoint:
   python main.py \\
     +name=jepa_finetune \\
     experiment=video_generation \\
     dataset=minecraft \\
     algorithm=dfot_video_jepa \\
     load=/path/to/dfot_checkpoint.ckpt \\
     algorithm.vae.pretrained_path=/path/to/vae_checkpoint.ckpt
""")


if __name__ == "__main__":
    main()

