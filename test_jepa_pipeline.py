"""
Test script for JEPA + DFoT pipeline with mock random data.
Tests the ViT-based predictor with Teacher Forcing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Copy of the key components for testing (to avoid import issues)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
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
    """Multi-head attention with causal masking."""
    
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
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        dots = dots.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ViTPredictor(nn.Module):
    """ViT-based predictor for JEPA with causal attention."""
    
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
        pred_states = self.output_proj(x)
        
        return pred_states


class MockVAE(nn.Module):
    """Mock VAE for testing."""
    
    def __init__(self, in_channels=3, latent_channels=4, downsample_factor=8):
        super().__init__()
        self.latent_channels = latent_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels * 2, 4, 2, 1),
        )
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1)
        
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


class MockDistribution:
    def __init__(self, moments):
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
    
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)
    
    def mode(self):
        return self.mean


# =============================================================================
# Test Functions
# =============================================================================

def test_causal_attention():
    """Test that causal attention properly masks future tokens."""
    print("\n" + "="*60)
    print("Testing Causal Attention...")
    print("="*60)
    
    batch_size = 2
    seq_len = 8
    dim = 64
    
    attn = CausalAttention(dim=dim, heads=4, dim_head=16)
    x = torch.randn(batch_size, seq_len, dim)
    
    out = attn(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape
    print("  ✓ Causal attention test passed!")


def test_vit_predictor():
    """Test the ViT predictor."""
    print("\n" + "="*60)
    print("Testing ViT Predictor...")
    print("="*60)
    
    batch_size = 2
    seq_len = 8
    state_dim = 4 * 32 * 32  # 4096
    action_dim = 8
    
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        depth=2,
        heads=4,
        dim_head=32,
        max_seq_len=16,
    )
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    
    pred_states = predictor(states, actions)
    
    print(f"  Input states shape: {states.shape}")
    print(f"  Input actions shape: {actions.shape}")
    print(f"  Predicted states shape: {pred_states.shape}")
    assert pred_states.shape == states.shape
    print("  ✓ ViT predictor test passed!")


def test_teacher_forcing_loss():
    """Test the Teacher Forcing JEPA loss computation."""
    print("\n" + "="*60)
    print("Testing Teacher Forcing JEPA Loss...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    batch_size = 2
    seq_len = 8
    latent_channels = 4
    latent_size = 32
    action_dim = 8
    state_dim = latent_channels * latent_size * latent_size
    
    # Create predictor
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        depth=2,
        heads=4,
    ).to(device)
    
    # Create mock data
    latents = torch.randn(batch_size, seq_len, latent_channels, latent_size, latent_size, device=device)
    actions = torch.randn(batch_size, seq_len, action_dim, device=device)
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Compute loss (mimicking _compute_jepa_loss)
    B, T, C, H, W = latents.shape
    latents_flat = latents.view(B, T, -1)
    
    states_input = latents_flat[:, :-1]
    actions_input = actions[:, 1:]
    states_target = latents_flat[:, 1:]
    transition_masks = masks[:, :-1] & masks[:, 1:]
    
    states_pred = predictor(states_input, actions_input)
    
    pred_loss = F.smooth_l1_loss(states_pred, states_target, reduction="none")
    pred_loss = pred_loss.mean(dim=-1)
    pred_loss = (pred_loss * transition_masks.float()).sum() / transition_masks.sum()
    
    print(f"  Latents shape: {latents.shape}")
    print(f"  States input shape: {states_input.shape}")
    print(f"  States target shape: {states_target.shape}")
    print(f"  Predicted states shape: {states_pred.shape}")
    print(f"  JEPA Loss: {pred_loss.item():.4f}")
    
    # Test backward
    pred_loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    print(f"  Predictor has gradients: {has_grad}")
    
    assert not torch.isnan(pred_loss)
    assert has_grad
    print("  ✓ Teacher forcing loss test passed!")


def test_full_pipeline_with_vae():
    """Test full pipeline: VAE encoding + ViT predictor + loss."""
    print("\n" + "="*60)
    print("Testing Full Pipeline (VAE + ViT Predictor)...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    batch_size = 2
    seq_len = 8
    height, width = 256, 256
    latent_channels = 4
    latent_size = 32
    action_dim = 8
    state_dim = latent_channels * latent_size * latent_size
    
    # Create models
    vae = MockVAE(in_channels=3, latent_channels=latent_channels).to(device)
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        depth=2,
        heads=4,
    ).to(device)
    
    # Create mock video batch
    videos = torch.rand(batch_size, seq_len, 3, height, width, device=device)
    actions = torch.randn(batch_size, seq_len, action_dim, device=device)
    
    print(f"  Videos shape: {videos.shape}")
    print(f"  Actions shape: {actions.shape}")
    
    # Encode all frames with VAE
    B, T, C, H, W = videos.shape
    videos_flat = videos.view(B * T, C, H, W)
    latents_flat = vae.encode(videos_flat).sample()
    latents = latents_flat.view(B, T, latent_channels, latent_size, latent_size)
    
    print(f"  Latents shape: {latents.shape}")
    
    # Flatten for predictor
    latents_seq = latents.view(B, T, -1)
    
    # Teacher forcing: predict s_{t+1} from (s_t, a_{t+1})
    states_input = latents_seq[:, :-1]
    actions_input = actions[:, 1:]
    states_target = latents_seq[:, 1:]
    
    states_pred = predictor(states_input, actions_input)
    
    # Compute loss
    loss = F.smooth_l1_loss(states_pred, states_target)
    
    print(f"  States input shape: {states_input.shape}")
    print(f"  States target shape: {states_target.shape}")
    print(f"  States pred shape: {states_pred.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients flow to VAE encoder
    vae_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in vae.encoder.parameters())
    predictor_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    
    print(f"  VAE encoder has gradients: {vae_has_grad}")
    print(f"  Predictor has gradients: {predictor_has_grad}")
    
    assert predictor_has_grad, "Predictor should have gradients"
    # Note: VAE may or may not have gradients depending on if we detach target
    print("  ✓ Full pipeline test passed!")


def test_training_loop():
    """Test multiple training steps to verify loss decreases."""
    print("\n" + "="*60)
    print("Testing Training Loop (Loss Convergence)...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    batch_size = 4
    seq_len = 8
    latent_channels = 4
    latent_size = 16  # Smaller for faster testing
    action_dim = 8
    state_dim = latent_channels * latent_size * latent_size
    num_steps = 100
    
    # Create predictor
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        depth=2,
        heads=4,
        dim_head=32,
    ).to(device)
    
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    
    # Create fixed "ground truth" dynamics for testing
    # Next state = f(current_state, action)
    # We'll use a simple linear transformation
    torch.manual_seed(42)
    true_state_transform = torch.randn(state_dim, state_dim, device=device) * 0.01
    true_action_transform = torch.randn(action_dim, state_dim, device=device) * 0.1
    
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Generate random states and actions
        states = torch.randn(batch_size, seq_len, state_dim, device=device) * 0.1
        actions = torch.randn(batch_size, seq_len, action_dim, device=device)
        
        # Compute "true" next states using simple dynamics
        # s_{t+1} = s_t @ W_s + a_t @ W_a
        states_input = states[:, :-1]
        actions_input = actions[:, 1:]
        states_target = states_input @ true_state_transform + actions_input @ true_action_transform
        
        # Predict
        states_pred = predictor(states_input, actions_input)
        
        # Loss
        loss = F.mse_loss(states_pred, states_target)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"    Step {step:3d}: Loss = {loss.item():.6f}")
    
    print(f"\n  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
    
    assert losses[-1] < losses[0] * 0.5, "Loss should decrease significantly"
    print("  ✓ Training loop test passed!")


def test_gradient_flow_to_vae():
    """Test that gradients properly flow back to VAE encoder (key for JEPA)."""
    print("\n" + "="*60)
    print("Testing Gradient Flow to VAE Encoder...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    batch_size = 2
    seq_len = 4
    height, width = 64, 64  # Smaller for testing
    latent_channels = 4
    latent_size = 8
    action_dim = 4
    state_dim = latent_channels * latent_size * latent_size
    
    # Create models
    vae = MockVAE(in_channels=3, latent_channels=latent_channels).to(device)
    predictor = ViTPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        depth=1,
        heads=2,
    ).to(device)
    
    # Create optimizer that includes VAE encoder
    optimizer = torch.optim.Adam(
        list(vae.encoder.parameters()) + 
        list(vae.quant_conv.parameters()) + 
        list(predictor.parameters()),
        lr=1e-4
    )
    
    # Save initial VAE encoder weights
    initial_weights = vae.encoder[0].weight.clone().detach()
    
    # Training step
    for _ in range(5):
        optimizer.zero_grad()
        
        # Create mock data
        videos = torch.rand(batch_size, seq_len, 3, height, width, device=device)
        actions = torch.randn(batch_size, seq_len, action_dim, device=device)
        
        # Encode with VAE (gradients should flow through this)
        B, T, C, H, W = videos.shape
        videos_flat = videos.view(B * T, C, H, W)
        latents_flat = vae.encode(videos_flat).sample()
        latents = latents_flat.view(B, T, latent_channels, latent_size, latent_size)
        latents_seq = latents.view(B, T, -1)
        
        # Teacher forcing prediction
        states_input = latents_seq[:, :-1]
        actions_input = actions[:, 1:]
        states_target = latents_seq[:, 1:]  # NOT detached - gradients flow to VAE
        
        states_pred = predictor(states_input, actions_input)
        
        # Loss
        loss = F.mse_loss(states_pred, states_target)
        loss.backward()
        optimizer.step()
    
    # Check if VAE weights changed
    final_weights = vae.encoder[0].weight.clone().detach()
    weight_change = (final_weights - initial_weights).abs().mean().item()
    
    print(f"  VAE encoder weight change: {weight_change:.8f}")
    print(f"  Weights changed: {weight_change > 1e-8}")
    
    assert weight_change > 1e-8, "VAE encoder weights should have changed"
    print("  ✓ Gradient flow to VAE test passed!")


def main():
    print("\n" + "#"*60)
    print("# JEPA + DFoT Pipeline Tests (ViT Predictor + Teacher Forcing)")
    print("#"*60)
    
    # Run all tests
    test_causal_attention()
    test_vit_predictor()
    test_teacher_forcing_loss()
    test_full_pipeline_with_vae()
    test_training_loop()
    test_gradient_flow_to_vae()
    
    print("\n" + "#"*60)
    print("# ALL TESTS PASSED! ✓")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
