"""
SigREG — Sketched Isotropic Gaussian Regularization.

Minimal standalone implementation based on:
    LeJEPA (https://github.com/galilai-group/lejepa)

Prevents representation collapse by pushing the embedding distribution
toward an isotropic standard Gaussian N(0, I) via sliced characteristic-
function testing (Epps-Pulley statistic).
"""

import torch
import torch.nn as nn
from torch import Tensor


class EppsPulley(nn.Module):
    """Univariate Gaussian goodness-of-fit via empirical characteristic function.

    Computes  T = N * ∫ |φ_emp(t) − φ_N(t)|² w(t) dt
    where φ_N(t) = exp(−t²/2) is the standard-normal CF and the integral
    is approximated with the trapezoidal rule on [0, t_max].
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt  # trapezoidal end-correction
        phi = (-0.5 * t ** 2).exp()  # standard-normal CF evaluated at t
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, N) — samples along the second-to-last dim.
        Returns:
            Scalar(s) test statistic per leading batch dim.
        """
        N = x.size(-2)
        xt = x.unsqueeze(-1) * self.t  # (*, N, n_points)
        cos_mean = torch.cos(xt).mean(dim=-2)  # (*, n_points)
        sin_mean = torch.sin(xt).mean(dim=-2)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N


class SigREG(nn.Module):
    """Sliced Isotropic Gaussian Regularization.

    Projects D-dimensional embeddings onto ``num_slices`` random unit
    directions, applies the Epps-Pulley univariate normality test to each
    1-D projection, and averages the statistics.

    Args:
        num_slices: Number of random projection directions.
        t_max:      Upper integration limit for the EP test.
        n_points:   Quadrature points (must be odd).
    """

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley(t_max=t_max, n_points=n_points)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: (N, D) batch of embedding vectors from the online
                        encoder (gradients must flow through these).
        Returns:
            Scalar regularization loss.
        """
        N, D = embeddings.shape

        # Random unit-norm projection directions
        directions = torch.randn(
            D, self.num_slices, device=embeddings.device, dtype=embeddings.dtype
        )
        directions = directions / directions.norm(dim=0, keepdim=True)

        # Project: (N, D) @ (D, K) -> (N, K)
        projections = embeddings @ directions

        # EP test expects (*, N, 1); transpose so slices are the batch dim
        stats = self.test(projections.t().unsqueeze(-1))  # (K, 1)

        return stats.mean()