from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance as _FrechetInceptionDistance
from .shared_registry import SharedVideoMetricModelRegistry

# Regularization for covariance matrices to avoid singular/near-singular FID (and FVD).
# With few or low-diversity samples, covariances can be singular and sqrtm/eigvals become unstable.
FID_COV_REG: float = 1e-6


def _compute_frechet_distance_stable(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = FID_COV_REG
) -> Tensor:
    """FID = ||mu1 - mu2||^2 + Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrt(sigma1 @ sigma2)).
    Uses eigvals of sigma1 @ sigma2 with regularization so singular covariances don't blow up.
    """
    sigma1 = sigma1 + eps * torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
    sigma2 = sigma2 + eps * torch.eye(sigma2.size(0), device=sigma2.device, dtype=sigma2.dtype)
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    # Eigenvalues of sigma1 @ sigma2 can be complex; for PSD matrices they are real and >= 0.
    # Clamp to avoid sqrt of tiny negative values from numerical error.
    eigvals = torch.linalg.eigvals(sigma1 @ sigma2).real.clamp(min=0.0)
    c = eigvals.sqrt().sum(dim=-1)
    return a + b - 2 * c


class BaseFrechetDistance(_FrechetInceptionDistance, ABC):
    """
    Base class for Fréchet distance metrics (e.g. FID, FVD).
    AAdapted from `torchmetrics.image.FrechetInceptionDistance` to work with shared model registry and support different feature extractors and modalities (e.g. images, videos).
    """

    orig_dtype: torch.dtype

    def __init__(
        self,
        registry: Optional[SharedVideoMetricModelRegistry],
        features: int,
        reset_real_features=True,
        **kwargs,
    ):
        # pylint: disable=non-parent-init-called
        Metric.__init__(self, **kwargs)

        self.registry = registry
        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        num_features = features
        mx_nb_feets = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        return (
            self.real_features_num_samples == 0 or self.fake_features_num_samples == 0
        )

    @abstractmethod
    def extract_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_input(fake: Tensor, real: Tensor) -> bool:
        return True

    def _update(self, x: Tensor, real: bool) -> None:
        # pylint: disable=no-member
        features = self.extract_features(x)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.size(0)
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.size(0)

    def update(self, fake: Tensor, real: Tensor) -> None:
        if not self._check_input(fake, real):
            return
        self._update(fake, real=False)
        self._update(real, real=True)

    def compute(self) -> Tensor:
        """Stabilized FID/FVD: regularize covariances to avoid singular-matrix sqrt and bad metrics."""
        # pylint: disable=no-member
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake "
                "distribution to compute Fréchet distance."
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        out = _compute_frechet_distance_stable(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        )
        return out.to(getattr(self, "orig_dtype", torch.float32))
