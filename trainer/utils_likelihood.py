import numpy as np
import torch
from torch.distributions import Independent, Normal


def _get_gaussian_log_density(self, shape, device):
    """
    Create log probability function for isotropic Gaussian noise distribution.

    Args:
        shape: Shape of a single sample (excluding batch dim), e.g., [D, T, H, W]
        device: Device to create the distribution on

    Returns:
        Callable that computes log p(x) for batched inputs [B, D, T, H, W] -> [B]
    """
    flat_dim = int(np.prod(shape))

    # Gaussian with configured noise_std
    dist = Independent(
        Normal(
            torch.zeros(flat_dim, device=device),
            torch.ones(flat_dim, device=device) * self.noise_std,
        ),
        1,  # Treat last dim as event dim -> returns [B] shaped log probs
    )

    def log_prob_fn(x):
        """Compute log probability of samples. x: [B, D, T, H, W] -> [B]."""
        x_flat = x.reshape(x.shape[0], -1)
        return dist.log_prob(x_flat)

    return log_prob_fn
