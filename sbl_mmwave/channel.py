"""Sparse mmWave channel and pilot measurement generation."""

import numpy as np


def generate_mmwave_channel(Phi_0, Psi_0, delta_u, K, snr_db, K_p=1, off_grid=True):
    """
    Generates sparse channel realizations x (N x K_p) and noisy pilot
    measurements y (M x K_p) with optional continuous off-grid perturbations.

    Returns
    -------
    Y, X_true, Phi_true, support, beta_true, sigma2
    """
    M, N = Phi_0.shape
    # Select K grid indices to represent physical clusters
    support = np.random.choice(N, size=K, replace=False)
    support.sort()

    # Off-grid perturbations beta_i in [-0.45*delta_u, +0.45*delta_u]
    beta_true = np.zeros(N)
    if off_grid:
        beta_true[support] = np.random.uniform(-0.45 * delta_u, 0.45 * delta_u, size=K)

    # Construct true perturbed sensing matrix
    Phi_true = Phi_0 + Psi_0 @ np.diag(beta_true)

    # Complex Gaussian path gains X (N x K_p)
    X_true = np.zeros((N, K_p), dtype=complex)
    X_true[support, :] = (
        np.random.randn(K, K_p) + 1j * np.random.randn(K, K_p)
    ) / np.sqrt(2.0)

    # Clean signal Y_clean = Phi_true @ X_true
    Y_clean = Phi_true @ X_true
    signal_power = np.mean(np.sum(np.abs(Y_clean) ** 2, axis=0)) / M
    sigma2 = signal_power * (10.0 ** (-snr_db / 10.0))

    noise = np.sqrt(sigma2 / 2.0) * (
        np.random.randn(M, K_p) + 1j * np.random.randn(M, K_p)
    )
    Y = Y_clean + noise

    return Y, X_true, Phi_true, support, beta_true, sigma2
