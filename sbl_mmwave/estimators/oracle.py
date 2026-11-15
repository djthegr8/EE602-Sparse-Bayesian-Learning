"""Oracle-MMSE estimator (lower-bound reference)."""

import numpy as np


def estimator_oracle_mmse(Phi_true, y, support, sigma2):
    """Oracle-MMSE estimator with exact knowledge of true support and perturbation."""
    M, N = Phi_true.shape
    K = len(support)
    Phi_S = Phi_true[:, support]
    # True prior variance on active support is 1.0
    A_mat = (1.0 / sigma2) * (Phi_S.conj().T @ Phi_S) + np.eye(K, dtype=complex)
    x_S_hat = (1.0 / sigma2) * np.linalg.solve(A_mat, Phi_S.conj().T @ y)

    x_hat = np.zeros((N, 1), dtype=complex)
    x_hat[support] = x_S_hat
    return x_hat
