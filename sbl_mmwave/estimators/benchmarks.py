"""Benchmark estimators: Sample-LMMSE, OMP, LASSO (FISTA)."""

import numpy as np


def estimator_sample_lmmse(Phi, y, sigma2):
    """Classical LMMSE with identity prior covariance R_x = (K/N)*I_N."""
    M, N = Phi.shape
    prior_var = np.linalg.norm(y) ** 2 / (M * N)
    Sigma_y = sigma2 * np.eye(M) + prior_var * (Phi @ Phi.conj().T)
    x_hat = prior_var * Phi.conj().T @ np.linalg.solve(Sigma_y, y)
    return x_hat


def estimator_omp(Phi, y, K_sparsity):
    """Orthogonal Matching Pursuit (OMP) for sparse recovery."""
    M, N = Phi.shape
    residual = y.copy()
    support_est = []
    x_hat = np.zeros((N, 1), dtype=complex)

    for _ in range(K_sparsity):
        correlations = np.abs(Phi.conj().T @ residual).flatten()
        correlations[support_est] = -1.0
        best_atom = int(np.argmax(correlations))
        support_est.append(best_atom)

        Phi_sub = Phi[:, support_est]
        x_sub, _, _, _ = np.linalg.lstsq(Phi_sub, y, rcond=None)
        residual = y - Phi_sub @ x_sub

    x_hat[support_est] = x_sub
    return x_hat


def estimator_lasso_fista(Phi, y, sigma2, max_iter=150):
    """LASSO (l1-regularized Basis Pursuit) solved via accelerated FISTA."""
    M, N = Phi.shape
    # Lipschitz constant L of gradient ||Phi^H (Phi x - y)||
    L_lip = np.linalg.norm(Phi, ord=2) ** 2
    step = 1.0 / L_lip
    # Regularization parameter scaled with noise standard deviation
    lam = 1.35 * np.sqrt(sigma2 * np.log(N))

    x_k = np.zeros((N, 1), dtype=complex)
    z_k = x_k.copy()
    t_k = 1.0

    for _ in range(max_iter):
        grad = Phi.conj().T @ (Phi @ z_k - y)
        v = z_k - step * grad
        # Complex soft-thresholding operator
        mag = np.abs(v)
        scale = np.maximum(0.0, 1.0 - (step * lam) / np.maximum(mag, 1e-12))
        x_next = v * scale

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        z_k = x_next + ((t_k - 1.0) / t_next) * (x_next - x_k)
        x_k = x_next
        t_k = t_next

    return x_k
