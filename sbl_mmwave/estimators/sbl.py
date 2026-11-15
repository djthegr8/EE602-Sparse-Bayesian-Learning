"""On-grid Sparse Bayesian Learning: EM-SBL, MacKay SBL, and MMV M-SBL."""

import numpy as np


def estimator_sbl(
    Phi,
    Y,
    sigma2_init=None,
    mode="em",
    max_iter=80,
    prune_thresh=1e-5,
    tol=1e-4,
    record_evidence=False,
):
    """
    Unified Sparse Bayesian Learning (SBL) Estimator supporting:
    - mode='em'     : Standard Expectation-Maximization SBL (SMV, or MMV M-SBL
                      when Y has multiple columns)
    - mode='mackay' : MacKay Accelerated Fixed-Point SBL

    Returns
    -------
    (X_hat, gamma) or, if record_evidence is True,
    (X_hat, gamma, evidence_history)
    """
    M, N = Phi.shape
    K_p = Y.shape[1]

    gamma = np.ones(N, dtype=float)
    if sigma2_init is None:
        sigma2 = 0.05 * np.mean(np.sum(np.abs(Y) ** 2, axis=0)) / M
    else:
        sigma2 = float(sigma2_init)

    active_idx = np.arange(N)
    evidence_history = []

    for it in range(max_iter):
        gamma_prev = gamma.copy()
        gamma_act = gamma[active_idx]
        Phi_act = Phi[:, active_idx]

        # E-Step via Woodbury Matrix Identity: Sigma_y (M x M)
        Phi_Gamma = Phi_act * gamma_act.reshape(1, -1)
        Sigma_y = sigma2 * np.eye(M, dtype=complex) + Phi_Gamma @ Phi_act.conj().T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.conj().T)  # Ensure Hermitian symmetry

        Sigma_y_inv_Phi = np.linalg.solve(Sigma_y, Phi_act)
        Sigma_y_inv_Y = np.linalg.solve(Sigma_y, Y)

        if record_evidence:
            sign, logdet = np.linalg.slogdet(Sigma_y)
            quad_term = np.real(np.sum(Y.conj() * Sigma_y_inv_Y)) / K_p
            log_ev = -M * np.log(np.pi) - logdet - quad_term
            evidence_history.append(float(log_ev))

        # Posterior mean mu_x (|A| x K_p) and diagonal of Sigma_x (|A|,)
        mu_x_act = gamma_act.reshape(-1, 1) * (Phi_act.conj().T @ Sigma_y_inv_Y)
        diag_PhiH_Sinv_Phi = np.real(np.sum(Phi_act.conj() * Sigma_y_inv_Phi, axis=0))
        Sigma_x_diag = np.maximum(
            1e-12, gamma_act - (gamma_act ** 2) * diag_PhiH_Sinv_Phi
        )

        # Average squared posterior magnitude across K_p pilot subcarriers
        mean_sq_mu = np.mean(np.abs(mu_x_act) ** 2, axis=1)
        zeta = np.clip(1.0 - Sigma_x_diag / gamma_act, 1e-6, 1.0)

        # M-Step Hyperparameter Update
        if mode == "mackay":
            gamma_new_act = mean_sq_mu / zeta
        else:
            gamma_new_act = mean_sq_mu + Sigma_x_diag

        gamma[active_idx] = gamma_new_act

        # M-Step Noise Variance Update
        residual_norm = np.mean(np.sum(np.abs(Y - Phi_act @ mu_x_act) ** 2, axis=0))
        if mode == "mackay":
            denom = max(1.0, M - np.sum(zeta))
            sigma2 = np.clip(residual_norm / denom, 1e-6, 10.0)
        else:
            sigma2 = np.clip(
                (residual_norm + sigma2 * np.sum(zeta)) / M, 1e-6, 10.0
            )

        # Prune inactive coordinates
        keep_mask = gamma[active_idx] > prune_thresh
        if np.sum(keep_mask) >= 1:
            pruned_out = active_idx[~keep_mask]
            gamma[pruned_out] = 0.0
            active_idx = active_idx[keep_mask]

        rel_change = np.linalg.norm(gamma - gamma_prev) / max(
            np.linalg.norm(gamma_prev), 1e-12
        )
        if rel_change < tol and it > 5:
            break

    # Reconstruct full N x K_p posterior mean
    X_hat = np.zeros((N, K_p), dtype=complex)
    Phi_act = Phi[:, active_idx]
    gamma_act = gamma[active_idx]
    Sigma_y = (
        sigma2 * np.eye(M, dtype=complex)
        + (Phi_act * gamma_act.reshape(1, -1)) @ Phi_act.conj().T
    )
    X_hat[active_idx, :] = gamma_act.reshape(-1, 1) * (
        Phi_act.conj().T @ np.linalg.solve(Sigma_y, Y)
    )

    if record_evidence:
        return X_hat, gamma, evidence_history
    return X_hat, gamma
