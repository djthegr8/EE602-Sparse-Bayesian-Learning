"""Off-Grid Sparse Bayesian Learning (OG-SBL)."""

import numpy as np


def estimator_offgrid_sbl(
    Phi_0,
    Psi_0,
    y,
    delta_u,
    K_active_refine=8,
    max_iter=70,
    prune_thresh=1e-5,
    tol=1e-4,
):
    """
    Off-Grid Sparse Bayesian Learning (OG-SBL) with joint 1st-order Taylor
    steering perturbation updates beta in [-delta_u/2, +delta_u/2].

    Returns
    -------
    x_hat, beta, gamma
    """
    M, N = Phi_0.shape
    gamma = np.ones(N, dtype=float)
    beta = np.zeros(N, dtype=float)
    sigma2 = float(0.05 * np.linalg.norm(y) ** 2 / M)
    active_idx = np.arange(N)

    for it in range(max_iter):
        gamma_prev = gamma.copy()
        gamma_act = gamma[active_idx]
        beta_act = beta[active_idx]

        Phi_0_act = Phi_0[:, active_idx]
        Psi_0_act = Psi_0[:, active_idx]
        Phi_act = Phi_0_act + Psi_0_act * beta_act.reshape(1, -1)

        # E-Step: Posterior Mean and Full Covariance on Active Set
        Phi_Gamma = Phi_act * gamma_act.reshape(1, -1)
        Sigma_y = sigma2 * np.eye(M, dtype=complex) + Phi_Gamma @ Phi_act.conj().T
        Sigma_y_inv_Phi = np.linalg.solve(Sigma_y, Phi_act)
        Sigma_y_inv_y = np.linalg.solve(Sigma_y, y)

        mu_x = gamma_act.reshape(-1, 1) * (Phi_act.conj().T @ Sigma_y_inv_y)
        Sigma_x = np.diag(gamma_act) - (
            gamma_act.reshape(-1, 1)
            * (Phi_act.conj().T @ Sigma_y_inv_Phi)
            * gamma_act.reshape(1, -1)
        )
        Sigma_x_diag = np.maximum(1e-12, np.real(np.diag(Sigma_x)))

        # M-Step: Update gamma and sigma2
        gamma_new_act = np.abs(mu_x.flatten()) ** 2 + Sigma_x_diag
        gamma[active_idx] = gamma_new_act

        zeta = np.clip(1.0 - Sigma_x_diag / gamma_act, 1e-6, 1.0)
        res_norm = float(np.linalg.norm(y - Phi_act @ mu_x) ** 2)
        sigma2 = np.clip((res_norm + sigma2 * np.sum(zeta)) / M, 1e-6, 10.0)

        # M-Step: Update off-grid perturbation beta on top K_active_refine indices
        if it >= 2 and len(active_idx) > 0:
            top_k = min(K_active_refine, len(active_idx))
            top_local = np.argsort(gamma_new_act)[::-1][:top_k]

            Psi_sub = Psi_0_act[:, top_local]
            Phi_0_sub = Phi_0_act[:, top_local]
            mu_sub = mu_x[top_local]
            Sigma_sub = Sigma_x[np.ix_(top_local, top_local)]

            # Construct P matrix and v vector
            R_xx_sub = mu_sub @ mu_sub.conj().T + Sigma_sub
            P_mat = np.real((Psi_sub.conj().T @ Psi_sub).conj() * R_xx_sub)
            P_mat += 1e-6 * np.eye(top_k)

            # Residual excluding top_local nominal contribution
            other_local = np.setdiff1d(np.arange(len(active_idx)), top_local)
            y_res = y - Phi_0_act[:, other_local] @ mu_x[other_local]
            y_res0 = y_res - Phi_0_sub @ mu_sub

            v_vec = np.real(
                mu_sub.conj().flatten() * (Psi_sub.conj().T @ y_res0).flatten()
            ) - np.real(np.diag(Psi_sub.conj().T @ Phi_0_act @ Sigma_x[:, top_local]))

            beta_sub_new = np.linalg.solve(P_mat, v_vec)
            beta_sub_new = np.clip(beta_sub_new, -0.5 * delta_u, 0.5 * delta_u)
            beta[active_idx[top_local]] = beta_sub_new

        # Prune small hyperparameters
        keep_mask = gamma[active_idx] > prune_thresh
        if np.sum(keep_mask) >= 1:
            gamma[active_idx[~keep_mask]] = 0.0
            active_idx = active_idx[keep_mask]

        if (
            np.linalg.norm(gamma - gamma_prev)
            / max(np.linalg.norm(gamma_prev), 1e-12)
            < tol
            and it > 6
        ):
            break

    # Final estimate reconstruction
    x_hat = np.zeros((N, 1), dtype=complex)
    Phi_final = Phi_0[:, active_idx] + Psi_0[:, active_idx] * beta[active_idx].reshape(
        1, -1
    )
    g_final = gamma[active_idx]
    Sigma_y = (
        sigma2 * np.eye(M, dtype=complex)
        + (Phi_final * g_final.reshape(1, -1)) @ Phi_final.conj().T
    )
    x_hat[active_idx] = g_final.reshape(-1, 1) * (
        Phi_final.conj().T @ np.linalg.solve(Sigma_y, y)
    )

    return x_hat, beta, gamma
