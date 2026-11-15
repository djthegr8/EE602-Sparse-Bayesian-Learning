"""Experiment 1: NMSE (dB) vs. SNR (dB) under off-grid channel conditions."""

import os

import numpy as np
import matplotlib.pyplot as plt

from sbl_mmwave import generate_mmwave_channel
from sbl_mmwave.estimators import (
    estimator_sample_lmmse,
    estimator_omp,
    estimator_lasso_fista,
    estimator_sbl,
    estimator_offgrid_sbl,
    estimator_oracle_mmse,
)

SNR_DB_LIST = [-5, 0, 5, 10, 15, 20, 25]
ALG_NAMES = [
    "Sample-LMMSE",
    "OMP",
    "LASSO-FISTA",
    "EM-SBL (On-Grid)",
    "MacKay-SBL",
    "M-SBL (MMV)",
    "Off-Grid SBL",
    "Oracle-MMSE",
]


def run(Phi_0, Psi_0, delta_u, K, K_p, num_trials, out_dir="figures"):
    nmse_snr = {name: [] for name in ALG_NAMES}

    print("Running Experiment 1: NMSE (dB) vs. SNR (dB) [Off-Grid Channel]...")
    for snr_db in SNR_DB_LIST:
        err_acc = {name: 0.0 for name in ALG_NAMES}
        sig_acc = 0.0

        for _ in range(num_trials):
            Y_mmv, X_true_mmv, Phi_true, support, beta_true, sigma2 = (
                generate_mmwave_channel(
                    Phi_0, Psi_0, delta_u, K, snr_db, K_p=K_p, off_grid=True
                )
            )
            y_smv = Y_mmv[:, [0]]
            x_true = X_true_mmv[:, [0]]
            h_true = Phi_true @ x_true
            sig_acc += float(np.linalg.norm(h_true) ** 2)

            # 1. Sample-LMMSE
            x_lmmse = estimator_sample_lmmse(Phi_0, y_smv, sigma2)
            err_acc["Sample-LMMSE"] += float(np.linalg.norm(Phi_0 @ x_lmmse - h_true) ** 2)

            # 2. OMP
            x_omp = estimator_omp(Phi_0, y_smv, K)
            err_acc["OMP"] += float(np.linalg.norm(Phi_0 @ x_omp - h_true) ** 2)

            # 3. LASSO (FISTA)
            x_lasso = estimator_lasso_fista(Phi_0, y_smv, sigma2)
            err_acc["LASSO-FISTA"] += float(np.linalg.norm(Phi_0 @ x_lasso - h_true) ** 2)

            # 4. EM-SBL (On-Grid SMV)
            x_emsbl, _ = estimator_sbl(Phi_0, y_smv, sigma2, mode="em")
            err_acc["EM-SBL (On-Grid)"] += float(
                np.linalg.norm(Phi_0 @ x_emsbl - h_true) ** 2
            )

            # 5. MacKay Fast-SBL (On-Grid SMV)
            x_mksbl, _ = estimator_sbl(Phi_0, y_smv, sigma2, mode="mackay")
            err_acc["MacKay-SBL"] += float(np.linalg.norm(Phi_0 @ x_mksbl - h_true) ** 2)

            # 6. MMV Block-SBL (M-SBL across K_p subcarriers)
            X_msbl, _ = estimator_sbl(Phi_0, Y_mmv, sigma2, mode="em")
            err_acc["M-SBL (MMV)"] += float(
                np.linalg.norm(Phi_0 @ X_msbl[:, [0]] - h_true) ** 2
            )

            # 7. Off-Grid SBL (OG-SBL)
            x_ogsbl, beta_est, _ = estimator_offgrid_sbl(Phi_0, Psi_0, y_smv, delta_u)
            Phi_est = Phi_0 + Psi_0 @ np.diag(beta_est)
            err_acc["Off-Grid SBL"] += float(
                np.linalg.norm(Phi_est @ x_ogsbl - h_true) ** 2
            )

            # 8. Oracle-MMSE / BCRLB
            x_oracle = estimator_oracle_mmse(Phi_true, y_smv, support, sigma2)
            err_acc["Oracle-MMSE"] += float(
                np.linalg.norm(Phi_true @ x_oracle - h_true) ** 2
            )

        for name in ALG_NAMES:
            nmse_snr[name].append(10.0 * np.log10(err_acc[name] / sig_acc))

        print(
            f"  SNR = {snr_db:3d} dB | OMP: {nmse_snr['OMP'][-1]:6.2f} dB | "
            f"LASSO: {nmse_snr['LASSO-FISTA'][-1]:6.2f} dB | "
            f"EM-SBL: {nmse_snr['EM-SBL (On-Grid)'][-1]:6.2f} dB | "
            f"OG-SBL: {nmse_snr['Off-Grid SBL'][-1]:6.2f} dB | "
            f"Oracle: {nmse_snr['Oracle-MMSE'][-1]:6.2f} dB"
        )

    _plot(nmse_snr, K_p, out_dir)
    return nmse_snr


def _plot(nmse_snr, K_p, out_dir):
    plt.figure(figsize=(7.5, 5.0))
    s = SNR_DB_LIST
    plt.plot(s, nmse_snr["Sample-LMMSE"], "o-", color="#556070", label="Sample-LMMSE")
    plt.plot(s, nmse_snr["OMP"], "s-", color="#d97706", label="Greedy OMP")
    plt.plot(s, nmse_snr["LASSO-FISTA"], "^-", color="#0d9488", label="LASSO (l1-FISTA)")
    plt.plot(s, nmse_snr["EM-SBL (On-Grid)"], "d-", color="#1e3a8a", label="EM-SBL (On-Grid)")
    plt.plot(s, nmse_snr["M-SBL (MMV)"], "v-", color="#7c3aed", label=f"MMV M-SBL (Kp={K_p})")
    plt.plot(s, nmse_snr["Off-Grid SBL"], "p-", color="#be123c", linewidth=2.2, label="Off-Grid SBL (OG-SBL)")
    plt.plot(s, nmse_snr["Oracle-MMSE"], "k--", linewidth=1.8, label="Oracle-MMSE / Hybrid BCRLB")
    plt.xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=11)
    plt.ylabel("Normalized Mean Square Error (NMSE in dB)", fontsize=11)
    plt.title("mmWave Massive MIMO-OFDM Channel Estimation: NMSE vs. SNR", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_nmse_vs_snr.pdf"))
    plt.close()
