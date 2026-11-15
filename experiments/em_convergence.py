"""Experiment 2: Convergence of the type-II ML marginal log-likelihood."""

import os

import matplotlib.pyplot as plt

from sbl_mmwave import generate_mmwave_channel
from sbl_mmwave.estimators import estimator_sbl


def run(Phi_0, Psi_0, delta_u, K, out_dir="figures", snr_db=15.0):
    Y_conv, _, _, _, _, sigma2_conv = generate_mmwave_channel(
        Phi_0, Psi_0, delta_u, K, snr_db=snr_db, K_p=1, off_grid=False
    )
    _, _, ev_em = estimator_sbl(
        Phi_0, Y_conv, sigma2_conv, mode="em", max_iter=50, tol=1e-9, record_evidence=True
    )
    _, _, ev_mk = estimator_sbl(
        Phi_0, Y_conv, sigma2_conv, mode="mackay", max_iter=50, tol=1e-9, record_evidence=True
    )

    plt.figure(figsize=(6.8, 4.4))
    plt.plot(range(1, len(ev_em) + 1), ev_em, "o-", color="#1e3a8a", label="Standard EM-SBL")
    plt.plot(range(1, len(ev_mk) + 1), ev_mk, "s-", color="#be123c", label="MacKay Fast-SBL")
    plt.xlabel("Iteration Number (t)", fontsize=11)
    plt.ylabel("Marginal Log-Likelihood L(gamma^(t))", fontsize=11)
    plt.title(f"Monotonic Convergence of SBL Marginal Log-Evidence (SNR = {snr_db:g} dB)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_em_convergence.pdf"))
    plt.close()

    return ev_em, ev_mk
