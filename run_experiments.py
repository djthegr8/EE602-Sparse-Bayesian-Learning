"""
EE602: Statistical Signal Processing - Term Paper Simulation Suite
Sparse Bayesian Learning (SBL) for Delay-Angular Channel Estimation
in Millimeter-Wave Massive MIMO-OFDM Systems.

Usage:
    python run_experiments.py [--trials 40] [--seed 602] [--out figures]
"""

import argparse
import os

import numpy as np

from sbl_mmwave import build_virtual_dictionary
from experiments import nmse_vs_snr, em_convergence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40, help="Monte Carlo trials per SNR (use 500 for publication)")
    parser.add_argument("--seed", type=int, default=602, help="NumPy random seed")
    parser.add_argument("--out", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--N", type=int, default=128, help="Virtual grid size")
    parser.add_argument("--M", type=int, default=48, help="Pilot dimension")
    parser.add_argument("--K", type=int, default=6, help="Number of active paths")
    parser.add_argument("--Kp", type=int, default=16, help="Pilot subcarriers (MMV)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 76)
    print("EE602 Statistical Signal Processing - Sparse Bayesian Learning Simulation")
    print("=" * 76)

    Phi_0, Psi_0, _, delta_u, coherence, _, _ = build_virtual_dictionary(args.M, args.N)
    print(f"Virtual Grid Dimension N = {args.N}, Pilot Dimension M = {args.M}, Active Paths K = {args.K}")
    print(f"Dictionary Mutual Coherence mu(Phi_0) = {coherence:.4f}")
    print("-" * 76)

    nmse_vs_snr.run(Phi_0, Psi_0, delta_u, args.K, args.Kp, args.trials, args.out)
    em_convergence.run(Phi_0, Psi_0, delta_u, args.K, args.out)

    print("-" * 76)
    print(f"Saved plots to '{args.out}/': fig_nmse_vs_snr.pdf, fig_em_convergence.pdf")
    print("=" * 76)


if __name__ == "__main__":
    main()
