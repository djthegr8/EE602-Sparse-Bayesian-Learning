# Sparse Bayesian Learning for mmWave Massive MIMO-OFDM Channel Estimation

Simulation code for the EE602 (Statistical Signal Processing) term paper on delay-angular channel estimation with Sparse Bayesian Learning (SBL).

## Implemented methods

| Method | Module |
|---|---|
| Sample-LMMSE, OMP, LASSO (FISTA) | `sbl_mmwave/estimators/benchmarks.py` |
| EM-SBL, MacKay SBL, MMV M-SBL | `sbl_mmwave/estimators/sbl.py` |
| Off-Grid SBL (first-order Taylor refinement) | `sbl_mmwave/estimators/offgrid_sbl.py` |
| Oracle-MMSE (lower-bound reference) | `sbl_mmwave/estimators/oracle.py` |

Channel model: $\mathbf{y} = (\boldsymbol{\Phi}_0 + \boldsymbol{\Psi}_0\,\mathrm{diag}(\boldsymbol{\beta}))\,\mathbf{x} + \mathbf{n}$, with $K$-sparse $\mathbf{x}$ and off-grid offsets $\beta_i \in [-\delta_u/2, \delta_u/2]$.

## Layout

```
sbl_mmwave/
  dictionary.py     virtual angular dictionary (Phi_0, Psi_0)
  channel.py        sparse channel and pilot generation
  estimators/       all estimators
experiments/
  nmse_vs_snr.py    Experiment 1: NMSE vs. SNR
  em_convergence.py Experiment 2: marginal log-likelihood convergence
run_experiments.py  entry point
figures/            generated PDFs
```

## Usage

```bash
pip install -r requirements.txt
python run_experiments.py                  # defaults: 40 trials, seed 602
python run_experiments.py --trials 500     # publication-grade smoothing
```

Options: `--trials`, `--seed`, `--out`, `--N`, `--M`, `--K`, `--Kp`.

Outputs: `figures/fig_nmse_vs_snr.pdf`, `figures/fig_em_convergence.pdf`.

With the default arguments and seed, results match the original single-file script.

## Citation

```bibtex
@misc{sbl_mmwave_ee602,
  title  = {Sparse Bayesian Learning for Delay-Angular Channel Estimation in mmWave Massive MIMO-OFDM},
  author = {Dweep Joshipura, Kshitij Shukla},
  year   = {2026},
  note   = {EE602 term paper, simulation code},
  url    = {https://github.com/djthegr8/EE602-Sparse-Bayesian-Learning}
}
```
