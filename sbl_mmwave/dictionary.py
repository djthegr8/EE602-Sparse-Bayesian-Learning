"""Virtual angular dictionary for mmWave massive MIMO."""

import numpy as np


def build_virtual_dictionary(M, N, d_over_lambda=0.5):
    """
    Constructs a compressed virtual angular sensing dictionary Phi_0 (M x N)
    and its angular derivative dictionary Psi_0 (M x N) for Off-Grid SBL.

    Returns
    -------
    Phi_0, Psi_0, u_grid, delta_u, coherence, W_p, n_ant
    """
    # Uniform spatial frequency grid u_n in [-1, 1 - 2/N]
    u_grid = np.linspace(-1.0, 1.0 - 2.0 / N, N)
    delta_u = u_grid[1] - u_grid[0]

    # Virtual array indices m = 0, ..., n_ant-1 (compressed pilot projection)
    n_ant = N // 2
    m_idx = np.arange(n_ant).reshape(-1, 1)

    # Full steering matrix A_full (n_ant x N) and derivative dA_full / du
    A_full = (1.0 / np.sqrt(n_ant)) * np.exp(-1j * np.pi * m_idx * u_grid.reshape(1, -1))
    dA_full = (-1j * np.pi * m_idx) * A_full

    # Unitary random pilot combining matrix W_p (n_ant x M)
    G_rand = (np.random.randn(n_ant, M) + 1j * np.random.randn(n_ant, M)) / np.sqrt(2)
    W_p, _ = np.linalg.qr(G_rand)

    Phi_0 = W_p.conj().T @ A_full
    Psi_0 = W_p.conj().T @ dA_full

    # Normalize columns of Phi_0 to unit norm and scale Psi_0 accordingly
    col_norms = np.linalg.norm(Phi_0, axis=0, keepdims=True)
    Phi_0 = Phi_0 / col_norms
    Psi_0 = Psi_0 / col_norms

    # Mutual coherence mu(Phi_0)
    gram = np.abs(Phi_0.conj().T @ Phi_0)
    np.fill_diagonal(gram, 0.0)
    coherence = np.max(gram)

    return Phi_0, Psi_0, u_grid, delta_u, coherence, W_p, n_ant
