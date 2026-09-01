"""Statevector evaluation of the saved LABS QAOA angles.

The LABS cost Hamiltonian is diagonal in the computational basis, so a state is
propagated by alternating a diagonal phase (the sidelobe energies) with the
transverse-field mixer.  This avoids building the depth-50 circuit and keeps the
whole table/figure sweep tractable up to the 21-spin instances used in the paper.

Angles follow the qiskit qaoa_ansatz layout, [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}].
The propagation runs on the GPU through cupy when it is available and on numpy otherwise.
"""

import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np


def labs_energies(n):
    """Sidelobe energy E(s) = sum_{k>=1} (sum_i s_i s_{i+k})^2 of every length-n sequence.

    Indexed by computational basis state, with x_i = 0, 1 mapped to s_i = +1, -1.
    """
    s = np.where((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1, -1, 1).astype(np.int32)
    e = np.zeros(2 ** n)
    for k in range(1, n):
        c = (s[:, : n - k] * s[:, k:]).sum(1)
        e += c * c
    return e


def _mix(psi, beta):
    # exp(-i beta X) on each qubit in turn, in place
    c, s = np.cos(beta), -1j * np.sin(beta)
    n = psi.shape[0].bit_length() - 1
    for j in range(n):
        v = psi.reshape(-1, 2, 2 ** j)
        lo = v[:, 0].copy()
        v[:, 0] = c * lo + s * v[:, 1]
        v[:, 1] = s * lo + c * v[:, 1]
    return psi


def qaoa_state(energies, angles):
    """Statevector of the p-layer QAOA circuit; returned on the host as a numpy array."""
    e = xp.asarray(energies)
    p = len(angles) // 2
    betas, gammas = angles[:p], angles[p:]
    psi = xp.full(len(e), len(e) ** -0.5, dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = _mix(psi * xp.exp(-1j * gamma * e), beta)
    return xp.asnumpy(psi) if xp is not np else psi


def energy_mf_tts(energies, angles):
    """Return (energy, mean merit factor, time-to-solution) of the QAOA state."""
    n = len(energies).bit_length() - 1
    prob = np.abs(qaoa_state(energies, angles)) ** 2
    p_opt = prob[energies == energies.min()].sum()
    mf = (prob * n * n / (2 * energies)).sum()
    tts = 1 / p_opt if p_opt > 0 else np.inf
    return prob @ energies, mf, tts
