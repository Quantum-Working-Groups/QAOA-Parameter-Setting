"""Figure: depth-one QAOA energy landscape of the 12-spin LABS instance.

Scans the energy of exp(-i beta X) exp(-i gamma H_C) |+> over the whole (gamma, beta) grid in one
pass.  Independent of the saved runs -- a property of the instance.  Saves landscape.pdf here.
"""

import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from labs_qaoa import labs_energies, xp

if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
plt.rcParams.update({"axes.labelsize": "small", "xtick.labelsize": "small", "ytick.labelsize": "small"})

n = 12
e = xp.asarray(labs_energies(n))
grid = np.linspace(-np.pi / 2, np.pi / 2, 201)
g = xp.asarray(grid)

# psi[b, m] is the state at beta = grid[b], gamma = grid[m]: mix(beta) applied to exp(-i gamma H)|+>
psi = xp.broadcast_to(xp.exp(-1j * g[None, :, None] * e) * e.size ** -0.5, (grid.size, grid.size, e.size)).copy()
cos, sin = xp.cos(g).reshape(-1, 1, 1, 1), (-1j * xp.sin(g)).reshape(-1, 1, 1, 1)
for q in range(n):
    v = psi.reshape(grid.size, grid.size, -1, 2, 2 ** q)
    lo = v[:, :, :, 0, :].copy()
    v[:, :, :, 0, :] = cos * lo + sin * v[:, :, :, 1, :]
    v[:, :, :, 1, :] = sin * lo + cos * v[:, :, :, 1, :]
energy = (xp.abs(psi) ** 2 * e).sum(-1)
energy = xp.asnumpy(energy) if xp is not np else energy

fig, ax = plt.subplots(figsize=(3.6, 2.9))
cs = ax.contourf(grid, grid, energy, levels=30)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\beta$")
fig.colorbar(cs, ax=ax, label="Energy")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "landscape.pdf")
