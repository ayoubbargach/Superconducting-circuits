"""
Reproduce Figure 17 from PRX Quantum 2, 040204 (2021)
"Superconducting Circuit Companion"

Energy levels of the Single Cooper-Pair Box / Transmon qubit
as a function of the bias charge n_g for different E_J/E_C ratios.

Hamiltonian in the charge basis {|n>}:
  H = 4 E_C sum_n (n - n_g)^2 |n><n| - (E_J/2) sum_n (|n><n+1| + |n+1><n|)

This is a tridiagonal matrix that we diagonalize numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def build_hamiltonian(n_g, EJ_over_EC, n_max=30):
    """
    Build the CPB Hamiltonian matrix in the charge basis.
    
    Parameters
    ----------
    n_g : float
        Gate charge (offset charge) in units of 2e.
    EJ_over_EC : float
        Ratio E_J / E_C.
    n_max : int
        Truncation: n ranges from -n_max to +n_max.
        Matrix size is (2*n_max + 1) x (2*n_max + 1).
    
    Returns
    -------
    H : ndarray
        Hamiltonian matrix in units of E_C (i.e. we set E_C = 1).
    """
    dim = 2 * n_max + 1
    n_values = np.arange(-n_max, n_max + 1)
    
    # Diagonal: 4 * E_C * (n - n_g)^2, with E_C = 1
    diag = 4.0 * (n_values - n_g) ** 2
    
    # Off-diagonal: -E_J / 2 = -EJ_over_EC * E_C / 2
    off_diag = -EJ_over_EC / 2.0 * np.ones(dim - 1)
    
    H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return H


def compute_spectrum(n_g_array, EJ_over_EC, n_levels=5, n_max=30):
    """
    Compute the lowest energy levels as a function of n_g.
    
    Returns energies normalized by E_01 = E_1 - E_0 at n_g = 0.5
    (following the paper's convention for the y-axis).
    """
    energies = np.zeros((len(n_g_array), n_levels))
    
    for i, n_g in enumerate(n_g_array):
        H = build_hamiltonian(n_g, EJ_over_EC, n_max)
        evals = np.linalg.eigvalsh(H)
        energies[i, :] = evals[:n_levels]
    
    # Shift so that ground state energy at n_g=0 is zero
    energies -= energies[:, 0].min()
    
    # Normalize by E_01 at n_g closest to 0 (or 0.5 depending on convention)
    # The paper normalizes by E_01. We use E_01 at the minimum gap point.
    idx_half = np.argmin(np.abs(n_g_array - 0.5))
    E01 = energies[idx_half, 1] - energies[idx_half, 0]
    
    if E01 > 1e-10:
        energies /= E01
    
    return energies


# ---------- Parameters ----------
n_g_array = np.linspace(-2, 2, 500)
n_levels = 7

# The four panels of Figure 17
EJ_EC_values = [0.01, 1.0, 3.0, 10.0]
panel_labels = [
    r"(a) $E_J/E_C \ll 1$",
    r"(b) $E_J/E_C = 1.0$",
    r"(c) $E_J/E_C = 3.0$",
    r"(d) $E_J/E_C = 10.0$",
]

# Colors matching typical PRX style (added 2 more colors)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# ---------- Plot ----------
fig = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

for idx, (EJ_EC, label) in enumerate(zip(EJ_EC_values, panel_labels)):
    ax = fig.add_subplot(gs[idx])
    
    # Adjust n_max for convergence at large EJ/EC
    n_max = 10 if EJ_EC < 5 else 30
    
    energies = compute_spectrum(n_g_array, EJ_EC, n_levels, n_max)
    
    for level in range(n_levels):
        ax.plot(n_g_array, energies[:, level], color=colors[level], lw=1.5)
    
    ax.set_xlabel(r"$n_g$", fontsize=13)
    ax.set_ylabel(r"$E / E_{01}$", fontsize=13)
    ax.set_title(label, fontsize=13)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)
    ax.tick_params(labelsize=11)

fig.suptitle(
    "Energy levels of the Cooper-pair box / Transmon",
    fontsize=14, y=0.98,
)

plt.savefig("figure.png", dpi=200, bbox_inches="tight")
plt.savefig("figure.pdf", bbox_inches="tight")
print("Figure saved.")
