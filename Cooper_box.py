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
import os

# Create Results directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

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
    # Adjust n_levels if n_max is too small
    max_dim = 2 * n_max + 1
    n_levels_actual = min(n_levels, max_dim)

    energies = np.zeros((len(n_g_array), n_levels_actual))

    for i, n_g in enumerate(n_g_array):
        H = build_hamiltonian(n_g, EJ_over_EC, n_max)
        evals = np.linalg.eigvalsh(H)
        energies[i, :] = evals[:n_levels_actual]

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
EJ_EC_values = [0.9, 1.0, 3.0, 10.0]
panel_labels = [
    r"(a) $E_J/E_C = 0.9$",
    r"(b) $E_J/E_C = 1.0$",
    r"(c) $E_J/E_C = 3.0$",
    r"(d) $E_J/E_C = 10.0$",
]

# Colors matching typical PRX style (added 2 more colors)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# ---------- Plot 1: Original figure ----------
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

plt.savefig("Results/figure.png", dpi=200, bbox_inches="tight")
print("Figure saved: Results/figure.png")

# ---------- Plot 2: 10 different Ej/Ec ratios with fixed truncation ----------
EJ_EC_range = np.logspace(np.log10(0.1), np.log10(9.9), 10)
n_max_fixed = 30
n_levels_range = 30  # Show all 30 levels

fig_range = plt.figure(figsize=(15, 12))
gs_range = GridSpec(2, 5, hspace=0.35, wspace=0.3)

for idx, EJ_EC in enumerate(EJ_EC_range):
    ax = fig_range.add_subplot(gs_range[idx])

    energies = compute_spectrum(n_g_array, EJ_EC, n_levels_range, n_max_fixed)

    n_levels_actual = energies.shape[1]
    # Generate colors using a colormap for many levels
    cmap = plt.colormaps.get_cmap('tab20')
    cmap2 = plt.colormaps.get_cmap('tab20b')
    for level in range(n_levels_actual):
        color = cmap(level % 20) if level < 20 else cmap2((level - 20) % 20)
        ax.plot(n_g_array, energies[:, level], color=color, lw=1.0)

    ax.set_xlabel(r"$n_g$", fontsize=11)
    ax.set_ylabel(r"$E / E_{01}$", fontsize=11)
    ax.set_title(f"$E_J/E_C = {EJ_EC:.2f}$", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 50)
    ax.tick_params(labelsize=9)

fig_range.suptitle(
    f"Energy levels for different $E_J/E_C$ ratios (fixed $n_{{max}} = {n_max_fixed}$)",
    fontsize=14, y=0.98,
)

plt.savefig("Results/figure_EJ_EC_range.png", dpi=200, bbox_inches="tight")
print("Figure saved: Results/figure_EJ_EC_range.png")

# ---------- Plot 2b: 10 different Ej/Ec ratios (higher range) with fixed truncation ----------
EJ_EC_range_high = np.logspace(np.log10(1.1), np.log10(100), 10)
n_max_fixed_high = 30
n_levels_range_high = 30  # Show all 30 levels

fig_range_high = plt.figure(figsize=(15, 12))
gs_range_high = GridSpec(2, 5, hspace=0.35, wspace=0.3)

for idx, EJ_EC in enumerate(EJ_EC_range_high):
    ax = fig_range_high.add_subplot(gs_range_high[idx])

    energies = compute_spectrum(n_g_array, EJ_EC, n_levels_range_high, n_max_fixed_high)

    n_levels_actual = energies.shape[1]
    # Generate colors using a colormap for many levels
    cmap = plt.colormaps.get_cmap('tab20')
    cmap2 = plt.colormaps.get_cmap('tab20b')
    for level in range(n_levels_actual):
        color = cmap(level % 20) if level < 20 else cmap2((level - 20) % 20)
        ax.plot(n_g_array, energies[:, level], color=color, lw=1.0)

    ax.set_xlabel(r"$n_g$", fontsize=11)
    ax.set_ylabel(r"$E / E_{01}$", fontsize=11)
    ax.set_title(f"$E_J/E_C = {EJ_EC:.2f}$", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 50)
    ax.tick_params(labelsize=9)

fig_range_high.suptitle(
    f"Energy levels for different $E_J/E_C$ ratios (fixed $n_{{max}} = {n_max_fixed_high}$)",
    fontsize=14, y=0.98,
)

plt.savefig("Results/figure_EJ_EC_range_high.png", dpi=200, bbox_inches="tight")
print("Figure saved: Results/figure_EJ_EC_range_high.png")

# ---------- Plot 3: Truncation comparison for each EJ/EC ----------
truncations = [10, 30, 50, 100]
truncation_labels = [
    r"(a) $n_{max} = 10$",
    r"(b) $n_{max} = 30$",
    r"(c) $n_{max} = 50$",
    r"(d) $n_{max} = 100$",
]

for EJ_EC in EJ_EC_values:
    fig_trunc = plt.figure(figsize=(10, 9))
    gs_trunc = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for idx, (n_max, trunc_label) in enumerate(zip(truncations, truncation_labels)):
        ax = fig_trunc.add_subplot(gs_trunc[idx])

        energies = compute_spectrum(n_g_array, EJ_EC, n_levels, n_max)

        # Plot only the available energy levels
        n_levels_actual = energies.shape[1]
        for level in range(n_levels_actual):
            ax.plot(n_g_array, energies[:, level], color=colors[level], lw=1.5)

        ax.set_xlabel(r"$n_g$", fontsize=13)
        ax.set_ylabel(r"$E / E_{01}$", fontsize=13)
        ax.set_title(trunc_label, fontsize=13)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        ax.tick_params(labelsize=11)

    fig_trunc.suptitle(
        f"Truncation comparison for $E_J/E_C = {EJ_EC}$",
        fontsize=14, y=0.98,
    )

    filename = f"Results/figure_truncation_EJ_EC_{EJ_EC}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {filename}")

plt.close('all')
print("\nAll truncation comparison figures saved.")
