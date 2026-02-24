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
try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    print("Warning: schemdraw not available, circuit diagram will be skipped")

# Create Results directory if it doesn't exist
# Get the name of the current python file without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = f"Results_{script_name}"
os.makedirs(results_dir, exist_ok=True)

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

plt.savefig(f"{results_dir}/figure.png", dpi=200, bbox_inches="tight")
print(f"Figure saved: {results_dir}/figure.png")

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

plt.savefig(f"{results_dir}/figure_EJ_EC_range.png", dpi=200, bbox_inches="tight")
print(f"Figure saved: {results_dir}/figure_EJ_EC_range.png")

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

plt.savefig(f"{results_dir}/figure_EJ_EC_range_high.png", dpi=200, bbox_inches="tight")
print(f"Figure saved: {results_dir}/figure_EJ_EC_range_high.png")

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

    filename = f"{results_dir}/figure_truncation_EJ_EC_{EJ_EC}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {filename}")

plt.close('all')
print("\nAll truncation comparison figures saved.")

# ---------- Plot 4: Energy levels vs truncation for fixed n_g = 0.5 ----------
n_g_fixed = 0.5
truncation_range = np.arange(10, 101, 2)  # From 10 to 100 by steps of 2
# Choose 4 representative Ej/Ec values: charge regime, transition, transmon, deep transmon
EJ_EC_convergence = [0.5, 1.0, 5.0, 20.0]
n_levels_convergence = 10  # Track first 10 energy levels

fig_conv = plt.figure(figsize=(12, 10))
gs_conv = GridSpec(2, 2, hspace=0.35, wspace=0.3)

for idx, EJ_EC in enumerate(EJ_EC_convergence):
    ax = fig_conv.add_subplot(gs_conv[idx])

    # Store energies for each truncation
    energies_vs_trunc = np.zeros((len(truncation_range), n_levels_convergence))

    for i, n_max in enumerate(truncation_range):
        # Compute spectrum at fixed n_g
        max_dim = 2 * n_max + 1
        n_levels_actual = min(n_levels_convergence, max_dim)

        H = build_hamiltonian(n_g_fixed, EJ_EC, n_max)
        evals = np.linalg.eigvalsh(H)

        # Store first n_levels_actual energy levels
        energies_vs_trunc[i, :n_levels_actual] = evals[:n_levels_actual]
        # Mark unavailable levels as NaN
        if n_levels_actual < n_levels_convergence:
            energies_vs_trunc[i, n_levels_actual:] = np.nan

    # Shift so ground state is at zero
    energies_vs_trunc -= np.nanmin(energies_vs_trunc[:, 0])

    # Normalize by E_01 at largest truncation
    E01_ref = energies_vs_trunc[-1, 1] - energies_vs_trunc[-1, 0]
    if E01_ref > 1e-10:
        energies_vs_trunc /= E01_ref

    # Plot each energy level vs truncation
    for level in range(n_levels_convergence):
        ax.plot(truncation_range, energies_vs_trunc[:, level],
                color=colors[level] if level < len(colors) else f'C{level}',
                lw=1.5, marker='o', markersize=2, label=f'E_{level}')

    # Find min and max values for the highest energy level (E_9)
    highest_level = n_levels_convergence - 1
    min_val = np.nanmin(energies_vs_trunc[:, highest_level])
    max_val = np.nanmax(energies_vs_trunc[:, highest_level])

    # Add text annotations for min and max values of highest level
    ax.text(0.02, 0.98, f'E_{highest_level} Min: {min_val:.4f}\nE_{highest_level} Max: {max_val:.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(r"$n_{max}$ (truncation)", fontsize=12)
    ax.set_ylabel(r"$E / E_{01}$", fontsize=12)
    ax.set_title(f"$E_J/E_C = {EJ_EC}$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    if idx == 0:
        ax.legend(fontsize=8, ncol=2, loc='best')

fig_conv.suptitle(
    f"Energy level convergence vs truncation at $n_g = {n_g_fixed}$",
    fontsize=14, y=0.98,
)

plt.savefig(f"{results_dir}/figure_convergence_vs_truncation.png", dpi=200, bbox_inches="tight")
print(f"Figure saved: {results_dir}/figure_convergence_vs_truncation.png")

# ---------- Plot 5: Highest energy level only vs truncation (log scale) ----------
fig_conv_log = plt.figure(figsize=(12, 10))
gs_conv_log = GridSpec(2, 2, hspace=0.35, wspace=0.3)

for idx, EJ_EC in enumerate(EJ_EC_convergence):
    ax = fig_conv_log.add_subplot(gs_conv_log[idx])

    # Store energies for each truncation
    energies_vs_trunc_log = np.zeros((len(truncation_range), n_levels_convergence))

    for i, n_max in enumerate(truncation_range):
        # Compute spectrum at fixed n_g
        max_dim = 2 * n_max + 1
        n_levels_actual = min(n_levels_convergence, max_dim)

        H = build_hamiltonian(n_g_fixed, EJ_EC, n_max)
        evals = np.linalg.eigvalsh(H)

        # Store first n_levels_actual energy levels
        energies_vs_trunc_log[i, :n_levels_actual] = evals[:n_levels_actual]
        # Mark unavailable levels as NaN
        if n_levels_actual < n_levels_convergence:
            energies_vs_trunc_log[i, n_levels_actual:] = np.nan

    # Shift so ground state is at zero
    energies_vs_trunc_log -= np.nanmin(energies_vs_trunc_log[:, 0])

    # Normalize by E_01 at largest truncation
    E01_ref = energies_vs_trunc_log[-1, 1] - energies_vs_trunc_log[-1, 0]
    if E01_ref > 1e-10:
        energies_vs_trunc_log /= E01_ref

    # Plot only the highest energy level vs truncation
    highest_level = n_levels_convergence - 1
    ax.semilogy(truncation_range, energies_vs_trunc_log[:, highest_level],
                color=colors[highest_level % len(colors)],
                lw=2, marker='o', markersize=3, label=f'E_{highest_level}')

    # Find min and max values for the highest energy level
    min_val = np.nanmin(energies_vs_trunc_log[:, highest_level])
    max_val = np.nanmax(energies_vs_trunc_log[:, highest_level])

    # Add text annotations for min and max values
    ax.text(0.02, 0.98, f'E_{highest_level} Min: {min_val:.4f}\nE_{highest_level} Max: {max_val:.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(r"$n_{max}$ (truncation)", fontsize=12)
    ax.set_ylabel(r"$E / E_{01}$ (log scale)", fontsize=12)
    ax.set_title(f"$E_J/E_C = {EJ_EC}$", fontsize=13)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='best')

fig_conv_log.suptitle(
    f"Highest energy level (E_{n_levels_convergence-1}) convergence vs truncation at $n_g = {n_g_fixed}$ (log scale)",
    fontsize=14, y=0.98,
)

plt.savefig(f"{results_dir}/figure_convergence_vs_truncation_log.png", dpi=200, bbox_inches="tight")
print(f"Figure saved: {results_dir}/figure_convergence_vs_truncation_log.png")

plt.close('all')
print("\nAll figures saved.")

# ---------- Plot 6: Cooper-pair box circuit diagram ----------
# Circuit: mass + Vg + Cg + (Ej * Cj) + mass
# Where + means series and * means parallel
if SCHEMDRAW_AVAILABLE:
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=12, font='sans-serif')

        # Start with ground/mass on the left
        d += elm.Ground()
        d += elm.Line().up().length(0.5)
        d += (gnd_left := elm.Dot())

        # Vg (voltage source) in series
        d += elm.SourceV().up().label('$V_g$\n$(n_g)$', loc='left')

        # Cg (gate capacitor) in series
        d += elm.Capacitor().up().label('$C_g$', loc='left')
        d += (after_cg := elm.Dot())

        # Now the parallel combination (Ej * Cj)
        # Continue upward with first branch (Ej)
        d += elm.Line().up().length(0.5)
        d += (parallel_top := elm.Dot())

        # First parallel element: Josephson junction
        d += elm.Line().right().length(1)
        d += (branch_ej_top := elm.Dot())
        d += elm.Josephson().down().length(2).label('$E_J$', loc='right')
        d += (branch_ej_bottom := elm.Dot())
        d += elm.Line().right().length(1)
        d += (parallel_bottom := elm.Dot())

        # Second parallel element: Cj (capacitor)
        d.move_from(branch_ej_top.start, dx=1.5, dy=0)
        d += elm.Dot()
        d += elm.Capacitor().down().length(2).label('$C_J$ $(E_C)$', loc='right')
        d += elm.Dot()
        d += elm.Line().to(parallel_bottom.start)

        # Connect top of parallel
        d.move_from(parallel_top.start, dx=0, dy=0)
        d += elm.Line().to(branch_ej_top.start)

        # Continue to final ground/mass
        d.move_from(parallel_bottom.start, dx=0, dy=0)
        d += elm.Line().down().length(0.5)
        d += elm.Ground()

        # Add title
        d += elm.Label().at((2, 5.5)).label(r'Cooper-Pair Box: $\bot + V_g + C_g + (E_J \parallel C_J) + \bot$',
                                           fontsize=12, halign='center')

        d.save(f'{results_dir}/circuit_diagram.png', dpi=300)
    print(f"Circuit diagram saved: {results_dir}/circuit_diagram.png")
else:
    print("Schemdraw not available, circuit diagram skipped")

print("\n=== All outputs generated ===")
