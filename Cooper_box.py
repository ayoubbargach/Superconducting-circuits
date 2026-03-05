"""
Cooper-Pair Box / Transmon Energy Level Analysis

Analyzes energy levels of the Single Cooper-Pair Box / Transmon qubit
as a function of the bias charge n_g for different E_J/E_C ratios.

Hamiltonian in the charge basis {|n>}:
  H = 4 E_C sum_n (n - n_g)^2 |n><n| - (E_J/2) sum_n (|n><n+1| + |n+1><n|)

This is a tridiagonal matrix that we diagonalize numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from circuit_diagram_cooper_box import draw_cooper_box_circuit


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


def compute_spectrum(n_g_array, EJ_over_EC, n_levels=5, n_max=30, normalize=True):
    """
    Compute the lowest energy levels as a function of n_g.

    Parameters
    ----------
    n_g_array : array
        Array of gate charge values.
    EJ_over_EC : float
        Ratio E_J / E_C.
    n_levels : int
        Number of energy levels to compute.
    n_max : int
        Truncation parameter.
    normalize : bool
        If True, normalize energies by E_01 at n_g = 0.5 (paper convention).
        If False, return raw energies in units of E_C (useful for convergence studies).

    Returns
    -------
    energies : ndarray
        Energy levels as a function of n_g.
        If normalize=True: energies normalized by E_01 = E_1 - E_0 at n_g = 0.5.
        If normalize=False: raw energies in units of E_C, shifted so ground state minimum is zero.
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

    if normalize:
        # Normalize by E_01 at n_g closest to 0.5 (paper convention)
        idx_half = np.argmin(np.abs(n_g_array - 0.5))
        E01 = energies[idx_half, 1] - energies[idx_half, 0]

        if E01 > 1e-10:
            energies /= E01

    return energies


def analyze_cooper_box(results_dir='Results_Cooper_box'):
    """
    Perform Cooper box analysis and generate all plots.

    Parameters
    ----------
    results_dir : str
        Directory where results will be saved
    """
    # Create Results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

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
    print("Generating main energy level plots...")
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
    plt.close()

    # ---------- Plot 2: 4 different Ej/Ec ratios with fixed truncation ----------
    print("\nGenerating EJ/EC range plots...")
    EJ_EC_range = np.logspace(np.log10(0.1), np.log10(9.9), 4)
    n_max_fixed = 30
    n_levels_range = 30  # Show all 30 levels

    fig_range = plt.figure(figsize=(12, 8))
    gs_range = GridSpec(2, 2, hspace=0.35, wspace=0.3)

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
    print("\nGenerating truncation comparison plots...")
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

    # ---------- Plot 4: Energy levels vs truncation for fixed n_g = 0.5 (UNNORMALIZED) ----------
    print("\nGenerating convergence analysis plots (unnormalized)...")
    n_g_fixed = 0.5
    truncation_range = np.arange(1, 21, 1)  # From 1 to 20 by steps of 1 - THIS IS THE KEY CHANGE
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

            # Store first n_levels_actual energy levels (RAW, no normalization)
            energies_vs_trunc[i, :n_levels_actual] = evals[:n_levels_actual]
            # Mark unavailable levels as NaN
            if n_levels_actual < n_levels_convergence:
                energies_vs_trunc[i, n_levels_actual:] = np.nan

        # Shift so ground state is at zero
        energies_vs_trunc -= np.nanmin(energies_vs_trunc[:, 0])

        # NO NORMALIZATION - this is the key change!

        # Plot each energy level vs truncation
        for level in range(n_levels_convergence):
            ax.plot(truncation_range, energies_vs_trunc[:, level],
                    color=colors[level] if level < len(colors) else f'C{level}',
                    lw=1.5, marker='o', markersize=2, label=f'E_{level}')

        # Find min and max values for the highest energy level (E_9)
        highest_level = n_levels_convergence - 1
        min_val = np.nanmin(energies_vs_trunc[:, highest_level])
        max_val = np.nanmax(energies_vs_trunc[:, highest_level])
        variation = max_val - min_val
        rel_variation = (variation / max_val * 100) if max_val > 0 else 0

        # Add text annotations showing variation
        ax.text(0.02, 0.98,
                f'E_{highest_level} Min: {min_val:.4f} E_C\nE_{highest_level} Max: {max_val:.4f} E_C\nVariation: {variation:.4f} E_C ({rel_variation:.2f}%)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(r"$n_{max}$ (truncation)", fontsize=12)
        ax.set_ylabel(r"$E$ (units of $E_C$)", fontsize=12)
        ax.set_title(f"$E_J/E_C = {EJ_EC}$", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2, loc='best')

    fig_conv.suptitle(
        f"Energy level convergence vs truncation at $n_g = {n_g_fixed}$ (UNNORMALIZED)",
        fontsize=14, y=0.98,
    )

    plt.savefig(f"{results_dir}/figure_convergence_vs_truncation_unnormalized.png", dpi=200, bbox_inches="tight")
    print(f"Figure saved: {results_dir}/figure_convergence_vs_truncation_unnormalized.png")

    # ---------- Plot 5: Convergence error (log scale) showing exponential convergence ----------
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

            # Store first n_levels_actual energy levels (RAW, no normalization)
            energies_vs_trunc_log[i, :n_levels_actual] = evals[:n_levels_actual]
            # Mark unavailable levels as NaN
            if n_levels_actual < n_levels_convergence:
                energies_vs_trunc_log[i, n_levels_actual:] = np.nan

        # Shift so ground state is at zero
        energies_vs_trunc_log -= np.nanmin(energies_vs_trunc_log[:, 0])

        # Plot absolute error from converged value (last n_max) for multiple levels
        for level in range(0, n_levels_convergence, 2):  # Plot every other level to avoid clutter
            E_converged = energies_vs_trunc_log[-1, level]
            abs_error = np.abs(energies_vs_trunc_log[:, level] - E_converged)

            # Replace zeros with a small number to avoid log issues
            abs_error = np.where(abs_error < 1e-14, 1e-14, abs_error)

            ax.semilogy(truncation_range, abs_error,
                        color=colors[level % len(colors)],
                        lw=1.5, marker='o', markersize=3, label=f'E_{level}')

        # Add machine precision line
        ax.axhline(1e-10, color='red', linestyle='--', linewidth=1.5,
                   label='Numerical precision', alpha=0.7)

        # Find where ground state converges
        E0_converged = energies_vs_trunc_log[-1, 0]
        E0_error = np.abs(energies_vs_trunc_log[:, 0] - E0_converged)
        converged_idx = np.argmax(E0_error < 1e-10)
        if converged_idx > 0:
            converged_nmax = truncation_range[converged_idx]
            ax.axvline(converged_nmax, color='green', linestyle='--', linewidth=1.5,
                       alpha=0.5, label=f'E_0 converged at n_max={converged_nmax}')

        ax.set_xlabel(r"$n_{max}$ (truncation)", fontsize=12)
        ax.set_ylabel(r"$|E - E_{converged}|$ (units of $E_C$, log scale)", fontsize=12)
        ax.set_title(f"$E_J/E_C = {EJ_EC}$", fontsize=13)
        ax.set_ylim(1e-14, 1e0)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=10)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2, loc='best')

    fig_conv_log.suptitle(
        f"Exponential convergence of energy levels vs truncation at $n_g = {n_g_fixed}$",
        fontsize=14, y=0.98,
    )

    plt.savefig(f"{results_dir}/figure_convergence_exponential.png", dpi=200, bbox_inches="tight")
    print(f"Figure saved: {results_dir}/figure_convergence_exponential.png")

    plt.close('all')
    print("\nAll figures saved.")

    # Draw circuit diagram
    print("\nGenerating circuit diagram...")
    draw_cooper_box_circuit(results_dir)

    print("\n=== Cooper box analysis complete ===")


if __name__ == '__main__':
    analyze_cooper_box()
