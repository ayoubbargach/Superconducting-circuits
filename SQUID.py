"""
SQUID (Superconducting Quantum Interference Device) Analysis
Phase basis implementation with Mathieu functions

The SQUID Hamiltonian is:
    H = 4*E_C*n^2 - E_eff*cos(φ_ext - φ)

Where:
    n: Phase momentum operator (conjugate to φ)
    φ: Phase operator across the SQUID loop
    φ_ext: External flux phase = 2π*Φ_ext/Φ_0

In the phase/momentum basis, this is solved using periodic boundary conditions.
The eigenfunctions are related to Mathieu functions.

Physical parameters:
    E_C: Charging energy = e^2/(2*C_e) where C_e = C + C_g
    E_eff: Effective Josephson energy = E_J1 + E_J2 (for symmetric SQUID)
    φ_ext: External flux phase = 2π*Φ_ext/Φ_0

Implementation:
    - Phase momentum basis |n⟩ with n ∈ [-n_max, ..., n_max]
    - Matrix elements: ⟨n|H|m⟩ = 4E_C n² δ_nm - (E_eff/2)[e^(iφ_ext)δ_{n,m-1} + e^(-iφ_ext)δ_{n,m+1}]
    - This is the Fourier representation in phase space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from circuit_diagram_squid import draw_squid_circuit

try:
    from scipy.special import mathieu_a, mathieu_b
    MATHIEU_AVAILABLE = True
except ImportError:
    MATHIEU_AVAILABLE = False
    print("Warning: scipy.special.mathieu functions not available")


def build_hamiltonian_phase_basis(phi_ext, EJ_over_EC, n_max=50):
    """
    Build SQUID Hamiltonian in phase momentum basis |n⟩ with truncation.

    In the phase representation, we use momentum eigenstates where:
        H = 4*E_C*n^2 - E_eff*cos(φ_ext - φ)

    Matrix elements in phase momentum basis |n⟩:
        ⟨n|H|m⟩ = 4*E_C*n^2 δ_nm - (E_eff/2)[e^(i*φ_ext)δ_{n,m-1} + e^(-i*φ_ext)δ_{n,m+1}]

    This gives a tridiagonal Hermitian matrix:
        - Diagonal: 4*E_C*n^2 (kinetic energy in phase space)
        - Off-diagonal: -(E_J/2)*e^(±iφ_ext) (Josephson potential with flux modulation)

    The eigenstates are related to Mathieu functions with periodic boundary conditions.

    Parameters
    ----------
    phi_ext : float
        External flux phase (2π*Φ_ext/Φ_0)
    EJ_over_EC : float
        Ratio E_J/E_C (E_eff = E_J for symmetric SQUID)
    n_max : int
        Truncation: phase momentum from -n_max to n_max

    Returns
    -------
    H : ndarray
        Hamiltonian matrix in phase momentum basis (2*n_max+1 × 2*n_max+1)
    """
    dim = 2 * n_max + 1
    H = np.zeros((dim, dim), dtype=complex)

    # Basis states: |n⟩ with n ∈ [-n_max, ..., 0, ..., n_max]
    n_values = np.arange(-n_max, n_max + 1)

    for i, n in enumerate(n_values):
        # Diagonal: kinetic energy 4*E_C*n^2
        H[i, i] = 4.0 * n**2

        # Off-diagonal: Josephson coupling -E_J*cos(φ_ext - φ)
        # In momentum basis: -E_J/2 * [e^(i*φ_ext)|n-1⟩ + e^(-i*φ_ext)|n+1⟩]
        if i > 0:  # n-1 exists
            H[i, i-1] = -EJ_over_EC / 2.0 * np.exp(1j * phi_ext)
        if i < dim - 1:  # n+1 exists
            H[i, i+1] = -EJ_over_EC / 2.0 * np.exp(-1j * phi_ext)

    return H


def compute_spectrum(phi_ext_array, EJ_over_EC, n_levels=5, n_max=50):
    """
    Compute energy spectrum as a function of external flux using numerical diagonalization.

    Parameters
    ----------
    phi_ext_array : array
        Array of external flux phases (2π*Φ_ext/Φ_0)
    EJ_over_EC : float
        Ratio E_J/E_C
    n_levels : int
        Number of energy levels to compute
    n_max : int
        Phase momentum truncation (n ∈ [-n_max, n_max])

    Returns
    -------
    energies : ndarray
        Energy levels (shape: len(phi_ext_array) × n_levels) in units of E_C
    """
    max_dim = 2 * n_max + 1
    n_levels_actual = min(n_levels, max_dim)

    energies = np.zeros((len(phi_ext_array), n_levels_actual))

    for i, phi_ext in enumerate(phi_ext_array):
        H = build_hamiltonian_phase_basis(phi_ext, EJ_over_EC, n_max)
        evals = np.linalg.eigvalsh(H)
        energies[i, :] = evals[:n_levels_actual].real

    # Shift so ground state minimum is at zero
    energies -= energies[:, 0].min()

    return energies


def compute_spectrum_mathieu(phi_ext_array, EJ_over_EC, n_levels=5):
    """
    Compute energy spectrum using analytical Mathieu functions.

    NOTE: This implementation shows the THEORETICAL connection between the SQUID
    problem and Mathieu functions, but it requires a flux-dependent effective parameter
    for a DC SQUID. For a single-junction SQUID (or treating the SQUID in the phase
    representation without flux modulation), the Mathieu equation provides exact solutions.

    The SQUID Schrödinger equation in phase representation (single effective junction):
        -d²ψ/dφ² - (E_J/E_C) cos(φ) ψ = (E/E_C) ψ

    The standard Mathieu equation is:
        d²y/dz² + (a - 2q cos(2z)) y = 0

    To match forms, we need to transform coordinates. Using z = φ/2:
        -4 d²ψ/dz² - (E_J/E_C) cos(2z) ψ = (E/E_C) ψ
        d²ψ/dz² + [E/(4E_C) + (E_J/E_C)/4 cos(2z)] ψ = 0

    This doesn't quite match the standard form. The proper connection requires:
        d²ψ/dφ² + [E/E_C + (E_J/E_C) cos(φ)] ψ = 0

    Rewriting with φ = 2z:
        4 d²ψ/dz² + [E/E_C + (E_J/E_C) cos(2z)] ψ = 0
        d²ψ/dz² + [E/(4E_C) + (E_J/E_C)/4 cos(2z)] ψ = 0

    Comparing with standard form d²y/dz² + (a - 2q cos(2z)) y = 0:
        a = E/(4E_C)
        -2q = (E_J/E_C)/4  =>  q = -E_J/(8E_C)

    Since characteristic values come with sign, we use q = E_J/(8E_C) and:
        E_n = 4*E_C * a_n(q)

    For a DC SQUID, external flux modulates the effective E_J.

    Parameters
    ----------
    phi_ext_array : array
        Array of external flux phases (2π*Φ_ext/Φ_0)
    EJ_over_EC : float
        Ratio E_J/E_C for a single junction (for DC SQUID, this is per junction)
    n_levels : int
        Number of energy levels to compute

    Returns
    -------
    energies : ndarray
        Energy levels (shape: len(phi_ext_array) × n_levels) in units of E_C
        Returns None if Mathieu functions are not available
    """
    if not MATHIEU_AVAILABLE:
        print("Warning: Mathieu functions not available")
        return None

    energies = np.zeros((len(phi_ext_array), n_levels))

    # For a DC SQUID, the effective Josephson coupling depends on flux:
    # E_J_eff(Φ) = 2*E_J*|cos(πΦ/Φ_0)| for two identical junctions
    # However, this changes the nature of the problem - the Hamiltonian itself
    # becomes flux-dependent, not just boundary conditions.

    # For demonstration, we compute using the flux-modulated effective coupling
    for i, phi_ext in enumerate(phi_ext_array):
        # DC SQUID effective coupling (assuming input E_J is per junction)
        # The factor 2 accounts for two junctions in parallel (at zero flux)
        cos_factor = np.abs(np.cos(phi_ext / 2.0))
        EJ_eff_ratio = 2.0 * EJ_over_EC * cos_factor

        # Mathieu parameter for this flux point
        q = EJ_eff_ratio / 8.0

        # Compute characteristic values
        for level in range(n_levels):
            if level == 0:
                char_val = mathieu_a(0, q)
            elif level % 2 == 1:
                k = (level + 1) // 2
                char_val = mathieu_b(k, q)
            else:
                k = level // 2
                char_val = mathieu_a(k, q)

            energies[i, level] = 4.0 * char_val

    # Shift so ground state minimum is at zero
    energies -= energies[:, 0].min()

    return energies


def analyze_squid(results_dir='Results_SQUID'):
    """
    Perform complete SQUID analysis:
    - Circuit diagram
    - Energy spectrum vs external flux for different E_J/E_C ratios
    - Convergence analysis vs truncation
    """
    os.makedirs(results_dir, exist_ok=True)

    print("="*70)
    print("SQUID QUANTUM ANALYSIS - Phase Basis")
    print("="*70)

    # Parameters
    n_levels = 10
    n_max_default = 50

    # External flux range: -0.5 to 0.5 (in units of Φ_0)
    # φ_ext = 2π * f_ext where f_ext = Φ_ext/Φ_0
    f_ext_array = np.linspace(-0.5, 0.5, 200)
    phi_ext_array = 2 * np.pi * f_ext_array

    # ========================================================================
    # 1. Draw circuit diagram
    # ========================================================================
    print("\nGenerating circuit diagram...")
    draw_squid_circuit(results_dir)

    # ========================================================================
    # 2. Energy spectrum vs external flux for different E_J/E_C ratios
    # ========================================================================
    print("\nGenerating energy spectrum plots...")

    # Different regimes:
    # E_J/E_C << 1: Charge regime (large charging energy)
    # E_J/E_C ~ 1: Intermediate regime
    # E_J/E_C >> 1: Transmon-like regime (small charging energy)
    EJ_EC_range = [0.5, 2.0, 5.0, 20.0]

    fig_spectrum = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, EJ_EC in enumerate(EJ_EC_range):
        ax = fig_spectrum.add_subplot(gs[idx])

        print(f"  Computing spectrum for E_J/E_C = {EJ_EC}...")
        energies = compute_spectrum(phi_ext_array, EJ_EC, n_levels, n_max_default)

        # Plot first n_levels energy bands
        for level in range(min(n_levels, energies.shape[1])):
            ax.plot(f_ext_array, energies[:, level],
                   color=colors[level % len(colors)], lw=2, label=f'$E_{level}$')

        ax.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=12)
        ax.set_ylabel(r'$E$ (units of $E_C$)', fontsize=12)
        ax.set_title(f'$E_J/E_C = {EJ_EC}$', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 0.5)

        # Add vertical lines at integer flux quanta
        for n in [-1, 0, 1]:
            if n == 0:
                ax.axvline(n, color='gray', linestyle='--', alpha=0.5, lw=1.5)
            else:
                ax.axvline(n, color='gray', linestyle=':', alpha=0.3, lw=1)

        if idx == 0:
            ax.legend(fontsize=9, ncol=2, loc='best')

        # Add annotation about regime
        if EJ_EC < 1:
            regime = "Charge regime"
        elif EJ_EC < 3:
            regime = "Intermediate"
        else:
            regime = "Phase regime"

        ax.text(0.98, 0.98, regime, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    fig_spectrum.suptitle('SQUID Energy Spectrum vs External Flux', fontsize=15, y=0.98)
    plt.savefig(f'{results_dir}/spectrum_vs_flux.png', dpi=200, bbox_inches='tight')
    print(f"Spectrum plot saved: {results_dir}/spectrum_vs_flux.png")
    plt.close()

    # ========================================================================
    # 3. Individual high-resolution plots for selected E_J/E_C
    # ========================================================================
    print("\nGenerating individual spectrum plots...")

    for EJ_EC in [0.5, 5.0, 20.0]:
        fig_ind, ax_ind = plt.subplots(figsize=(10, 7))

        energies = compute_spectrum(phi_ext_array, EJ_EC, n_levels, n_max_default)

        for level in range(min(n_levels, energies.shape[1])):
            ax_ind.plot(f_ext_array, energies[:, level],
                       color=colors[level % len(colors)], lw=2.5, label=f'$E_{level}$')

        ax_ind.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=14)
        ax_ind.set_ylabel(r'$E$ (units of $E_C$)', fontsize=14)
        ax_ind.set_title(f'SQUID Energy Spectrum: $E_J/E_C = {EJ_EC}$', fontsize=15)
        ax_ind.grid(True, alpha=0.3)
        ax_ind.set_xlim(-0.5, 0.5)
        ax_ind.legend(fontsize=11, ncol=2, loc='best')

        # Flux quantum lines
        for n in range(-1, 2):
            if n == 0:
                ax_ind.axvline(n, color='gray', linestyle='--', alpha=0.5, lw=1.5,
                              label=r'$\Phi_0$ quantum' if n == 0 else '')
            else:
                ax_ind.axvline(n, color='gray', linestyle=':', alpha=0.3, lw=1)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/spectrum_EJ_EC_{EJ_EC}.png', dpi=200, bbox_inches='tight')
        print(f"Individual spectrum saved: {results_dir}/spectrum_EJ_EC_{EJ_EC}.png")
        plt.close()

    # ========================================================================
    # 4. Convergence analysis vs truncation at fixed flux
    # ========================================================================
    print("\nGenerating convergence analysis...")

    # Fix flux at f_ext = 0.25 (quarter flux quantum - maximum sensitivity)
    f_ext_fixed = 0.25
    phi_ext_fixed = 2 * np.pi * f_ext_fixed

    truncation_range = np.arange(5, 101, 5)
    EJ_EC_convergence = [0.5, 2.0, 5.0, 20.0]
    n_levels_conv = 5

    fig_conv = plt.figure(figsize=(14, 10))
    gs_conv = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for idx, EJ_EC in enumerate(EJ_EC_convergence):
        ax = fig_conv.add_subplot(gs_conv[idx])

        print(f"  Convergence analysis for E_J/E_C = {EJ_EC}...")

        energies_vs_trunc = np.zeros((len(truncation_range), n_levels_conv))

        for i, n_max in enumerate(truncation_range):
            H = build_hamiltonian_phase_basis(phi_ext_fixed, EJ_EC, n_max)
            evals = np.linalg.eigvalsh(H)
            energies_vs_trunc[i, :] = evals[:n_levels_conv].real

        # Shift so ground state is at zero
        energies_vs_trunc -= energies_vs_trunc[:, 0].min()

        # Plot each energy level
        for level in range(n_levels_conv):
            ax.plot(truncation_range, energies_vs_trunc[:, level],
                   color=colors[level % len(colors)], lw=2, marker='o',
                   markersize=4, label=f'$E_{level}$')

        # Calculate convergence for ground state
        E0_converged = energies_vs_trunc[-1, 0]
        E0_error = np.abs(energies_vs_trunc[:, 0] - E0_converged)
        converged_idx = np.argmax(E0_error < 1e-6)

        if converged_idx > 0:
            converged_nmax = truncation_range[converged_idx]
            ax.axvline(converged_nmax, color='green', linestyle='--',
                      linewidth=1.5, alpha=0.5,
                      label=f'Converged: n_max={converged_nmax}')

        ax.set_xlabel(r'$n_{max}$ (truncation)', fontsize=12)
        ax.set_ylabel(r'$E$ (units of $E_C$)', fontsize=12)
        ax.set_title(f'$E_J/E_C = {EJ_EC}$, $f_{{ext}} = {f_ext_fixed}$', fontsize=13)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=9, ncol=2, loc='best')

    fig_conv.suptitle(f'Energy Convergence vs Truncation at $\\Phi_{{ext}} = {f_ext_fixed}\\Phi_0$',
                      fontsize=15, y=0.98)
    plt.savefig(f'{results_dir}/convergence_vs_truncation.png', dpi=200, bbox_inches='tight')
    print(f"Convergence plot saved: {results_dir}/convergence_vs_truncation.png")
    plt.close()

    # ========================================================================
    # 5. Convergence error (log scale)
    # ========================================================================
    print("\nGenerating exponential convergence plots...")

    fig_conv_log = plt.figure(figsize=(14, 10))
    gs_conv_log = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for idx, EJ_EC in enumerate(EJ_EC_convergence):
        ax = fig_conv_log.add_subplot(gs_conv_log[idx])

        energies_vs_trunc = np.zeros((len(truncation_range), n_levels_conv))

        for i, n_max in enumerate(truncation_range):
            H = build_hamiltonian_phase_basis(phi_ext_fixed, EJ_EC, n_max)
            evals = np.linalg.eigvalsh(H)
            energies_vs_trunc[i, :] = evals[:n_levels_conv].real

        energies_vs_trunc -= energies_vs_trunc[:, 0].min()

        # Plot absolute error from converged value
        for level in range(n_levels_conv):
            E_converged = energies_vs_trunc[-1, level]
            abs_error = np.abs(energies_vs_trunc[:, level] - E_converged)
            abs_error = np.where(abs_error < 1e-14, 1e-14, abs_error)

            ax.semilogy(truncation_range, abs_error,
                       color=colors[level % len(colors)], lw=2,
                       marker='o', markersize=4, label=f'$E_{level}$')

        ax.axhline(1e-10, color='red', linestyle='--', linewidth=1.5,
                  label='Numerical precision', alpha=0.7)

        ax.set_xlabel(r'$n_{max}$ (truncation)', fontsize=12)
        ax.set_ylabel(r'$|E - E_{converged}|$ (log scale)', fontsize=12)
        ax.set_title(f'$E_J/E_C = {EJ_EC}$', fontsize=13)
        ax.set_ylim(1e-14, 1e0)
        ax.grid(True, alpha=0.3, which='both')

        if idx == 0:
            ax.legend(fontsize=9, ncol=2, loc='best')

    fig_conv_log.suptitle('Exponential Convergence of SQUID Energy Levels',
                          fontsize=15, y=0.98)
    plt.savefig(f'{results_dir}/convergence_exponential.png', dpi=200, bbox_inches='tight')
    print(f"Exponential convergence plot saved: {results_dir}/convergence_exponential.png")
    plt.close()

    # ========================================================================
    # 6. Transition frequencies vs flux
    # ========================================================================
    print("\nGenerating transition frequency plots...")

    EJ_EC_transitions = [2.0, 5.0, 20.0]

    for EJ_EC in EJ_EC_transitions:
        fig_trans, ax_trans = plt.subplots(figsize=(10, 7))

        energies = compute_spectrum(phi_ext_array, EJ_EC, 5, n_max_default)

        # Plot E_01, E_12, E_23
        transitions = [
            (0, 1, r'$E_{01}$'),
            (1, 2, r'$E_{12}$'),
            (2, 3, r'$E_{23}$'),
        ]

        for i, (n1, n2, label) in enumerate(transitions):
            transition_freq = energies[:, n2] - energies[:, n1]
            ax_trans.plot(f_ext_array, transition_freq, lw=2.5,
                         label=label, color=colors[i])

        ax_trans.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=14)
        ax_trans.set_ylabel(r'Transition frequency (units of $E_C$)', fontsize=14)
        ax_trans.set_title(f'SQUID Transition Frequencies: $E_J/E_C = {EJ_EC}$', fontsize=15)
        ax_trans.grid(True, alpha=0.3)
        ax_trans.set_xlim(-0.5, 0.5)
        ax_trans.legend(fontsize=12, loc='best')

        # Flux quantum lines
        for n in range(-1, 2):
            ax_trans.axvline(n, color='gray', linestyle='--' if n == 0 else ':',
                           alpha=0.5, lw=1.5 if n == 0 else 1)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/transitions_EJ_EC_{EJ_EC}.png',
                   dpi=200, bbox_inches='tight')
        print(f"Transition plot saved: {results_dir}/transitions_EJ_EC_{EJ_EC}.png")
        plt.close()

    # ========================================================================
    # 7. Comparison: Numerical vs Mathieu Functions
    # ========================================================================
    if MATHIEU_AVAILABLE:
        print("\nComparing numerical diagonalization vs Mathieu function methods...")

        EJ_EC_comparison = [0.5, 2.0, 5.0, 20.0]

        fig_comp = plt.figure(figsize=(14, 10))
        gs_comp = GridSpec(2, 2, hspace=0.35, wspace=0.3)

        for idx, EJ_EC in enumerate(EJ_EC_comparison):
            ax = fig_comp.add_subplot(gs_comp[idx])

            print(f"  Comparing methods for E_J/E_C = {EJ_EC}...")

            # Compute with both methods
            energies_numerical = compute_spectrum(phi_ext_array, EJ_EC, 5, n_max_default)
            energies_mathieu = compute_spectrum_mathieu(phi_ext_array, EJ_EC, 5)

            if energies_mathieu is not None:
                # Plot difference in E_0 (ground state)
                delta_E0 = energies_numerical[:, 0] - energies_mathieu[:, 0]

                ax.plot(f_ext_array, delta_E0, lw=2, color='blue',
                       label=r'$\Delta E_0$ (Numerical - Mathieu)')

                # Also plot differences for higher levels
                for level in range(1, min(3, energies_numerical.shape[1])):
                    delta_E = energies_numerical[:, level] - energies_mathieu[:, level]
                    ax.plot(f_ext_array, delta_E, lw=1.5, alpha=0.7,
                           label=f'$\\Delta E_{level}$')

                ax.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=12)
                ax.set_ylabel(r'$\Delta E$ (units of $E_C$)', fontsize=12)
                ax.set_title(f'$E_J/E_C = {EJ_EC}$', fontsize=13)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.5, 0.5)
                ax.axhline(0, color='black', linestyle='--', alpha=0.5, lw=1)

                if idx == 0:
                    ax.legend(fontsize=9, loc='best')

                # Add statistics about the difference
                mean_diff = np.mean(np.abs(delta_E0))
                max_diff = np.max(np.abs(delta_E0))
                ax.text(0.02, 0.98,
                       f'Mean |ΔE₀|: {mean_diff:.2e}\nMax |ΔE₀|: {max_diff:.2e}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        fig_comp.suptitle('Comparison: Numerical Diagonalization vs Mathieu Functions',
                         fontsize=15, y=0.98)
        plt.savefig(f'{results_dir}/comparison_numerical_vs_mathieu.png',
                   dpi=200, bbox_inches='tight')
        print(f"Comparison plot saved: {results_dir}/comparison_numerical_vs_mathieu.png")
        plt.close()

        # ====================================================================
        # 7b. Side-by-side spectrum comparison for one E_J/E_C ratio
        # ====================================================================
        print("\nGenerating side-by-side spectrum comparison...")

        EJ_EC_demo = 5.0

        fig_side, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        energies_numerical = compute_spectrum(phi_ext_array, EJ_EC_demo, 5, n_max_default)
        energies_mathieu = compute_spectrum_mathieu(phi_ext_array, EJ_EC_demo, 5)

        # Left: Numerical
        for level in range(energies_numerical.shape[1]):
            ax1.plot(f_ext_array, energies_numerical[:, level],
                    color=colors[level % len(colors)], lw=2.5, label=f'$E_{level}$')
        ax1.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=14)
        ax1.set_ylabel(r'$E$ (units of $E_C$)', fontsize=14)
        ax1.set_title(f'Numerical Diagonalization\n$E_J/E_C = {EJ_EC_demo}$', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, 0.5)
        ax1.legend(fontsize=11, loc='best')

        # Right: Mathieu
        if energies_mathieu is not None:
            for level in range(energies_mathieu.shape[1]):
                ax2.plot(f_ext_array, energies_mathieu[:, level],
                        color=colors[level % len(colors)], lw=2.5, label=f'$E_{level}$')
        ax2.set_xlabel(r'$f_{ext} = \Phi_{ext}/\Phi_0$', fontsize=14)
        ax2.set_ylabel(r'$E$ (units of $E_C$)', fontsize=14)
        ax2.set_title(f'Mathieu Functions\n$E_J/E_C = {EJ_EC_demo}$', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.5, 0.5)
        ax2.legend(fontsize=11, loc='best')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/spectrum_comparison_sidebyside.png',
                   dpi=200, bbox_inches='tight')
        print(f"Side-by-side comparison saved: {results_dir}/spectrum_comparison_sidebyside.png")
        plt.close()

    else:
        print("\nMathieu functions not available, skipping comparison.")

    print("\n" + "="*70)
    print("SQUID ANALYSIS COMPLETE")
    print("="*70)
    print(f"All results saved to: {results_dir}/")
    print("\nGenerated files:")
    print("  - Circuit diagram")
    print("  - Energy spectrum vs flux (combined + individual)")
    print("  - Convergence analysis")
    print("  - Exponential convergence")
    print("  - Transition frequencies")
    if MATHIEU_AVAILABLE:
        print("  - Numerical vs Mathieu comparison")
        print("  - Side-by-side spectrum comparison")
    print("="*70)


if __name__ == '__main__':
    analyze_squid()
