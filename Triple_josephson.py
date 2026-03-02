"""
Triple Josephson Junction Circuit Analysis

Analyzes energy levels of a triple Josephson junction circuit with topology:
  Cg + (EJ1 + L1) || EJ0 || (EJ2 + L2)

Two independent external fluxes (Φ₁, Φ₂) thread the two loops.

Hamiltonian in the charge basis {|n>}:
  H = Σ_n 4E_C n² |n⟩⟨n|
      - (E_J^eff/2)[e^(-iφ₀) Σ_n |n⟩⟨n+1| + e^(iφ₀) Σ_n |n+1⟩⟨n|]
      + (g₁/2) Σ_n |n⟩⟨n| - (g₁/4) Σ_n [e^(i4πf₁)|n⟩⟨n+2| + e^(-i4πf₁)|n⟩⟨n-2|]
      + (g₂/2) Σ_n |n⟩⟨n| - (g₂/4) Σ_n [e^(i4πf₂)|n⟩⟨n+2| + e^(-i4πf₂)|n⟩⟨n-2|]

This is a pentadiagonal Hermitian matrix with complex elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from circuit_diagram_triple_josephson import draw_circuit

# Physical constants
PHI_0 = 2.067833848e-15  # Flux quantum in Wb (h/2e)

# ============================================================================
# TYPICAL SUPERCONDUCTING CIRCUIT PARAMETERS
# ============================================================================
# Based on typical transmon/fluxonium parameters from literature

# Charging energy (set as reference scale)
E_C = 1.0  # We work in units where E_C = 1

# Josephson energies (in units of E_C)
E_J0 = 10.0   # Center junction (strongest)
E_J1 = 8.0    # Left junction
E_J2 = 8.0    # Right junction (symmetric with left)

# Inductances (same for both)
# Typical: L ~ 1-10 nH for superconducting circuits
L_1 = 5.0e-9  # 5 nH
L_2 = 5.0e-9  # 5 nH

# Critical currents (typical: 10-100 nA for small junctions)
I_c1 = 50e-9  # 50 nA
I_c2 = 50e-9  # 50 nA

# Inductive energies E_L = Φ₀²/(4π²L)
E_L1 = (PHI_0**2) / (4 * np.pi**2 * L_1)  # in Joules
E_L2 = (PHI_0**2) / (4 * np.pi**2 * L_2)

# Convert to E_C units (assuming E_C corresponds to typical value)
# Typical E_C ~ 0.2-1 GHz × h ~ 1-5 × 10^-25 J
E_C_joules = 2e-25  # Example: ~0.3 GHz
E_L1_normalized = E_L1 / E_C_joules
E_L2_normalized = E_L2 / E_C_joules

# Coupling strengths g = (2πLI_c/Φ₀)² E_L
g_1 = ((2 * np.pi * L_1 * I_c1) / PHI_0)**2 * E_L1_normalized
g_2 = ((2 * np.pi * L_2 * I_c2) / PHI_0)**2 * E_L2_normalized

print(f"Circuit parameters:")
print(f"  E_C = {E_C}")
print(f"  E_J0 = {E_J0} E_C")
print(f"  E_J1 = {E_J1} E_C")
print(f"  E_J2 = {E_J2} E_C")
print(f"  L_1 = L_2 = {L_1*1e9:.1f} nH")
print(f"  I_c1 = I_c2 = {I_c1*1e9:.1f} nA")
print(f"  E_L1 = E_L2 = {E_L1_normalized:.4f} E_C")
print(f"  g_1 = {g_1:.6f}")
print(f"  g_2 = {g_2:.6f}")


def compute_effective_parameters(f1, f2):
    """
    Compute effective Josephson energy and phase offset.

    E_J^eff(f₁, f₂) = √[A² + B²]
    tan(φ₀) = B/A

    where:
      A = E_J1 cos(2πf₁) + E_J0 + E_J2 cos(2πf₂)
      B = E_J1 sin(2πf₁) + E_J2 sin(2πf₂)

    Parameters
    ----------
    f1, f2 : float
        Reduced fluxes (Φ/Φ₀)

    Returns
    -------
    E_J_eff : float
        Effective Josephson energy
    phi_0 : float
        Phase offset
    """
    A = E_J1 * np.cos(2 * np.pi * f1) + E_J0 + E_J2 * np.cos(2 * np.pi * f2)
    B = E_J1 * np.sin(2 * np.pi * f1) + E_J2 * np.sin(2 * np.pi * f2)

    E_J_eff = np.sqrt(A**2 + B**2)
    phi_0 = np.arctan2(B, A)

    return E_J_eff, phi_0


def build_hamiltonian(f1, f2, n_max=30):
    """
    Build the triple Josephson junction Hamiltonian matrix in the charge basis.

    H = Σ_n 4E_C n² |n⟩⟨n|
        - (E_J^eff/2)[e^(-iφ₀) Σ_n |n⟩⟨n+1| + e^(iφ₀) Σ_n |n+1⟩⟨n|]
        + (g₁/2) Σ_n |n⟩⟨n| - (g₁/4) Σ_n [e^(i4πf₁)|n⟩⟨n+2| + e^(-i4πf₁)|n⟩⟨n-2|]
        + (g₂/2) Σ_n |n⟩⟨n| - (g₂/4) Σ_n [e^(i4πf₂)|n⟩⟨n+2| + e^(-i4πf₂)|n⟩⟨n-2|]

    Parameters
    ----------
    f1, f2 : float
        Reduced fluxes Φ₁/Φ₀, Φ₂/Φ₀
    n_max : int
        Truncation: n ranges from -n_max to +n_max.
        Matrix size is (2*n_max + 1) x (2*n_max + 1).

    Returns
    -------
    H : ndarray (complex)
        Hamiltonian matrix in units of E_C.
    """
    dim = 2 * n_max + 1
    n_values = np.arange(-n_max, n_max + 1)

    # Initialize Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)

    # Compute effective parameters
    E_J_eff, phi_0 = compute_effective_parameters(f1, f2)

    # ========================================================================
    # DIAGONAL TERMS: 4E_C n² + (g₁/2) + (g₂/2)
    # ========================================================================
    diagonal = 4.0 * E_C * n_values**2 + (g_1 + g_2) / 2.0
    H += np.diag(diagonal)

    # ========================================================================
    # OFF-DIAGONAL n ↔ n±1: Josephson coupling with phase offset
    # ========================================================================
    # ⟨n|H|n+1⟩ = -(E_J^eff/2) e^(-iφ₀)
    # ⟨n+1|H|n⟩ = -(E_J^eff/2) e^(iφ₀)  [Hermitian conjugate]

    off_diag_1_upper = -(E_J_eff / 2.0) * np.exp(-1j * phi_0) * np.ones(dim - 1)
    off_diag_1_lower = -(E_J_eff / 2.0) * np.exp(1j * phi_0) * np.ones(dim - 1)

    H += np.diag(off_diag_1_upper, 1)  # Upper diagonal
    H += np.diag(off_diag_1_lower, -1)  # Lower diagonal

    # ========================================================================
    # OFF-DIAGONAL n ↔ n±2: Inductor coupling with flux-dependent phases
    # ========================================================================
    # From L₁: ⟨n|H|n+2⟩ = -(g₁/4) e^(i4πf₁)
    # From L₂: ⟨n|H|n+2⟩ = -(g₂/4) e^(i4πf₂)

    off_diag_2_upper = (-(g_1 / 4.0) * np.exp(1j * 4 * np.pi * f1)
                        - (g_2 / 4.0) * np.exp(1j * 4 * np.pi * f2)) * np.ones(dim - 2)
    off_diag_2_lower = (-(g_1 / 4.0) * np.exp(-1j * 4 * np.pi * f1)
                        - (g_2 / 4.0) * np.exp(-1j * 4 * np.pi * f2)) * np.ones(dim - 2)

    H += np.diag(off_diag_2_upper, 2)   # Upper diagonal (+2)
    H += np.diag(off_diag_2_lower, -2)  # Lower diagonal (-2)

    return H


def compute_spectrum_2D(f1_array, f2_array, n_levels=10, n_max=30):
    """
    Compute energy spectrum over a 2D grid of (f1, f2) values.

    Parameters
    ----------
    f1_array : ndarray
        Array of f1 values
    f2_array : ndarray
        Array of f2 values
    n_levels : int
        Number of energy levels to compute
    n_max : int
        Charge basis truncation

    Returns
    -------
    energies : ndarray, shape (len(f1_array), len(f2_array), n_levels)
        Energy eigenvalues in units of E_C
    """
    n_f1 = len(f1_array)
    n_f2 = len(f2_array)
    energies = np.zeros((n_f1, n_f2, n_levels))

    for i, f1 in enumerate(f1_array):
        for j, f2 in enumerate(f2_array):
            H = build_hamiltonian(f1, f2, n_max)
            evals = np.linalg.eigvalsh(H)  # Returns real eigenvalues (H is Hermitian)
            energies[i, j, :] = evals[:n_levels]

    # Shift ground state to zero
    energies -= energies.min()

    return energies


def analyze_triple_josephson(results_dir='Results_Triple_Josephson'):
    """
    Main analysis function for triple Josephson junction circuit.

    Generates:
      1. Circuit diagram
      2. Energy levels vs f1 at fixed f2
      3. Energy levels vs f2 at fixed f1
      4. 2D colormap of ground state energy E_0(f1, f2)
      5. 2D colormap of transition frequency E_01(f1, f2)
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print("TRIPLE JOSEPHSON JUNCTION CIRCUIT ANALYSIS")
    print("="*60)

    # ========================================================================
    # Determine flux interval for φ₀ to sweep 2π
    # ========================================================================
    # φ₀ = arctan(B/A) where:
    #   A = E_J1 cos(2πf₁) + E_J0 + E_J2 cos(2πf₂)
    #   B = E_J1 sin(2πf₁) + E_J2 sin(2πf₂)
    #
    # For φ₀ to sweep full 2π, we need B/A to go through all values.
    # Sweeping f1, f2 from -0.5 to 0.5 (one flux quantum) is natural.

    print("\nFlux range selection:")
    print("  For phi_0 to vary significantly, we choose:")
    print("  f_1 in [-0.5, 0.5]  (one flux quantum)")
    print("  f_2 in [-0.5, 0.5]  (one flux quantum)")

    # Test: compute φ₀ range
    f_test = np.linspace(-0.5, 0.5, 100)
    phi_0_range = []
    for f1 in f_test:
        for f2 in f_test:
            _, phi_0 = compute_effective_parameters(f1, f2)
            phi_0_range.append(phi_0)

    phi_0_range = np.array(phi_0_range)
    print(f"  phi_0 range: [{phi_0_range.min():.3f}, {phi_0_range.max():.3f}] rad")
    print(f"  phi_0 span: {phi_0_range.max() - phi_0_range.min():.3f} rad = {(phi_0_range.max() - phi_0_range.min())/np.pi:.2f}pi")

    # ========================================================================
    # Plot 1: Energy levels vs f1 at fixed f2 = 0
    # ========================================================================
    print("\nGenerating energy levels vs f1 plot...")

    f1_array = np.linspace(-0.5, 0.5, 200)
    f2_fixed = 0.0
    n_levels = 7
    n_max = 30

    energies_vs_f1 = np.zeros((len(f1_array), n_levels))

    for i, f1 in enumerate(f1_array):
        H = build_hamiltonian(f1, f2_fixed, n_max)
        evals = np.linalg.eigvalsh(H)
        energies_vs_f1[i, :] = evals[:n_levels]

    # Shift and normalize
    energies_vs_f1 -= energies_vs_f1.min()

    # Normalize by E_01 at f1 = 0
    idx_zero = np.argmin(np.abs(f1_array))
    E01_ref = energies_vs_f1[idx_zero, 1] - energies_vs_f1[idx_zero, 0]
    if E01_ref > 1e-10:
        energies_vs_f1 /= E01_ref

    # Plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for level in range(n_levels):
        ax1.plot(f1_array, energies_vs_f1[:, level], color=colors[level], lw=2, label=f'$E_{level}$')

    ax1.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
    ax1.set_ylabel(r'$E / E_{01}$', fontsize=14)
    ax1.set_title(f'Energy levels vs $f_1$ at $f_2 = {f2_fixed}$', fontsize=14)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(0, 5)
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/energy_vs_f1.png', dpi=200)
    print(f"  Saved: {results_dir}/energy_vs_f1.png")
    plt.close()

    # ========================================================================
    # Plot 2: Energy levels vs f2 at fixed f1 = 0
    # ========================================================================
    print("Generating energy levels vs f2 plot...")

    f2_array = np.linspace(-0.5, 0.5, 200)
    f1_fixed = 0.0

    energies_vs_f2 = np.zeros((len(f2_array), n_levels))

    for i, f2 in enumerate(f2_array):
        H = build_hamiltonian(f1_fixed, f2, n_max)
        evals = np.linalg.eigvalsh(H)
        energies_vs_f2[i, :] = evals[:n_levels]

    # Shift and normalize
    energies_vs_f2 -= energies_vs_f2.min()

    idx_zero = np.argmin(np.abs(f2_array))
    E01_ref = energies_vs_f2[idx_zero, 1] - energies_vs_f2[idx_zero, 0]
    if E01_ref > 1e-10:
        energies_vs_f2 /= E01_ref

    # Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for level in range(n_levels):
        ax2.plot(f2_array, energies_vs_f2[:, level], color=colors[level], lw=2, label=f'$E_{level}$')

    ax2.set_xlabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
    ax2.set_ylabel(r'$E / E_{01}$', fontsize=14)
    ax2.set_title(f'Energy levels vs $f_2$ at $f_1 = {f1_fixed}$', fontsize=14)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, 5)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/energy_vs_f2.png', dpi=200)
    print(f"  Saved: {results_dir}/energy_vs_f2.png")
    plt.close()

    # ========================================================================
    # Plot 3: 2D colormap - Ground state energy E_0(f1, f2)
    # ========================================================================
    print("Generating 2D ground state energy map...")

    f1_2d = np.linspace(-0.5, 0.5, 100)
    f2_2d = np.linspace(-0.5, 0.5, 100)

    energies_2d = compute_spectrum_2D(f1_2d, f2_2d, n_levels=2, n_max=30)
    E_0 = energies_2d[:, :, 0]

    fig3, ax3 = plt.subplots(figsize=(10, 8))

    im = ax3.contourf(f1_2d, f2_2d, E_0.T, levels=50, cmap='viridis')
    ax3.contour(f1_2d, f2_2d, E_0.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label(r'$E_0$ (units of $E_C$)', fontsize=12)

    ax3.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
    ax3.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
    ax3.set_title('Ground state energy $E_0(f_1, f_2)$', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/E0_2D_map.png', dpi=200)
    print(f"  Saved: {results_dir}/E0_2D_map.png")
    plt.close()

    # ========================================================================
    # Plot 4: 2D colormap - Transition frequency E_01(f1, f2)
    # ========================================================================
    print("Generating 2D transition frequency map...")

    E_01 = energies_2d[:, :, 1] - energies_2d[:, :, 0]

    fig4, ax4 = plt.subplots(figsize=(10, 8))

    im = ax4.contourf(f1_2d, f2_2d, E_01.T, levels=50, cmap='plasma')
    ax4.contour(f1_2d, f2_2d, E_01.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label(r'$E_{01}$ (units of $E_C$)', fontsize=12)

    ax4.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
    ax4.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
    ax4.set_title('Transition frequency $E_{01}(f_1, f_2)$', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/E01_2D_map.png', dpi=200)
    print(f"  Saved: {results_dir}/E01_2D_map.png")
    plt.close()

    # ========================================================================
    # Plot 5: Effective parameters vs flux
    # ========================================================================
    print("Generating effective parameter plots...")

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

    # E_J^eff(f1, f2)
    E_J_eff_2d = np.zeros((len(f1_2d), len(f2_2d)))
    phi_0_2d = np.zeros((len(f1_2d), len(f2_2d)))

    for i, f1 in enumerate(f1_2d):
        for j, f2 in enumerate(f2_2d):
            E_J_eff_2d[i, j], phi_0_2d[i, j] = compute_effective_parameters(f1, f2)

    im1 = ax5a.contourf(f1_2d, f2_2d, E_J_eff_2d.T, levels=50, cmap='coolwarm')
    cbar1 = plt.colorbar(im1, ax=ax5a)
    cbar1.set_label(r'$E_J^{\rm eff}$ (units of $E_C$)', fontsize=11)
    ax5a.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=12)
    ax5a.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=12)
    ax5a.set_title(r'Effective Josephson energy $E_J^{\rm eff}(f_1, f_2)$', fontsize=12)

    # φ₀(f1, f2)
    im2 = ax5b.contourf(f1_2d, f2_2d, phi_0_2d.T, levels=50, cmap='twilight')
    cbar2 = plt.colorbar(im2, ax=ax5b)
    cbar2.set_label(r'$\varphi_0$ (rad)', fontsize=11)
    ax5b.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=12)
    ax5b.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=12)
    ax5b.set_title(r'Phase offset $\varphi_0(f_1, f_2)$', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/effective_parameters_2D.png', dpi=200)
    print(f"  Saved: {results_dir}/effective_parameters_2D.png")
    plt.close()

    # ========================================================================
    # Draw circuit diagram
    # ========================================================================
    print("\nGenerating circuit diagram...")
    draw_circuit(results_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {results_dir}/")


if __name__ == '__main__':
    analyze_triple_josephson()
