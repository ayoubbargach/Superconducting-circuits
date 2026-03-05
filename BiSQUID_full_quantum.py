"""
BiSQUID Circuit Analysis - Full Quantum Treatment with LC Oscillators

Analyzes energy levels including explicit LC oscillator degrees of freedom.

Circuit topology: Cg + ((EJ1 || C1) + L1) || (EJ0 || C0) || ((EJ2 || C2) + L2)

Three quantum degrees of freedom:
  1. Island charge n (on Cg)
  2. LC1 oscillator excitation m1
  3. LC2 oscillator excitation m2

Full Hilbert space: |n, m1, m2⟩
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from scipy.sparse import lil_matrix, csr_matrix, eye as speye, kron as sparse_kron
from scipy.sparse.linalg import eigsh
import time
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Physical constants
PHI_0 = 2.067833848e-15  # Flux quantum in Wb (h/2e)
e = 1.602176634e-19      # Elementary charge in C
h = 6.62607015e-34       # Planck constant in J⋅s
hbar = h / (2 * np.pi)   # Reduced Planck constant

# ============================================================================
# CIRCUIT PARAMETERS
# ============================================================================

# Island charging energy (reference scale)
E_C = 1.0  # We work in units where E_C = 1

# Josephson energies (in units of E_C)
E_J0 = 10.0   # Center junction
E_J1 = 8.0    # Left junction
E_J2 = 8.0    # Right junction

# Inductances
L_1 = 5.0e-9  # 5 nH
L_2 = 5.0e-9  # 5 nH

# Junction capacitances (typical values for superconducting junctions)
# For Al/AlOx/Al junctions: C ~ 50-100 fF typical
C_1 = 50e-15  # 50 fF
C_2 = 50e-15  # 50 fF
C_0 = 70e-15  # 70 fF (slightly larger for center junction)

# Gate capacitance
C_g = 10e-15  # 10 fF

# ============================================================================
# CALCULATE DERIVED PARAMETERS
# ============================================================================

# Inductive energies E_L = Φ₀²/(4π²L)
E_L1 = (PHI_0**2) / (4 * np.pi**2 * L_1)  # in Joules
E_L2 = (PHI_0**2) / (4 * np.pi**2 * L_2)

# Capacitive energies E_C = e²/(2C)
E_C1 = e**2 / (2 * C_1)  # in Joules
E_C2 = e**2 / (2 * C_2)
E_C0 = e**2 / (2 * C_0)

# Assume island E_C corresponds to typical value for normalization
E_C_joules = 2e-25  # ~0.3 GHz

# Normalize energies to island E_C units
E_L1_norm = E_L1 / E_C_joules
E_L2_norm = E_L2 / E_C_joules
E_C1_norm = E_C1 / E_C_joules
E_C2_norm = E_C2 / E_C_joules
E_C0_norm = E_C0 / E_C_joules

# LC oscillator frequencies ω = 1/√(LC)
omega_1 = np.sqrt(E_L1_norm * E_C1_norm)  # in units of E_C
omega_2 = np.sqrt(E_L2_norm * E_C2_norm)

# Zero-point phase fluctuations φ_ZPF = (E_C/E_L)^(1/4)
phi_ZPF1 = (E_C1_norm / E_L1_norm)**(0.25)
phi_ZPF2 = (E_C2_norm / E_L2_norm)**(0.25)

print("="*70)
print("BiSQUID FULL QUANTUM ANALYSIS")
print("="*70)
print("\nCircuit parameters:")
print(f"  E_C (island) = {E_C} (reference)")
print(f"  E_J0 = {E_J0} E_C")
print(f"  E_J1 = {E_J1} E_C")
print(f"  E_J2 = {E_J2} E_C")
print(f"\nCapacitances:")
print(f"  C_1 = {C_1*1e15:.1f} fF")
print(f"  C_2 = {C_2*1e15:.1f} fF")
print(f"  C_0 = {C_0*1e15:.1f} fF")
print(f"  C_g = {C_g*1e15:.1f} fF")
print(f"\nInductances:")
print(f"  L_1 = {L_1*1e9:.1f} nH")
print(f"  L_2 = {L_2*1e9:.1f} nH")
print(f"\nLC Oscillator parameters:")
print(f"  E_L1 = {E_L1_norm:.2f} E_C")
print(f"  E_L2 = {E_L2_norm:.2f} E_C")
print(f"  E_C1 = {E_C1_norm:.2f} E_C")
print(f"  E_C2 = {E_C2_norm:.2f} E_C")
print(f"  hbar*omega_1 = {omega_1:.3f} E_C")
print(f"  hbar*omega_2 = {omega_2:.3f} E_C")
print(f"  phi_ZPF1 = {phi_ZPF1:.4f}")
print(f"  phi_ZPF2 = {phi_ZPF2:.4f}")
print("="*70)

# ============================================================================
# OPERATOR CONSTRUCTION
# ============================================================================

def annihilation_operator(m_max):
    """Create annihilation operator matrix in Fock basis."""
    dim = m_max + 1
    a = np.zeros((dim, dim))
    for m in range(1, dim):
        a[m, m-1] = np.sqrt(m)
    return a

def creation_operator(m_max):
    """Create creation operator matrix in Fock basis."""
    return annihilation_operator(m_max).T

def number_operator(m_max):
    """Create number operator matrix in Fock basis."""
    return np.diag(np.arange(m_max + 1))

# ============================================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================================

def state_index(n, m1, m2, n_max, m1_max, m2_max):
    """Convert quantum numbers (n, m1, m2) to linear index."""
    n_idx = n + n_max  # Shift n to [0, 2*n_max]
    return (n_idx * (m1_max + 1) + m1) * (m2_max + 1) + m2

def build_hamiltonian_sparse(f1, f2, n_max=10, m1_max=5, m2_max=5):
    """
    Build full quantum Hamiltonian including LC oscillators.

    Uses sparse matrices for efficiency.

    Parameters
    ----------
    f1, f2 : float
        Reduced external fluxes (Φ/Φ₀)
    n_max : int
        Max island charge (charge ranges from -n_max to +n_max)
    m1_max, m2_max : int
        Max oscillator excitation numbers

    Returns
    -------
    H : sparse matrix
        Full Hamiltonian in units of E_C
    total_dim : int
        Total Hilbert space dimension
    """

    dim_n = 2*n_max + 1
    dim_m1 = m1_max + 1
    dim_m2 = m2_max + 1
    total_dim = dim_n * dim_m1 * dim_m2

    print(f"\nBuilding Hamiltonian: f1={f1:.3f}, f2={f2:.3f}")
    print(f"  Hilbert space dimension: {total_dim}")
    print(f"    Island: {dim_n} states (n ∈ [{-n_max}, {n_max}])")
    print(f"    Oscillator 1: {dim_m1} states (m1 ∈ [0, {m1_max}])")
    print(f"    Oscillator 2: {dim_m2} states (m2 ∈ [0, {m2_max}])")

    # Initialize sparse Hamiltonian
    H = lil_matrix((total_dim, total_dim), dtype=complex)

    # Build single-space operators
    a1 = annihilation_operator(m1_max)
    a1_dag = creation_operator(m1_max)
    n1_op = number_operator(m1_max)

    a2 = annihilation_operator(m2_max)
    a2_dag = creation_operator(m2_max)
    n2_op = number_operator(m2_max)

    # ========================================================================
    # PART 1: Diagonal terms (island charging + oscillator energies)
    # ========================================================================

    print("  Adding diagonal terms...")
    for n in range(-n_max, n_max + 1):
        for m1 in range(dim_m1):
            for m2 in range(dim_m2):
                idx = state_index(n, m1, m2, n_max, m1_max, m2_max)

                # Island charging energy: 4*E_C*n²
                E_charge = 4.0 * E_C * n**2

                # Oscillator energies: ℏω(m + 1/2)
                E_osc1 = omega_1 * (m1 + 0.5)
                E_osc2 = omega_2 * (m2 + 0.5)

                H[idx, idx] = E_charge + E_osc1 + E_osc2

    # ========================================================================
    # PART 2: Josephson terms in charge basis
    # ========================================================================
    # In charge basis, cos(φ) creates charge tunneling n ↔ n±1
    # We need: exp(±iφ)|n⟩ = |n±1⟩

    print("  Adding Josephson coupling terms...")

    # Precompute phase factors
    phase1 = 2 * np.pi * f1
    phase2 = 2 * np.pi * f2

    # Branch 1: -E_J1 * cos(φ + 2πf1 - φ_L1)
    # Branch 2: -E_J0 * cos(φ)
    # Branch 3: -E_J2 * cos(φ + 2πf2 - φ_L2)

    # Expand: cos(φ + phase - φ_L) ≈ cos(φ + phase) - sin(φ + phase)*φ_L - (1/2)*cos(φ + phase)*φ_L²

    # φ_Li = φ_ZPF * (a + a†)
    phi_L1_matrix = phi_ZPF1 * (a1 + a1_dag)
    phi_L2_matrix = phi_ZPF2 * (a2 + a2_dag)
    phi_L1_sq = phi_L1_matrix @ phi_L1_matrix
    phi_L2_sq = phi_L2_matrix @ phi_L2_matrix

    for n in range(-n_max, n_max):  # n to n+1 transitions
        for m1 in range(dim_m1):
            for m2 in range(dim_m2):
                idx_n = state_index(n, m1, m2, n_max, m1_max, m2_max)
                idx_np1 = state_index(n+1, m1, m2, n_max, m1_max, m2_max)

                # Branch 2: -E_J0/2 * (exp(iφ) + exp(-iφ))
                # exp(iφ)|n⟩ = |n+1⟩
                H[idx_np1, idx_n] += -E_J0 / 2.0
                H[idx_n, idx_np1] += -E_J0 / 2.0

    # Branch 1 and 3: Include oscillator coupling
    # This is more complex - we need matrix elements ⟨m1'|φ_L1|m1⟩

    for n in range(-n_max, n_max):
        # Branch 1: cos(φ + 2πf1 - φ_L1)
        # Leading order: cos(φ + 2πf1) - sin(φ + 2πf1)*φ_L1

        # cos term (no oscillator change)
        for m1 in range(dim_m1):
            for m2 in range(dim_m2):
                idx_n = state_index(n, m1, m2, n_max, m1_max, m2_max)
                idx_np1 = state_index(n+1, m1, m2, n_max, m1_max, m2_max)

                # exp(i(φ + 2πf1)) ≈ exp(i*2πf1) * |n+1⟩⟨n|
                H[idx_np1, idx_n] += -E_J1 / 2.0 * np.exp(1j * phase1)
                H[idx_n, idx_np1] += -E_J1 / 2.0 * np.exp(-1j * phase1)

        # φ_L1 term (oscillator m1 ↔ m1±1, charge n ↔ n+1)
        for m1 in range(dim_m1):
            for m2 in range(dim_m2):
                idx_n = state_index(n, m1, m2, n_max, m1_max, m2_max)
                idx_np1 = state_index(n+1, m1, m2, n_max, m1_max, m2_max)

                # -sin(φ + 2πf1) * φ_L1 contribution
                # ⟨m1|a + a†|m1⟩ = 0 (diagonal vanishes)
                # ⟨m1±1|a + a†|m1⟩ = √m1 or √(m1+1)

                if m1 < m1_max:  # m1 → m1+1
                    idx_np1_m1p = state_index(n+1, m1+1, m2, n_max, m1_max, m2_max)
                    coupling = -E_J1 * (-1j/2.0) * np.exp(1j * phase1) * phi_ZPF1 * np.sqrt(m1 + 1)
                    H[idx_np1_m1p, idx_n] += coupling
                    H[idx_n, idx_np1_m1p] += np.conj(coupling)

                if m1 > 0:  # m1 → m1-1
                    idx_np1_m1m = state_index(n+1, m1-1, m2, n_max, m1_max, m2_max)
                    coupling = -E_J1 * (-1j/2.0) * np.exp(1j * phase1) * phi_ZPF1 * np.sqrt(m1)
                    H[idx_np1_m1m, idx_n] += coupling
                    H[idx_n, idx_np1_m1m] += np.conj(coupling)

        # Branch 3: cos(φ + 2πf2 - φ_L2) - similar structure
        for m1 in range(dim_m1):
            for m2 in range(dim_m2):
                idx_n = state_index(n, m1, m2, n_max, m1_max, m2_max)
                idx_np1 = state_index(n+1, m1, m2, n_max, m1_max, m2_max)

                # cos term
                H[idx_np1, idx_n] += -E_J2 / 2.0 * np.exp(1j * phase2)
                H[idx_n, idx_np1] += -E_J2 / 2.0 * np.exp(-1j * phase2)

                # φ_L2 term
                if m2 < m2_max:
                    idx_np1_m2p = state_index(n+1, m1, m2+1, n_max, m1_max, m2_max)
                    coupling = -E_J2 * (-1j/2.0) * np.exp(1j * phase2) * phi_ZPF2 * np.sqrt(m2 + 1)
                    H[idx_np1_m2p, idx_n] += coupling
                    H[idx_n, idx_np1_m2p] += np.conj(coupling)

                if m2 > 0:
                    idx_np1_m2m = state_index(n+1, m1, m2-1, n_max, m1_max, m2_max)
                    coupling = -E_J2 * (-1j/2.0) * np.exp(1j * phase2) * phi_ZPF2 * np.sqrt(m2)
                    H[idx_np1_m2m, idx_n] += coupling
                    H[idx_n, idx_np1_m2m] += np.conj(coupling)

    # Convert to CSR format for efficient operations
    H = H.tocsr()

    # Check Hermiticity
    H_diff = H - H.conj().T
    if H_diff.data.size > 0:
        hermiticity_error = np.max(np.abs(H_diff.data))
        print(f"  Hermiticity error: {hermiticity_error:.2e}")
    else:
        print(f"  Hermiticity check: Matrix is Hermitian")

    return H, total_dim

# ============================================================================
# SPECTRUM CALCULATION
# ============================================================================

def compute_spectrum(f1, f2, n_max=10, m1_max=5, m2_max=5, n_levels=10):
    """
    Compute energy spectrum for given flux values.

    Returns
    -------
    energies : ndarray
        Lowest n_levels eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors
    """

    H, total_dim = build_hamiltonian_sparse(f1, f2, n_max, m1_max, m2_max)

    # Diagonalize (get lowest eigenvalues only)
    n_compute = min(n_levels, total_dim - 2)
    print(f"  Diagonalizing for {n_compute} lowest eigenvalues...")

    start_time = time.time()
    eigenvalues, eigenvectors = eigsh(H, k=n_compute, which='SA')
    elapsed = time.time() - start_time

    print(f"  Diagonalization completed in {elapsed:.2f} seconds")

    # Sort by energy
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def compute_spectrum_2D(f1_array, f2_array, n_max=10, m1_max=5, m2_max=5, n_levels=5):
    """
    Compute energy spectrum over 2D grid of flux values.

    Returns
    -------
    energies : ndarray, shape (len(f1_array), len(f2_array), n_levels)
    """
    n_f1 = len(f1_array)
    n_f2 = len(f2_array)
    energies = np.zeros((n_f1, n_f2, n_levels))

    total_points = n_f1 * n_f2

    print(f"\nComputing spectrum over {n_f1} × {n_f2} = {total_points} flux points...")

    for i, f1 in enumerate(f1_array):
        for j, f2 in enumerate(f2_array):
            point_num = i * n_f2 + j + 1
            print(f"\n--- Point {point_num}/{total_points} ---")

            E, _ = compute_spectrum(f1, f2, n_max, m1_max, m2_max, n_levels)
            energies[i, j, :] = E[:n_levels]

    # Shift ground state to zero
    energies -= energies.min()

    return energies

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_bisquid_full_quantum(results_dir='Results_BiSQUID_FullQuantum'):
    """
    Full quantum analysis of BiSQUID circuit with LC oscillators.
    """

    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STARTING FULL QUANTUM ANALYSIS")
    print("="*70)

    # Parameters for computation
    n_max = 8        # ±8 charge states = 17 total
    m1_max = 4       # 5 oscillator states
    m2_max = 4       # 5 oscillator states
    # Total dimension: 17 × 5 × 5 = 425

    n_levels = 7     # Number of energy levels to track

    # ========================================================================
    # Plot 1: Energy levels vs f1 at fixed f2 = 0
    # ========================================================================
    print("\n" + "="*70)
    print("Computing energy vs f1...")
    print("="*70)

    f1_array = np.linspace(-0.5, 0.5, 21)  # Reduced resolution for speed
    f2_fixed = 0.0

    energies_vs_f1 = np.zeros((len(f1_array), n_levels))

    for i, f1 in enumerate(f1_array):
        print(f"\n--- f1 point {i+1}/{len(f1_array)}: f1={f1:.3f} ---")
        E, _ = compute_spectrum(f1, f2_fixed, n_max, m1_max, m2_max, n_levels)
        energies_vs_f1[i, :] = E

    # Normalize
    energies_vs_f1 -= energies_vs_f1.min()
    idx_zero = np.argmin(np.abs(f1_array))
    E01_ref = energies_vs_f1[idx_zero, 1] - energies_vs_f1[idx_zero, 0]
    if E01_ref > 1e-10:
        energies_vs_f1 /= E01_ref

    # Plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for level in range(n_levels):
        ax1.plot(f1_array, energies_vs_f1[:, level], color=colors[level], lw=2,
                label=f'$E_{level}$', marker='o', markersize=4)

    ax1.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
    ax1.set_ylabel(r'$E / E_{01}$', fontsize=14)
    ax1.set_title(f'Energy levels vs $f_1$ at $f_2 = {f2_fixed}$ (Full Quantum)', fontsize=14)
    ax1.set_xlim(-0.5, 0.5)
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/energy_vs_f1_full_quantum.png', dpi=200)
    print(f"\nSaved: {results_dir}/energy_vs_f1_full_quantum.png")
    plt.close()

    # ========================================================================
    # Plot 2: Energy levels vs f2 at fixed f1 = 0
    # ========================================================================
    print("\n" + "="*70)
    print("Computing energy vs f2...")
    print("="*70)

    f2_array = np.linspace(-0.5, 0.5, 21)
    f1_fixed = 0.0

    energies_vs_f2 = np.zeros((len(f2_array), n_levels))

    for i, f2 in enumerate(f2_array):
        print(f"\n--- f2 point {i+1}/{len(f2_array)}: f2={f2:.3f} ---")
        E, _ = compute_spectrum(f1_fixed, f2, n_max, m1_max, m2_max, n_levels)
        energies_vs_f2[i, :] = E

    # Normalize
    energies_vs_f2 -= energies_vs_f2.min()
    idx_zero = np.argmin(np.abs(f2_array))
    E01_ref = energies_vs_f2[idx_zero, 1] - energies_vs_f2[idx_zero, 0]
    if E01_ref > 1e-10:
        energies_vs_f2 /= E01_ref

    # Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for level in range(n_levels):
        ax2.plot(f2_array, energies_vs_f2[:, level], color=colors[level], lw=2,
                label=f'$E_{level}$', marker='o', markersize=4)

    ax2.set_xlabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
    ax2.set_ylabel(r'$E / E_{01}$', fontsize=14)
    ax2.set_title(f'Energy levels vs $f_2$ at $f_1 = {f1_fixed}$ (Full Quantum)', fontsize=14)
    ax2.set_xlim(-0.5, 0.5)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/energy_vs_f2_full_quantum.png', dpi=200)
    print(f"\nSaved: {results_dir}/energy_vs_f2_full_quantum.png")
    plt.close()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {results_dir}/")
    print("\nNote: 2D maps skipped due to computational cost.")
    print("For full 2D analysis, reduce n_max, m1_max, m2_max or increase patience!")

if __name__ == '__main__':
    analyze_bisquid_full_quantum()
