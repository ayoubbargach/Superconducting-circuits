"""
Generate 2D E_0 map for BiSQUID full quantum system
With n_max=10, m1_max=5, m2_max=5 and 50x50 grid
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from BiSQUID_full_quantum import compute_spectrum

# Parameters
n_max = 10
m1_max = 5
m2_max = 5
n_levels = 2  # Only need ground state and first excited for E_0 and E_01

# Grid parameters
grid_size = 50
f1_array = np.linspace(-0.5, 0.5, grid_size)
f2_array = np.linspace(-0.5, 0.5, grid_size)

results_dir = 'Results_BiSQUID_FullQuantum'
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("BiSQUID FULL QUANTUM 2D MAP GENERATION")
print("="*70)
print(f"\nParameters:")
print(f"  n_max = {n_max} (charge: {2*n_max+1} states)")
print(f"  m1_max = {m1_max} (oscillator 1: {m1_max+1} states)")
print(f"  m2_max = {m2_max} (oscillator 2: {m2_max+1} states)")
print(f"  Total Hilbert space dimension: {(2*n_max+1)*(m1_max+1)*(m2_max+1)}")
print(f"\nGrid: {grid_size} × {grid_size} = {grid_size**2} points")
print(f"\nEstimated time: ~4 minutes")
print("="*70)

# Storage arrays
E_0_map = np.zeros((grid_size, grid_size))
E_01_map = np.zeros((grid_size, grid_size))

total_points = grid_size * grid_size
start_time = time.time()

print("\nComputing 2D energy map...")
for i, f1 in enumerate(f1_array):
    for j, f2 in enumerate(f2_array):
        point_num = i * grid_size + j + 1

        # Compute spectrum
        E, _ = compute_spectrum(f1, f2, n_max, m1_max, m2_max, n_levels)

        E_0_map[i, j] = E[0]
        if len(E) > 1:
            E_01_map[i, j] = E[1] - E[0]

        # Progress report every 10%
        if point_num % (total_points // 10) == 0:
            elapsed = time.time() - start_time
            progress = point_num / total_points
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            print(f"  Progress: {point_num}/{total_points} ({progress*100:.0f}%) | "
                  f"Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")

total_time = time.time() - start_time
print(f"\nTotal computation time: {total_time/60:.2f} minutes")

# Shift ground state to zero
E_0_map -= E_0_map.min()

# ============================================================================
# Plot E_0(f1, f2)
# ============================================================================
print("\nGenerating E_0 plot...")

fig1, ax1 = plt.subplots(figsize=(10, 8))

im1 = ax1.contourf(f1_array, f2_array, E_0_map.T, levels=50, cmap='viridis')
ax1.contour(f1_array, f2_array, E_0_map.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label(r'$E_0$ (units of $E_C$)', fontsize=12)

ax1.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
ax1.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
ax1.set_title(f'Ground state energy $E_0(f_1, f_2)$ - Full Quantum\n'
              f'n_max={n_max}, m1_max={m1_max}, m2_max={m2_max}', fontsize=14)

plt.tight_layout()
plt.savefig(f'{results_dir}/E0_2D_map_full_quantum_{grid_size}x{grid_size}.png', dpi=200)
print(f"Saved: {results_dir}/E0_2D_map_full_quantum_{grid_size}x{grid_size}.png")
plt.close()

# ============================================================================
# Plot E_01(f1, f2)
# ============================================================================
print("Generating E_01 plot...")

fig2, ax2 = plt.subplots(figsize=(10, 8))

im2 = ax2.contourf(f1_array, f2_array, E_01_map.T, levels=50, cmap='plasma')
ax2.contour(f1_array, f2_array, E_01_map.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label(r'$E_{01}$ (units of $E_C$)', fontsize=12)

ax2.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
ax2.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
ax2.set_title(f'Transition frequency $E_{{01}}(f_1, f_2)$ - Full Quantum\n'
              f'n_max={n_max}, m1_max={m1_max}, m2_max={m2_max}', fontsize=14)

plt.tight_layout()
plt.savefig(f'{results_dir}/E01_2D_map_full_quantum_{grid_size}x{grid_size}.png', dpi=200)
print(f"Saved: {results_dir}/E01_2D_map_full_quantum_{grid_size}x{grid_size}.png")
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"All results saved to: {results_dir}/")
print(f"Total time: {total_time/60:.2f} minutes")
