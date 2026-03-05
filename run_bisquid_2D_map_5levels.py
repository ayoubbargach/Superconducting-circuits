"""
Generate 2D maps for first 5 energy levels of BiSQUID full quantum system
With n_max=10, m1_max=5, m2_max=5 and 50x50 grid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
from BiSQUID_full_quantum import compute_spectrum

# Parameters
n_max = 10
m1_max = 5
m2_max = 5
n_levels = 5  # First 5 energy levels

# Grid parameters
grid_size = 50
f1_array = np.linspace(-0.5, 0.5, grid_size)
f2_array = np.linspace(-0.5, 0.5, grid_size)

results_dir = 'Results_BiSQUID_FullQuantum'
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("BiSQUID FULL QUANTUM - 5 ENERGY LEVELS 2D MAPS")
print("="*70)
print(f"\nParameters:")
print(f"  n_max = {n_max} (charge: {2*n_max+1} states)")
print(f"  m1_max = {m1_max} (oscillator 1: {m1_max+1} states)")
print(f"  m2_max = {m2_max} (oscillator 2: {m2_max+1} states)")
print(f"  Total Hilbert space dimension: {(2*n_max+1)*(m1_max+1)*(m2_max+1)}")
print(f"\nGrid: {grid_size} × {grid_size} = {grid_size**2} points")
print(f"Computing {n_levels} energy levels")
print(f"\nEstimated time: ~4 minutes")
print("="*70)

# Storage arrays for all 5 levels
energy_maps = np.zeros((n_levels, grid_size, grid_size))

total_points = grid_size * grid_size
start_time = time.time()

print("\nComputing 2D energy maps...")
for i, f1 in enumerate(f1_array):
    for j, f2 in enumerate(f2_array):
        point_num = i * grid_size + j + 1

        # Compute spectrum
        E, _ = compute_spectrum(f1, f2, n_max, m1_max, m2_max, n_levels)

        # Store all levels
        for level in range(n_levels):
            if level < len(E):
                energy_maps[level, i, j] = E[level]

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

# Shift all levels so ground state minimum is at zero
E_0_min = energy_maps[0].min()
for level in range(n_levels):
    energy_maps[level] -= E_0_min

# ============================================================================
# Plot all 5 energy levels in a 2x3 grid
# ============================================================================
print("\nGenerating energy level plots...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)

for level in range(n_levels):
    ax = fig.add_subplot(gs[level // 3, level % 3])

    im = ax.contourf(f1_array, f2_array, energy_maps[level].T, levels=50, cmap='viridis')
    ax.contour(f1_array, f2_array, energy_maps[level].T, levels=10,
               colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$E$ (units of $E_C$)', fontsize=11)

    ax.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=12)
    ax.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=12)
    ax.set_title(f'Energy level $E_{level}(f_1, f_2)$', fontsize=13)

# Leave the 6th subplot empty or add summary info
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
info_text = f"""
Full Quantum BiSQUID
n_max = {n_max}
m1_max = {m1_max}
m2_max = {m2_max}

Grid: {grid_size}×{grid_size}
Computation time:
{total_time/60:.2f} minutes

Hilbert space:
{(2*n_max+1)*(m1_max+1)*(m2_max+1)} dimensions
"""
ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes, fontsize=12,
         verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Energy Levels $E_n(f_1, f_2)$ - Full Quantum Treatment',
             fontsize=16, y=0.98)
plt.savefig(f'{results_dir}/Energy_levels_0to4_2D_map_{grid_size}x{grid_size}.png',
            dpi=200, bbox_inches='tight')
print(f"Saved: {results_dir}/Energy_levels_0to4_2D_map_{grid_size}x{grid_size}.png")
plt.close()

# ============================================================================
# Plot transition frequencies
# ============================================================================
print("Generating transition frequency plots...")

fig2 = plt.figure(figsize=(18, 6))
gs2 = GridSpec(1, 3, hspace=0.3, wspace=0.3)

transitions = [
    (0, 1, 'E_{01}'),
    (1, 2, 'E_{12}'),
    (2, 3, 'E_{23}'),
]

for idx, (level1, level2, label) in enumerate(transitions):
    ax = fig2.add_subplot(gs2[0, idx])

    transition_map = energy_maps[level2] - energy_maps[level1]

    im = ax.contourf(f1_array, f2_array, transition_map.T, levels=50, cmap='plasma')
    ax.contour(f1_array, f2_array, transition_map.T, levels=10,
               colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta E$ (units of $E_C$)', fontsize=11)

    ax.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=12)
    ax.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=12)
    ax.set_title(f'Transition ${label}(f_1, f_2) = E_{level2} - E_{level1}$', fontsize=13)

plt.suptitle('Transition Frequencies - Full Quantum Treatment', fontsize=16, y=1.02)
plt.savefig(f'{results_dir}/Transition_frequencies_2D_map_{grid_size}x{grid_size}.png',
            dpi=200, bbox_inches='tight')
print(f"Saved: {results_dir}/Transition_frequencies_2D_map_{grid_size}x{grid_size}.png")
plt.close()

# ============================================================================
# Individual high-resolution plots for each level
# ============================================================================
print("Generating individual energy level plots...")

for level in range(n_levels):
    fig_ind, ax_ind = plt.subplots(figsize=(10, 8))

    im = ax_ind.contourf(f1_array, f2_array, energy_maps[level].T, levels=50, cmap='viridis')
    ax_ind.contour(f1_array, f2_array, energy_maps[level].T, levels=15,
                   colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax_ind)
    cbar.set_label(r'$E_{' + str(level) + r'}$ (units of $E_C$)', fontsize=13)

    ax_ind.set_xlabel(r'$f_1 = \Phi_1/\Phi_0$', fontsize=14)
    ax_ind.set_ylabel(r'$f_2 = \Phi_2/\Phi_0$', fontsize=14)
    ax_ind.set_title(f'Energy level $E_{level}(f_1, f_2)$ - Full Quantum\n'
                     f'n_max={n_max}, m1_max={m1_max}, m2_max={m2_max}', fontsize=14)

    # Add energy range annotation
    E_min = energy_maps[level].min()
    E_max = energy_maps[level].max()
    ax_ind.text(0.02, 0.98, f'Range: [{E_min:.3f}, {E_max:.3f}] $E_C$',
                transform=ax_ind.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{results_dir}/E{level}_2D_map_full_quantum_{grid_size}x{grid_size}.png',
                dpi=200, bbox_inches='tight')
    print(f"Saved: {results_dir}/E{level}_2D_map_full_quantum_{grid_size}x{grid_size}.png")
    plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"All results saved to: {results_dir}/")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"\nGenerated files:")
print(f"  - Combined 5-level plot")
print(f"  - Transition frequencies plot")
print(f"  - 5 individual high-resolution plots (E_0 through E_4)")
