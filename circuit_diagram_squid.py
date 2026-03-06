"""
SQUID Circuit Diagram Generator

Draws SQUID (Superconducting Quantum Interference Device) circuit diagram
showing two Josephson junctions in a loop with capacitors and external flux.

Uses schemdraw if available, otherwise falls back to matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    print("Warning: schemdraw not available, will use matplotlib fallback")


def draw_squid_circuit(results_dir='Results_SQUID'):
    """
    Draw SQUID circuit diagram showing two Josephson junctions in a loop
    with capacitors and external flux.
    Uses schemdraw if available, otherwise falls back to matplotlib.

    Parameters
    ----------
    results_dir : str
        Directory to save the circuit diagram (default: 'Results_SQUID')
    """
    os.makedirs(results_dir, exist_ok=True)

    if SCHEMDRAW_AVAILABLE:
        try:
            with schemdraw.Drawing(show=False) as d:
                d.config(fontsize=12)

                # Top node
                d += elm.Line().right().length(0)
                d += (top := elm.Dot())

                # Left branch: J1 (with C1 in parallel)
                d += elm.Line().down().length(0.3)
                d += (j1_top := elm.Dot())
                d += (j1 := elm.Josephson().down().label('$E_{J1}$', loc='left'))
                d += (j1_bottom := elm.Dot())

                # Capacitor C in parallel with J1
                d.move_from(j1_top.start)
                d += elm.Line().left().length(0.8)
                d += elm.Capacitor().down().label('$C$', loc='left')
                d += elm.Line().to(j1_bottom.start)

                # Continue down to bottom on left
                d.move_from(j1_bottom.start)
                d += elm.Line().down().length(0.3)
                d += (bottom_left := elm.Dot())

                # Right branch: J2 (with Cg in parallel)
                d.move_from(top.start)
                d += elm.Line().right().length(3)
                d += elm.Line().down().length(0.3)
                d += (j2_top := elm.Dot())
                d += (j2 := elm.Josephson().down().label('$E_{J2}$', loc='left'))
                d += (j2_bottom := elm.Dot())

                # Capacitor Cg in parallel with J2
                d.move_from(j2_top.start)
                d += elm.Line().left().length(0.8)
                d += elm.Capacitor().down().label('$C_g$', loc='left')
                d += elm.Line().to(j2_bottom.start)

                # Continue down to bottom on right
                d.move_from(j2_bottom.start)
                d += elm.Line().down().length(0.3)
                d += (bottom_right := elm.Dot())

                # Connect bottom nodes
                d.move_from(bottom_left.start)
                d += elm.Line().to(bottom_right.start)
                d += (bottom := elm.Dot())

                # Ground
                d += elm.Line().down().length(0.3)
                d += elm.Ground()

                # External flux indicator in the loop
                loop_center_x = (j1.center[0] + j2.center[0]) / 2
                loop_center_y = top.start[1] - 1.2
                d += elm.Arc2(arrow='->').at((loop_center_x - 0.6, loop_center_y)).to((loop_center_x + 0.6, loop_center_y))
                d += elm.Label().at((loop_center_x, loop_center_y + 0.4)).label(r'$\Phi_{ext}$', fontsize=11, color='green')

                filepath = f'{results_dir}/circuit_diagram.png'
                d.save(filepath, dpi=300)

            print(f"Circuit diagram saved: {filepath}")
            return

        except Exception as e:
            print(f"schemdraw failed ({e}), falling back to matplotlib")

    # Matplotlib fallback (original code)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'SQUID Circuit', fontsize=16, ha='center', weight='bold')

    # Left branch - J1
    x_j1, y_j1 = 3, 5
    rect1 = patches.Rectangle((x_j1-0.2, y_j1-0.3), 0.4, 0.6,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect1)
    ax.plot([x_j1-0.2, x_j1+0.2], [y_j1, y_j1], 'r-', lw=2)
    ax.text(x_j1-0.7, y_j1, r'$J_1$', fontsize=12, ha='center', color='red')
    ax.plot([x_j1, x_j1], [6.5, y_j1+0.3], 'k-', lw=2)

    # Capacitor C
    x_c, y_c = 3, 3.5
    ax.plot([x_c-0.3, x_c+0.3], [y_c+0.2, y_c+0.2], 'b-', lw=2)
    ax.plot([x_c-0.3, x_c+0.3], [y_c-0.2, y_c-0.2], 'b-', lw=2)
    ax.text(x_c-0.7, y_c, r'$C$', fontsize=12, ha='center', color='blue')
    ax.plot([x_j1, x_j1], [y_j1-0.3, y_c+0.2], 'k-', lw=2)
    ax.plot([x_c, x_c], [y_c-0.2, 2], 'k-', lw=2)

    # Right branch - J2
    x_j2, y_j2 = 7, 5
    rect2 = patches.Rectangle((x_j2-0.2, y_j2-0.3), 0.4, 0.6,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect2)
    ax.plot([x_j2-0.2, x_j2+0.2], [y_j2, y_j2], 'r-', lw=2)
    ax.text(x_j2+0.7, y_j2, r'$J_2$', fontsize=12, ha='center', color='red')
    ax.plot([x_j2, x_j2], [6.5, y_j2+0.3], 'k-', lw=2)

    # Capacitor C_g
    x_cg, y_cg = 7, 3.5
    ax.plot([x_cg-0.3, x_cg+0.3], [y_cg+0.2, y_cg+0.2], 'b-', lw=2)
    ax.plot([x_cg-0.3, x_cg+0.3], [y_cg-0.2, y_cg-0.2], 'b-', lw=2)
    ax.text(x_cg+0.7, y_cg, r'$C_g$', fontsize=12, ha='center', color='blue')
    ax.plot([x_j2, x_j2], [y_j2-0.3, y_cg+0.2], 'k-', lw=2)
    ax.plot([x_cg, x_cg], [y_cg-0.2, 2], 'k-', lw=2)

    ax.plot([3, 7], [2, 2], 'k-', lw=2)
    ax.plot([3, 7], [6.5, 6.5], 'k-', lw=2)

    # External flux
    theta = np.linspace(0, 1.5*np.pi, 50)
    r = 0.6
    x_center, y_center = 5, 4
    ax.plot(x_center + r*np.cos(theta), y_center + r*np.sin(theta), 'g-', lw=2, alpha=0.7)
    ax.arrow(x_center + r*np.cos(theta[-5]), y_center + r*np.sin(theta[-5]),
             0.1, -0.1, head_width=0.15, head_length=0.1, fc='green', ec='green', alpha=0.7)
    ax.text(x_center, y_center-1, r'$\Phi_{ext}$', fontsize=14, ha='center', color='green', weight='bold')

    # Nodes
    for x in [3, 7]:
        for y in [6.5, 2]:
            ax.plot([x], [y], 'ko', markersize=8)

    # Equations
    eq_text = r'$\mathcal{H} = 4E_C \hat{n}^2 - E_{eff}\cos(\varphi_{ext} - \hat{\varphi})$'
    ax.text(5, 0.8, eq_text, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{results_dir}/circuit_diagram.png', dpi=200, bbox_inches='tight')
    print(f"Circuit diagram saved: {results_dir}/circuit_diagram.png")
    plt.close()


if __name__ == '__main__':
    """Generate SQUID circuit diagram when run as standalone script"""
    print("="*70)
    print("SQUID Circuit Diagram Generator")
    print("="*70)
    draw_squid_circuit()
    print("\nDone!")
