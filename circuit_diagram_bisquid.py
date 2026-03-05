"""
Triple Josephson Junction Circuit Diagram

Draws the circuit diagram for: Cg + (EJ1 + L1) || EJ0 || (EJ2 + L2)
"""

import os
try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    print("Warning: schemdraw not available, circuit diagram will be skipped")


def draw_circuit(results_dir='Results_BiSQUID'):
    """
    Draw the triple Josephson junction circuit diagram.

    Circuit: Cg + ((EJ1 || C1) + L1) || (EJ0 || C0) || ((EJ2 || C2) + L2)

    Parameters
    ----------
    results_dir : str
        Directory where the circuit diagram will be saved

    Returns
    -------
    str
        Path to the saved circuit diagram, or None if drawing failed
    """
    if not SCHEMDRAW_AVAILABLE:
        print("schemdraw not available, circuit diagram skipped")
        return None

    # Create results directory if needed
    os.makedirs(results_dir, exist_ok=True)

    try:
        with schemdraw.Drawing(show=False) as d:
            d.config(fontsize=12)

            # Gate capacitor Cg
            d += elm.Line().up().length(1)
            d += elm.Capacitor().up().label('$C_g$')
            d += elm.Line().right()

            # Save the top node position (start of parallel branches)
            d += (top := elm.Dot())

            # Branch 1: (EJ1 || C1) + L1 (series)
            # First, create a mini-parallel of EJ1 || C1
            d += elm.Line().down().length(0.3)
            d += (branch1_top := elm.Dot())

            # EJ1
            d += (ej1 := elm.Josephson().down().label('$E_{J1}$', loc='left'))
            d += (branch1_bottom := elm.Dot())

            # C1 in parallel with EJ1
            d.move_from(branch1_top.start)
            d += elm.Line().left().length(0.8)
            d += elm.Capacitor().down().label('$C_1$', loc='left')
            d += elm.Line().to(branch1_bottom.start)

            # Continue with L1
            d.move_from(branch1_bottom.start)
            d += (l1 := elm.Inductor().right().label('$L_1$'))
            d += (bottom_left := elm.Dot())

            # Branch 2: EJ0 || C0 (center branch)
            d.move_from(top.start)
            d += elm.Line().right().length(3)
            d += elm.Line().down().length(0.3)
            d += (branch2_top := elm.Dot())

            # EJ0
            d += (ej0 := elm.Josephson().down().label('$E_{J0}$', loc='left'))
            d += (branch2_bottom := elm.Dot())

            # C0 in parallel with EJ0
            d.move_from(branch2_top.start)
            d += elm.Line().left().length(0.8)
            d += elm.Capacitor().down().label('$C_0$', loc='left')
            d += elm.Line().to(branch2_bottom.start)

            # Use branch2_bottom as the main bottom junction point
            d.move_from(branch2_bottom.start)
            d += (bottom_center := elm.Dot())

            # Branch 3: (EJ2 || C2) + L2 (series)
            d.move_from(top.start)
            d += elm.Line().right().length(6)
            d += elm.Line().down().length(0.3)
            d += (branch3_top := elm.Dot())

            # EJ2
            d += (ej2 := elm.Josephson().down().label('$E_{J2}$', loc='left'))
            d += (branch3_bottom := elm.Dot())

            # C2 in parallel with EJ2
            d.move_from(branch3_top.start)
            d += elm.Line().left().length(0.8)
            d += elm.Capacitor().down().label('$C_2$', loc='left')
            d += elm.Line().to(branch3_bottom.start)

            # Continue with L2 - connect directly to bottom_center
            d.move_from(branch3_bottom.start)
            d += (l2 := elm.Inductor().left().label('$L_2$'))
            d += elm.Line().to(bottom_center.start)

            # Connect bottom_left to bottom_center
            d.move_from(bottom_left.start)
            d += elm.Line().to(bottom_center.start)

            # Add flux indicator for left loop (EJ1-L1 || EJ0)
            # Draw a circular arrow in the middle of the left loop
            loop1_center_x = (ej1.center[0] + ej0.center[0]) / 2
            loop1_center_y = top.start[1] - 1.5  # Midpoint vertically
            d += elm.Arc2(arrow='->').at((loop1_center_x - 0.5, loop1_center_y)).to((loop1_center_x + 0.5, loop1_center_y))
            d += elm.Label().at((loop1_center_x, loop1_center_y + 0.3)).label(r'$\Phi_1$', fontsize=10)

            # Add flux indicator for right loop (EJ0 || EJ2-L2)
            # Draw a circular arrow in the middle of the right loop
            loop2_center_x = (ej0.center[0] + ej2.center[0]) / 2
            loop2_center_y = top.start[1] - 1.5  # Midpoint vertically
            d += elm.Arc2(arrow='->').at((loop2_center_x - 0.5, loop2_center_y)).to((loop2_center_x + 0.5, loop2_center_y))
            d += elm.Label().at((loop2_center_x, loop2_center_y + 0.3)).label(r'$\Phi_2$', fontsize=10)

            # Ground connection
            d.move_from(bottom_center.start)
            d += elm.Line().down().length(0.5)
            d += elm.Line().left().length(6)
            d += elm.Ground()

            filepath = f'{results_dir}/circuit_diagram.png'
            d.save(filepath, dpi=300)

        print(f"Circuit diagram saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error drawing circuit: {e}")
        return None


if __name__ == '__main__':
    # Test the circuit diagram generation
    draw_circuit()
