"""
Cooper-Pair Box Circuit Diagram

Draws the circuit diagram for the Cooper-pair box / Transmon qubit:
Circuit: Vg + Cg + (EJ || CJ)
"""

import os
try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    print("Warning: schemdraw not available, circuit diagram will be skipped")


def draw_cooper_box_circuit(results_dir='Results_Cooper_box'):
    """
    Draw the Cooper-pair box circuit diagram.

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

            # Voltage source Vg going up
            d += elm.SourceV().up().label('$V_g$')

            # Gate capacitor Cg going right
            d += elm.Capacitor().right().label('$C_g$')

            # Save the top node position
            d += (top := elm.Dot())

            # Josephson junction going down
            d += elm.Josephson().down().label('$E_J$')
            d += (bottom_right := elm.Dot())

            # Capacitor CJ - go back to top and draw parallel path
            d.move_from(top.start)
            d += elm.Line().right().length(1.5)
            d += elm.Capacitor().down().label('$C_J(E_C)$')

            # Connect back to bottom
            d += elm.Line().to(bottom_right.start)
            d.move_from(bottom_right.start, dx=0, dy=0)
            d += elm.Line().left().length(3)
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
    draw_cooper_box_circuit()
