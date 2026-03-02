"""
Superconducting Circuit Analysis - Main Wrapper

This script runs different superconducting circuit analyses.
Add new circuit analyses to the CIRCUITS dictionary to include them.
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Available circuit analyses
CIRCUITS = {
    'cooper_box': {
        'name': 'Cooper-Pair Box / Transmon',
        'description': 'Energy levels as a function of gate charge for different EJ/EC ratios',
        'module': 'Cooper_box',
        'function': 'analyze_cooper_box'
    },
    'triple_josephson': {
        'name': 'Triple Josephson Junction',
        'description': 'Energy levels vs two external fluxes for (EJ1+L1)||EJ0||(EJ2+L2) circuit',
        'module': 'Triple_josephson',
        'function': 'analyze_triple_josephson'
    },
    # Add more circuits here as you implement them
    # 'flux_qubit': {
    #     'name': 'Flux Qubit',
    #     'description': 'Flux qubit analysis',
    #     'module': 'Flux_qubit',
    #     'function': 'analyze_flux_qubit'
    # },
}


def list_circuits():
    """Display all available circuit analyses."""
    print("\n" + "="*60)
    print("Available Superconducting Circuit Analyses")
    print("="*60)
    for key, info in CIRCUITS.items():
        print(f"\n  {key}:")
        print(f"    Name: {info['name']}")
        print(f"    Description: {info['description']}")
    print("\n" + "="*60)


def run_circuit(circuit_key):
    """
    Run a specific circuit analysis.

    Parameters
    ----------
    circuit_key : str
        Key identifying the circuit in CIRCUITS dictionary
    """
    if circuit_key not in CIRCUITS:
        print(f"Error: Circuit '{circuit_key}' not found.")
        list_circuits()
        return False

    circuit_info = CIRCUITS[circuit_key]

    try:
        print(f"\n{'='*60}")
        print(f"Running: {circuit_info['name']}")
        print(f"{'='*60}\n")

        # Dynamically import and run the circuit analysis
        module = __import__(circuit_info['module'])
        analysis_func = getattr(module, circuit_info['function'])
        analysis_func()

        print(f"\n{'='*60}")
        print(f"[OK] {circuit_info['name']} - Complete")
        print(f"{'='*60}\n")
        return True

    except ImportError as e:
        print(f"Error: Could not import module '{circuit_info['module']}'")
        print(f"Details: {e}")
        return False
    except AttributeError as e:
        print(f"Error: Function '{circuit_info['function']}' not found in module")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"Error running {circuit_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_circuits():
    """Run all available circuit analyses."""
    print("\n" + "="*60)
    print("Running ALL Circuit Analyses")
    print("="*60)

    results = {}
    for key in CIRCUITS.keys():
        results[key] = run_circuit(key)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for key, success in results.items():
        status = "[OK] Success" if success else "[FAILED]"
        print(f"  {CIRCUITS[key]['name']}: {status}")
    print("="*60 + "\n")


def main():
    """Main entry point for the script."""
    if len(sys.argv) == 1:
        # No arguments - show menu
        list_circuits()
        print("\nUsage:")
        print("  python Sc_analysis.py <circuit_key>  - Run specific circuit")
        print("  python Sc_analysis.py all            - Run all circuits")
        print("  python Sc_analysis.py list           - List available circuits")
        print()

    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()

        if arg == 'list':
            list_circuits()
        elif arg == 'all':
            run_all_circuits()
        elif arg in CIRCUITS:
            run_circuit(arg)
        else:
            print(f"Error: Unknown circuit or command '{arg}'")
            list_circuits()
    else:
        print("Error: Too many arguments")
        print("Usage: python Sc_analysis.py [circuit_key|all|list]")


if __name__ == '__main__':
    main()
