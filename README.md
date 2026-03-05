# Superconducting Circuit Analysis

A modular framework for analyzing different superconducting circuits.

## Project Structure

```
Code/
├── Sc_analysis.py                    # Main wrapper - runs all circuits
├── Cooper_box.py                     # Cooper box analysis module
├── circuit_diagram_cooper_box.py    # Cooper box circuit diagram
├── new_circuit_template.py          # Template for new circuits
├── circuit_diagram_template.py      # Template for circuit diagrams
├── explanation.txt                  # Detailed explanation of Cooper box calculation
└── Results_*/                       # Output directories
```

## How to Run

### Main Wrapper (Recommended)

Run the main wrapper to see available circuits:
```bash
python Sc_analysis.py
```

List all available circuits:
```bash
python Sc_analysis.py list
```

Run a specific circuit:
```bash
python Sc_analysis.py cooper_box
```

Run all circuits:
```bash
python Sc_analysis.py all
```

### Individual Circuits

You can also run individual circuit analyses directly:
```bash
python Cooper_box.py
```

Or just generate a circuit diagram:
```bash
python circuit_diagram_cooper_box.py
```

## Adding a New Circuit

1. **Copy the templates:**
   ```bash
   cp new_circuit_template.py My_new_circuit.py
   cp circuit_diagram_template.py circuit_diagram_my_circuit.py
   ```

2. **Edit `My_new_circuit.py`:**
   - Implement `build_hamiltonian()` - construct the Hamiltonian matrix
   - Implement `compute_spectrum()` - calculate energy eigenvalues
   - Implement `analyze_circuit()` - generate plots and analysis
   - Import your circuit diagram function

3. **Edit `circuit_diagram_my_circuit.py`:**
   - Add circuit elements using schemdraw
   - Example elements: `elm.SourceV()`, `elm.Capacitor()`, `elm.Josephson()`, etc.

4. **Register in `Sc_analysis.py`:**
   ```python
   CIRCUITS = {
       'cooper_box': {...},
       'my_circuit': {
           'name': 'My New Circuit',
           'description': 'Description of what it does',
           'module': 'My_new_circuit',
           'function': 'analyze_circuit'
       },
   }
   ```

5. **Run it:**
   ```bash
   python Sc_analysis.py my_circuit
   ```

## Available Circuits

### Cooper-Pair Box / Transmon
- **File**: `Cooper_box.py`
- **Description**: Energy levels as a function of gate charge for different EJ/EC ratios
- **Hamiltonian**: H = 4 E_C Σ(n - n_g)² |n⟩⟨n| - (E_J/2) Σ(|n⟩⟨n+1| + |n+1⟩⟨n|)
- **Run**: `python Sc_analysis.py cooper_box`
- **Outputs**:
  - Energy level plots vs gate charge
  - Circuit diagram
- **Details**: See `explanation.txt` for detailed calculation explanation

## Requirements

- Python 3.8+
- numpy
- matplotlib
- schemdraw
- LaTeX (for circuit diagrams - MiKTeX or TeX Live)

Install Python dependencies:
```bash
pip install numpy matplotlib schemdraw
```

## Output

Each circuit creates its own results directory:
- `Results_Cooper_box/` - Cooper box results
- `Results_<circuit_name>/` - Other circuit results

Output includes:
- Energy level plots (PNG)
- Circuit diagrams (PNG)
- Additional analysis plots (depending on circuit)

## File Descriptions

### Core Files
- **Sc_analysis.py**: Main entry point, orchestrates all circuit analyses
- **Cooper_box.py**: Cooper box/Transmon analysis implementation
- **circuit_diagram_cooper_box.py**: Draws Cooper box circuit using schemdraw

### Templates
- **new_circuit_template.py**: Template for implementing new circuit analyses
- **circuit_diagram_template.py**: Template for drawing new circuit diagrams

### Documentation
- **explanation.txt**: Detailed step-by-step explanation of Cooper box calculation
- **README.md**: This file

## Example Usage

```bash
# See all available circuits
python Sc_analysis.py

# Run Cooper box analysis
python Sc_analysis.py cooper_box

# Run all available circuits
python Sc_analysis.py all

# Run just the Cooper box module directly
python Cooper_box.py

# Generate just the circuit diagram
python circuit_diagram_cooper_box.py
```
