# BiSQUID Hamiltonian Analysis with LC Oscillators

## Executive Summary

This document analyzes the proper quantum Hamiltonian for the BiSQUID (Bi-SQUID) circuit with explicit capacitors in parallel with each Josephson junction. The addition of capacitors C₁, C₂, C₀ creates **LC oscillator modes** that significantly modify the system dynamics and require an extended Hilbert space treatment.

## Circuit Topology

**Complete Circuit: Cg + ((EJ1 || C1) + L1) || (EJ0 || C0) || ((EJ2 || C2) + L2)**

### Branch Structure

- **Branch 1**: (EJ₁ || C₁) in series with L₁
- **Branch 2**: (EJ₀ || C₀) alone (reference branch)
- **Branch 3**: (EJ₂ || C₂) in series with L₂

All three branches connect in parallel between the superconducting island (top node) and ground (bottom node).

### External Fluxes

- **Φ₁**: External flux threading the loop formed by Branch 1 and Branch 2
- **Φ₂**: External flux threading the loop formed by Branch 2 and Branch 3

## Current Hamiltonian Limitations

### Existing Implementation

The current code in `BiSQUID.py` uses a **single degree of freedom** (island charge n) and treats inductors perturbatively:

```
H = Σₙ 4E_C n² |n⟩⟨n|
    - (E_J^eff/2)[e^(-iφ₀) Σₙ |n⟩⟨n+1| + e^(iφ₀) Σₙ |n+1⟩⟨n|]
    + (g₁/2) Σₙ |n⟩⟨n| - (g₁/4) Σₙ [e^(i4πf₁)|n⟩⟨n+2| + e^(-i4πf₁)|n⟩⟨n-2|]
    + (g₂/2) Σₙ |n⟩⟨n| - (g₂/4) Σₙ [e^(i4πf₂)|n⟩⟨n+2| + e^(-i4πf₂)|n⟩⟨n-2|]
```

Where:
- **4E_C n²**: Charging energy of island
- **E_J^eff**: Effective Josephson energy from three junctions
- **g₁, g₂**: Inductive coupling strengths `g = (2πLI_c/Φ₀)² E_L`

### Problems with Current Approach

1. **Ignores oscillator dynamics**: C₁ and C₂ create additional quantum degrees of freedom
2. **Assumes weak coupling**: Valid only when `E_L >> g`, but current parameters show `g ≈ 62.5 >> E_J ≈ 8-10`
3. **Charge basis only**: LC oscillators are better described in Fock/number basis
4. **Perturbative treatment**: The n↔n±2 terms assume small inductor phases

### Parameter Analysis

From current code:
```python
E_C = 1.0        # Reference scale
E_J0 = 10.0      # Center junction
E_J1 = E_J2 = 8.0
E_L ≈ 108.0      # Inductive energy
g_1 = g_2 ≈ 62.5 # Coupling strength
```

**Critical observation**: `g >> E_J` indicates **strong inductive coupling**, making perturbative treatment questionable.

## Correct Quantum Description

### Degrees of Freedom

The system has **THREE independent quantum degrees of freedom**:

1. **Island charge n** (on Cg)
   - Conjugate variable: island phase φ
   - Basis: |n⟩ with n ∈ [-n_max, +n_max]

2. **LC Oscillator 1** (formed by C₁ and L₁)
   - Number of excitations: m₁
   - Basis: |m₁⟩ with m₁ ∈ [0, m_max]
   - Ladder operators: a₁†, a₁

3. **LC Oscillator 2** (formed by C₂ and L₂)
   - Number of excitations: m₂
   - Basis: |m₂⟩ with m₂ ∈ [0, m_max]
   - Ladder operators: a₂†, a₂

### Complete Hilbert Space

```
|ψ⟩ = Σ c_nm₁m₂ |n, m₁, m₂⟩
```

Where:
- |n⟩_island ⊗ |m₁⟩_osc1 ⊗ |m₂⟩_osc2

**Matrix dimension**: (2n_max + 1) × (m_max + 1) × (m_max + 1)

For example: n_max = 30, m_max = 20 → 61 × 21 × 21 ≈ 27,000 dimensional Hilbert space

## Full Hamiltonian Structure

### H = H_island + H_osc1 + H_osc2 + H_J + H_cross

### 1. Island Charging Energy

```
H_island = 4E_C Σₙ n² |n⟩⟨n| ⊗ I_osc1 ⊗ I_osc2
```

### 2. LC Oscillator Hamiltonians

```
H_osc1 = I_island ⊗ ℏω₁(a₁†a₁ + 1/2) ⊗ I_osc2
H_osc2 = I_island ⊗ I_osc1 ⊗ ℏω₂(a₂†a₂ + 1/2)
```

Where the oscillator frequencies are:
```
ω₁ = 1/√(L₁C₁)
ω₂ = 1/√(L₂C₂)
```

In energy units:
```
ℏω₁ = √(E_L1 · E_C1)
ℏω₂ = √(E_L2 · E_C2)
```

Where:
- E_L = Φ₀²/(4π²L) is the inductive energy
- E_C = e²/(2C) is the capacitive energy

### 3. Josephson Coupling Terms

The Josephson energy depends on gauge-invariant phases across each junction:

```
H_J = -E_J1 cos(φ_EJ1) - E_J0 cos(φ_EJ0) - E_J2 cos(φ_EJ2)
```

From flux quantization:
```
φ_EJ1 = φ + 2πf₁ - φ_L1
φ_EJ0 = φ
φ_EJ2 = φ + 2πf₂ - φ_L2
```

Where:
- φ = island phase = conjugate to charge n
- f₁ = Φ₁/Φ₀, f₂ = Φ₂/Φ₀ (reduced fluxes)
- φ_L1, φ_L2 = inductor phases

### 4. Oscillator-Phase Relations

The inductor phases are quantum operators related to oscillator variables:

```
φ_L1 = φ_ZPF1 (a₁ + a₁†)
φ_L2 = φ_ZPF2 (a₂ + a₂†)
```

The **zero-point fluctuation** (ZPF) of phase is:

```
φ_ZPF = (ℏ/2)^(1/2) · (1/√(LC))^(1/2) · (2e/ℏ)
      = (2e)^(1/2) · (ℏ/(LC))^(1/4)
```

Or equivalently:
```
φ_ZPF = (E_C/E_L)^(1/4)
```

### 5. Effective Josephson Hamiltonian

Expanding the cosines:

```
cos(φ + 2πf₁ - φ_L1) = cos(φ + 2πf₁) cos(φ_L1) + sin(φ + 2πf₁) sin(φ_L1)
```

The φ_L terms become:
```
cos(φ_ZPF(a + a†)) ≈ 1 - (φ_ZPF²/2)(a + a†)² + ...
sin(φ_ZPF(a + a†)) ≈ φ_ZPF(a + a†) - (φ_ZPF³/6)(a + a†)³ + ...
```

This creates:
- **Diagonal terms**: (a†a) → oscillator number-dependent shifts
- **Off-diagonal terms**: (a + a†) → oscillator excitation/de-excitation
- **Multi-photon terms**: (a†a†), (aa), etc.

## Matrix Elements in Full Basis

### Diagonal Elements

```
⟨n,m₁,m₂|H|n,m₁,m₂⟩ = 4E_C n² + ℏω₁(m₁ + 1/2) + ℏω₂(m₂ + 1/2)
                        + ⟨Josephson energy diagonal contributions⟩
```

### Island Charge Transitions (n ↔ n±1)

From the phase operator φ (conjugate to n):
```
⟨n±1,m₁,m₂|H|n,m₁,m₂⟩ = -E_J/2 · ⟨m₁,m₂|exp[±i(φ_L1 + φ_L2)]|m₁,m₂⟩
```

This creates **displaced oscillator states**.

### Oscillator Transitions (m₁ ↔ m₁±1, m₂ ↔ m₂±1)

The inductor phase operators create:
```
⟨n,m₁±1,m₂|H|n,m₁,m₂⟩ ~ √(m₁ + 1/2 ± 1/2)
⟨n,m₁,m₂±1|H|n,m₁,m₂⟩ ~ √(m₂ + 1/2 ± 1/2)
```

### Cross-Coupling Terms

The Josephson Hamiltonian creates **simultaneous transitions**:
```
⟨n±1,m₁±1,m₂|H|n,m₁,m₂⟩ ≠ 0
⟨n±1,m₁,m₂±1|H|n,m₁,m₂⟩ ≠ 0
```

This represents **entanglement** between island charge and oscillator excitations.

## Physical Regimes

### Regime A: High-Frequency Oscillators (ℏω >> E_J, E_C)

**Condition**: ℏω₁, ℏω₂ >> 10 E_C

**Physics**:
- Oscillators remain in ground state |m₁=0, m₂=0⟩
- Can project full Hamiltonian onto this subspace
- Recovers effective single-DOF description

**Effective Hamiltonian**:
```
H_eff = 4E_C n² - E_J1 cos(φ + 2πf₁) - E_J0 cos(φ) - E_J2 cos(φ + 2πf₂)
```

Plus small corrections from virtual excitations of oscillators (Lamb shift).

### Regime B: Low-Frequency Oscillators (ℏω << E_J)

**Condition**: ℏω₁, ℏω₂ << 0.1 E_C

**Physics**:
- Oscillators highly excited (large ⟨m⟩)
- Semiclassical limit: φ_L becomes classical variable
- Phase distribution is Gaussian with large width

**Treatment**:
- Replace operators with classical fields
- Minimize energy with respect to φ_L1, φ_L2
- Leads to self-consistent equations

### Regime C: Intermediate Coupling (ℏω ~ E_J)

**Condition**: 0.1 E_C < ℏω < 10 E_C

**Physics**:
- Full quantum effects in oscillators
- Strong entanglement between island and oscillators
- Multi-photon processes important

**Treatment**:
- Requires full multi-dimensional Hamiltonian
- No simplifications possible
- Most interesting regime for novel physics!

### Current Parameter Regime

Based on:
```
E_L ≈ 108 E_C
g ≈ 62.5
L = 5 nH
```

We need to determine C₁, C₂ to find ℏω = √(E_L · E_C).

**Critical question**: What are the junction capacitances?

For typical Josephson junctions:
- Small junctions: C ~ 1-10 fF → E_C ~ 1-10 GHz
- Large junctions: C ~ 10-100 fF → E_C ~ 0.1-1 GHz

If C₁ ~ 10 fF and L₁ = 5 nH:
```
ℏω₁ = √(E_L1 · E_C1) = √(108 · 1) E_C ≈ 10 E_C
```

This suggests **Regime A** (high-frequency oscillators).

If C₁ ~ 100 fF:
```
E_C1 ~ 0.1 E_C
ℏω₁ ~ √(108 · 0.1) ≈ 3 E_C
```

This suggests **Regime C** (intermediate coupling).

## Implementation Strategies

### Strategy 1: Adiabatic Elimination (Regime A)

**When to use**: ℏω >> E_J

**Method**:
1. Treat oscillators in lowest-order perturbation theory
2. Integrate out oscillator degrees of freedom
3. Result: effective potential for island

**Effective Hamiltonian**:
```
H_eff(n, φ) = 4E_C n² - E_J^eff(f₁,f₂) cos(φ - φ₀)
```

Where E_J^eff and φ₀ are renormalized by oscillator vacuum fluctuations.

**Advantage**: Small Hilbert space (~ 100 dimensions)

**Disadvantage**: Misses oscillator excitations and entanglement

### Strategy 2: Full Quantum Treatment (Regime C)

**When to use**: ℏω ~ E_J or when oscillator physics is important

**Method**:
1. Build full Hamiltonian in |n,m₁,m₂⟩ basis
2. Diagonalize large sparse matrix
3. Extract energy eigenvalues and eigenstates

**Matrix structure**:
- Mostly sparse (pentadiagonal-like in each subspace)
- Size: (2n_max+1) × (m_max+1)²

**Implementation**:
```python
def build_hamiltonian_full(f1, f2, n_max=20, m_max=10):
    dim_n = 2*n_max + 1
    dim_m1 = m_max + 1
    dim_m2 = m_max + 1
    total_dim = dim_n * dim_m1 * dim_m2

    H = scipy.sparse.lil_matrix((total_dim, total_dim), dtype=complex)

    # Map (n, m1, m2) → single index
    def index(n, m1, m2):
        return ((n+n_max) * dim_m1 + m1) * dim_m2 + m2

    # Fill matrix elements...
```

**Advantage**: Exact treatment, captures all physics

**Disadvantage**: Computationally expensive (> 10⁴ dimensions)

### Strategy 3: Semiclassical Approximation (Regime B)

**When to use**: ℏω << E_J and large oscillator excitations

**Method**:
1. Replace a, a† with complex numbers α, α*
2. Minimize total energy ⟨H⟩ over α₁, α₂ for each n, φ
3. Self-consistent solution

**Equations**:
```
∂⟨H⟩/∂α₁* = 0  →  α₁(φ, f₁)
∂⟨H⟩/∂α₂* = 0  →  α₂(φ, f₂)
```

Then:
```
H_eff(φ) = ⟨H⟩|_{α₁=α₁(φ), α₂=α₂(φ)}
```

**Advantage**: Moderate computational cost

**Disadvantage**: Misses quantum fluctuations in oscillators

## Recommended Next Steps

### Step 1: Determine Junction Capacitances

**Critical parameters needed**:
- C₁, C₂, C₀ (junction capacitances)
- Or equivalently: E_C1, E_C2, E_C0

**Methods**:
1. Estimate from junction area and critical current
2. Use typical values from literature
3. Measure experimentally if fabricated

### Step 2: Calculate Oscillator Frequencies

```
ℏω₁ = √(E_L1 · E_C1)
ℏω₂ = √(E_L2 · E_C2)
```

Compare to E_J and E_C to determine regime.

### Step 3: Choose Implementation Strategy

- **If ℏω > 5 E_J**: Use adiabatic elimination (Strategy 1)
- **If 0.2 E_J < ℏω < 5 E_J**: Use full quantum (Strategy 2)
- **If ℏω < 0.2 E_J**: Use semiclassical (Strategy 3)

### Step 4: Implement and Compare

1. Implement chosen strategy
2. Compare energy spectra with original code
3. Identify new physics (if any):
   - Oscillator-mediated coupling
   - Multi-photon transitions
   - Entanglement between subsystems

## Additional Physics Considerations

### 1. Josephson Junction Capacitance vs. External Capacitance

In the circuit diagram, C₁, C₂, C₀ could represent:

**Option A**: Intrinsic junction capacitance
- Always present in real junctions
- Typically C_J = 1-100 fF

**Option B**: Intentionally added shunt capacitors
- Can be much larger (pF range)
- Used to engineer oscillator frequencies

**Implication**: Need to clarify which interpretation is intended.

### 2. Dissipation and Damping

Real oscillators have losses. The quality factor Q:
```
Q = ωL/R
```

If Q >> 1, quantum treatment is valid. If Q ~ 1, need to include dissipation.

### 3. Anharmonicity from Josephson Nonlinearity

The Josephson cosine creates anharmonic corrections to oscillators:
```
cos(φ - φ_L) ≈ cos(φ) + sin(φ)·φ_L - (1/2)cos(φ)·φ_L² - ...
```

The φ_L² term creates **Kerr nonlinearity** in oscillators:
```
H_Kerr = K₁(a₁†a₁)² + K₂(a₂†a₂)²
```

This is important for quantum information applications.

### 4. Flux-Dependent Effective Mass

The oscillator frequencies depend on flux:
```
ω₁(f₁) = ω₁⁰ · √[1 + (E_J1/E_L1)cos(2πf₁)]
```

This creates flux-tunable resonators.

## Comparison with Original Hamiltonian

### What the Original Code Gets Right

1. ✓ Correct flux quantization (2πf₁, 2πf₂ terms)
2. ✓ Effective Josephson energy from three junctions
3. ✓ Island charging energy 4E_C n²

### What the Original Code Misses

1. ✗ Oscillator energy levels ℏω(m + 1/2)
2. ✗ Multi-level oscillator structure
3. ✗ Entanglement between island and oscillators
4. ✗ Oscillator-mediated interactions
5. ✗ Correct treatment of strong coupling regime (g >> E_J)

### When Original Code is Valid

The perturbative treatment is valid when:
1. **ℏω >> E_J**: Oscillators in ground state
2. **g << E_L**: Weak inductive coupling
3. **T = 0**: No thermal excitations

Current parameters violate condition 2 (g ≈ 62, E_L ≈ 108), suggesting corrections are needed.

## Conclusion

The addition of explicit capacitors C₁, C₂, C₀ fundamentally changes the BiSQUID from a **single-mode system** (island charge) to a **multi-mode system** (island + two oscillators).

The proper quantum treatment requires:
1. **Extended Hilbert space**: |n, m₁, m₂⟩
2. **Coupled dynamics**: Island and oscillators are entangled
3. **Regime-dependent methods**: Choice of approximation depends on ℏω/E_J

The current code provides a useful starting point but should be extended to capture the full quantum behavior of the LC oscillators, especially given the strong coupling parameters (g >> E_J).

## References and Further Reading

### Key Papers on Circuit QED and Superconducting Circuits

1. **Koch et al., Phys. Rev. A 76, 042319 (2007)**
   - Transmon qubit, strong E_J/E_C regime

2. **Manucharyan et al., Science 326, 113 (2009)**
   - Fluxonium qubit with large inductance

3. **Nguyen et al., Phys. Rev. X 9, 041041 (2019)**
   - High-coherence fluxonium qubits

4. **Gyenis et al., PRX Quantum 2, 010339 (2021)**
   - Protected qubits with multiple junctions

### Useful Reviews

1. **Devoret & Schoelkopf, Science 339, 1169 (2013)**
   - Superconducting circuits for quantum information

2. **Blais et al., Rev. Mod. Phys. 93, 025005 (2021)**
   - Circuit quantum electrodynamics

