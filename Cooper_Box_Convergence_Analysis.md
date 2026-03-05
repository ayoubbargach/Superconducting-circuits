# Cooper Box Truncation Convergence Analysis

## Overview

This document explains why the Cooper box eigenvalue calculation converges **exponentially fast** with respect to the Fock space truncation parameter `n_max`, and derives the mathematical formula for the convergence rate.

## The Hamiltonian

The Cooper box Hamiltonian in the charge basis {|n⟩} is:

```
H = 4E_C Σ_n (n - n_g)² |n⟩⟨n| - (E_J/2) Σ_n (|n⟩⟨n+1| + |n+1⟩⟨n|)
```

In matrix form (with E_C = 1), this is a **tridiagonal matrix**:

```
       n=-3    n=-2    n=-1    n=0     n=1     n=2     n=3
n=-3 [  49.0   -0.5     0       0       0       0       0   ]
n=-2 [ -0.5    25.0   -0.5     0       0       0       0   ]
n=-1 [  0      -0.5    9.0   -0.5     0       0       0   ]
n=0  [  0       0     -0.5    1.0   -0.5     0       0   ]
n=1  [  0       0      0     -0.5    1.0   -0.5     0   ]
n=2  [  0       0      0      0     -0.5    9.0   -0.5 ]
n=3  [  0       0      0      0      0     -0.5    25.0]
```

**Key properties:**
- **Diagonal elements**: E_diag(n) = 4(n - n_g)² (charging energy)
- **Off-diagonal elements**: -E_J/2 (Josephson coupling)
- **Tridiagonal structure**: Each basis state |n⟩ couples only to |n±1⟩

## Why Convergence is Exponentially Fast

### 1. Energy Scale Separation

The diagonal energies grow **quadratically** with |n|:

```
E_diag(n) = 4(n - n_g)² E_C
```

For n_g ≈ 0.5:
- n = 0, 1: E_diag ≈ 1 E_C
- n = ±2: E_diag ≈ 9 E_C
- n = ±3: E_diag ≈ 25 E_C
- n = ±5: E_diag ≈ 81 E_C
- n = ±10: E_diag ≈ 361 E_C

Meanwhile, the coupling strength is constant: |⟨n|H|n±1⟩| = E_J/2 ≈ 0.5 E_C (for E_J/E_C = 1).

### 2. Wavefunction Localization

The ground state wavefunction ψ_0(n) must decay exponentially away from n ≈ n_g due to the tridiagonal structure and energy scale separation.

**Heuristic derivation:**

Consider the ground state as ψ_0(n) ≈ A·exp(-α|n - n̄|) where n̄ ≈ n_g.

From the Schrödinger equation in the charge basis:
```
E_diag(n)ψ(n) - (E_J/2)[ψ(n+1) + ψ(n-1)] = E_0 ψ(n)
```

For large |n| (far from the minimum), E_diag(n) >> E_0, so:
```
4(n - n_g)² ψ(n) ≈ (E_J/2)[ψ(n+1) + ψ(n-1)]
```

Assuming exponential decay ψ(n) = A·exp(-α|n|) for |n| large, we get:
```
4n² exp(-α|n|) ≈ (E_J/2) exp(-α|n|)[exp(-α) + exp(α)]
```

This gives approximately:
```
α ≈ asinh(4n²/(E_J))
```

For large n, this grows logarithmically: α ≈ ln(8n²/E_J)

But more importantly, for **moderate n** (the regime we care about), we can estimate:

```
ψ(n+1)/ψ(n) ≈ exp(-α) ≈ E_J/(8(n-n_g)²)
```

### 3. Perturbation Theory Estimate

When we **add** a new basis state |n_max + 1⟩ to our truncated Hilbert space, its effect on the ground state energy is given by second-order perturbation theory:

```
ΔE_0 ≈ |⟨ψ_0|H|n_max+1⟩|² / [E_diag(n_max+1) - E_0]
```

Where:
- **Numerator**: |⟨ψ_0|H|n_max+1⟩|² ≈ (E_J/2)² |ψ_0(n_max)|² ∝ exp(-2α·n_max)
- **Denominator**: E_diag(n_max+1) - E_0 ≈ 4n_max² E_C

Therefore:
```
ΔE_0 ≈ (E_J/2)² exp(-2α·n_max) / (4n_max² E_C)
```

## Convergence Formula

The error in the ground state energy due to truncation at n_max is:

```
|E_0(n_max) - E_0(∞)| ≈ C · exp(-2α·n_max) / n_max²
```

Where:
- **C** is a constant of order (E_J)²/E_C
- **α** is the decay rate of the wavefunction, approximately:
  ```
  α ≈ ln[4(n_max - n_g)²/E_J] for large n_max
  ```

### For typical parameters (E_J/E_C = 1, n_g = 0.5):

At n_max = 5:
- Decay factor: exp(-2α·5) ≈ exp(-6) ≈ 2.5×10⁻³
- Polynomial suppression: 1/25
- **Total error**: ~10⁻⁴ to 10⁻⁵

At n_max = 10:
- Decay factor: exp(-2α·10) ≈ exp(-12) ≈ 6×10⁻⁶
- Polynomial suppression: 1/100
- **Total error**: ~10⁻⁷ to 10⁻⁸

## Numerical Verification

From the code (E_J/E_C = 1.0, n_g = 0.5):

```
n_max   E_0 (E_C)        |Error|       Scaling
  1     0.4851043830     1.45e-02     (reference)
  2     0.4706718347     1.75e-05     ~1000× reduction
  3     0.4706543586     3.68e-09     ~5000× reduction
  4     0.4706543549     1.52e-13     ~24000× reduction
  5     0.4706543549     ~0           (machine precision)
```

**Observed decay rate**:
- From n=1 to n=2: factor of ~1000
- From n=2 to n=3: factor of ~5000
- This is **super-exponential** for small n, then becomes purely exponential

The super-exponential behavior for small n_max is because:
1. At n_max = 1, 2, the Hilbert space is so small that important states are missing
2. At n_max = 3, 4, we enter the exponential regime
3. At n_max ≥ 5, we reach machine precision

## Why Convergence is So Fast

### Summary of factors:

1. **Quadratic potential**: E_diag(n) ∝ n² creates strong confinement
2. **Weak coupling**: E_J/(8n²) << 1 for |n| > 2
3. **Exponential wavefunction decay**: ψ(n) ∝ exp(-α|n|)
4. **Tridiagonal structure**: New states at |n| = n_max+1 only couple to |n| = n_max

**Result**: Each additional charge state contributes exponentially less to the ground state wavefunction.

## Implications for Different Regimes

### Charge Regime (E_J/E_C << 1)
- Ground state more localized
- Even faster convergence
- n_max = 3-4 sufficient

### Transmon Regime (E_J/E_C >> 1)
- Ground state slightly more delocalized
- Still very fast convergence
- n_max = 5-8 sufficient

### High Energy States
- Higher excited states are more delocalized
- Require larger n_max
- But still exponential convergence

## Practical Rule of Thumb

For the Cooper box:

```
n_max ≥ 3·σ_n
```

where σ_n is the charge spread of the wavefunction:
```
σ_n ≈ √(E_J/(8E_C))
```

For E_J/E_C = 1: σ_n ≈ 0.35, so n_max ≥ 2 is sufficient
For E_J/E_C = 10: σ_n ≈ 1.1, so n_max ≥ 4 is sufficient
For E_J/E_C = 100: σ_n ≈ 3.5, so n_max ≥ 11 is sufficient

## Conclusion

The exponential convergence of the Cooper box eigenvalues is a consequence of:
1. The quadratic growth of diagonal energies
2. The constant (weak) off-diagonal coupling
3. The resulting exponential localization of wavefunctions

This makes the Cooper box a numerically **very efficient** system to simulate - even small truncations give excellent accuracy for low-lying states.

The convergence rate is approximately:
```
Error ∝ exp(-2·ln(n_max)·n_max) / n_max² = n_max^(-2n_max-2)
```

This is **super-exponential** and explains why n_max = 5 reaches machine precision even for excited states.
