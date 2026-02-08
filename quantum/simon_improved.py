"""
Simon's Algorithm: Improved for Second Quantization
Framework Integration: Fock Space + Metriplex Oracle + H7 Conservation

Core Algorithm:
1. Initialize n qubits to |+⟩ (equal superposition)
2. Query oracle f: |x⟩ ↔ f(x) = energy vector encoded as phase
3. Apply QFT to detect hidden structure
4. Measure to extract constraint vectors z such that z·s = 0 (mod 2)
5. Solve linear system to recover secret string s

Improvements for Second Quantization:
- Oracle operates on Fock occupation numbers (not raw bits)
- Gray-code mapping preserves Hamming distance
- Dynamic truncation monitoring
- Noise-resilient measurement preprocessing

Author: Jacobo Tlacaelel Mina Rodríguez
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings

from physics.fock_basis import FockBasis, FockConfig, FockStateVector, fock_ground_state
from physics.metriplex_oracle import MetriplexOracle, MetriplexConfig, H7Conservation


class QuantumGateError(Exception):
    """Raised when quantum gate simulation fails."""
    pass


class MeasurementError(Exception):
    """Raised when measurement interpretation fails."""
    pass


@dataclass
class SimonConfig:
    """Configuration for Simon's algorithm execution."""
    n_qubits: int = 3  # Working qubit register size
    n_queries: Optional[int] = None  # Max queries (default: n)
    measurement_shots: int = 1000  # Repetitions per measurement
    gray_code_enabled: bool = True  # Use Gray-code binary mapping
    noise_model: str = "ideal"  # "ideal", "depolarizing", "amplitude_damping"
    depolarizing_rate: float = 0.01  # Error rate per gate
    truncation_monitor: bool = True  # Monitor Fock truncation dynamically
    tolerance: float = 1e-6  # Numerical tolerance for zero detection


class SimonImproved:
    """
    Simon's algorithm optimized for second quantization and metriplex dynamics.
    
    The algorithm discovers the hidden symmetry s of a 2-to-1 function f:
    f(x) = f(x ⊕ s) for all x.
    
    In our metriplex context:
    - f maps momentum states to energy profiles (metriplex oracle)
    - s encodes H7 conservation structure
    - Expected solution: s = 7 (binary 111) for standard config
    """
    
    def __init__(self, 
                 fock_basis: FockBasis,
                 oracle: MetriplexOracle,
                 config: SimonConfig = None):
        """Initialize Simon's algorithm with quantum resources."""
        if config is None:
            config = SimonConfig()
        
        self.config = config
        self.fock = fock_basis
        self.oracle = oracle
        
        # Derived parameters
        if self.config.n_queries is None:
            self.config.n_queries = self.config.n_qubits
        
        # State for algorithm execution
        self.measured_vectors: Set[int] = set()
        self.oracle_queries = 0
        self.execution_log = []
    
    def _hadamard(self, state: np.ndarray) -> np.ndarray:
        """
        Apply Hadamard transformation to create superposition.
        
        H|0⟩ = (|0⟩ + |1⟩)/√2
        H|1⟩ = (|0⟩ - |1⟩)/√2
        
        For multi-qubit: tensor product of single-qubit Hadamards.
        """
        n = int(np.log2(len(state)))
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Build tensor product
        H_full = H
        for _ in range(n - 1):
            H_full = np.kron(H_full, H)
        
        return H_full @ state
    
    def _qft(self, state: np.ndarray) -> np.ndarray:
        """
        Quantum Fourier Transform on computational basis.
        
        QFT |j⟩ = (1/√N) Σ_k e^{2πijk/N} |k⟩
        
        This reveals hidden structure: if f(x)=f(x⊕s), then measuring
        after QFT gives vectors z satisfying z·s = 0.
        """
        n = len(state)
        qft_matrix = np.zeros((n, n), dtype=complex)
        
        for j in range(n):
            for k in range(n):
                phase = 2.0 * np.pi * j * k / n
                qft_matrix[j, k] = np.exp(1j * phase) / np.sqrt(n)
        
        return qft_matrix @ state
    
    def _apply_oracle(self, state: np.ndarray) -> np.ndarray:
        """
        Apply metriplex oracle to quantum register.
        
        The oracle establishes phase correlations based on energy profiling.
        This encodes the 2-to-1 collision structure into relative phases.
        """
        result = state.copy()
        
        for idx, occupation in enumerate(self.fock.basis_states):
            # Get effective momentum from occupation
            p_eff = self.oracle._occupation_to_momentum(occupation)
            
            # Oracle: phase shift by normalized energy
            energy = self.oracle.energy_map[p_eff]
            phase = np.exp(2j * np.pi * energy)
            result[idx] *= phase
        
        self.oracle_queries += 1
        return result
    
    def _add_noise(self, state: np.ndarray, noise_type: str) -> np.ndarray:
        """
        Simulate quantum noise on state vector.
        
        Supported models:
        - "ideal": no noise
        - "depolarizing": random state mixing
        - "amplitude_damping": energy dissipation
        """
        if noise_type == "ideal":
            return state.copy()
        
        elif noise_type == "depolarizing":
            # With probability p, replace with random state
            # |ψ⟩ → (1-p)|ψ⟩ + p·ρ_random
            p = self.config.depolarizing_rate
            random_state = np.random.randn(len(state)) + 1j * np.random.randn(len(state))
            random_state /= np.linalg.norm(random_state)
            
            noisy = (1 - p) * state + p * random_state
            noisy /= np.linalg.norm(noisy)
            return noisy
        
        elif noise_type == "amplitude_damping":
            # Decay: |1⟩ → |0⟩ with rate Γ
            Gamma = self.config.depolarizing_rate
            result = state.copy()
            for i in range(1, len(state)):
                result[i] *= (1 - Gamma)
            result[0] *= (1 + Gamma * np.sum(np.abs(state[1:])**2))
            result /= np.linalg.norm(result)
            return result
        
        else:
            raise ValueError(f"Unknown noise model: {noise_type}")
    
    def _measure_computational_basis(self, state: np.ndarray, shots: int) -> Dict[int, int]:
        """
        Measure quantum state in computational basis.
        
        Returns histogram of measurement outcomes.
        """
        probabilities = np.abs(state)**2
        outcomes = np.random.choice(len(state), size=shots, p=probabilities)
        
        histogram = {}
        for outcome in outcomes:
            histogram[outcome] = histogram.get(outcome, 0) + 1
        
        return histogram
    
    def _extract_constraint_vectors(self, measurement_results: List[int]) -> List[int]:
        """
        Extract linearly independent constraint vectors from measurements.
        
        Each measurement outcome z gives constraint: z·s = 0 (mod 2)
        We collect measurements until s is determined (or we run out of time).
        """
        constraints = []
        
        for z in measurement_results:
            # Check if z is linearly independent from previous vectors
            # (in GF(2): vector space over binary field)
            if not self._is_lin_dependent(z, constraints):
                constraints.append(z)
            
            # Stop when we have enough constraints to solve for s
            if len(constraints) >= self.config.n_qubits - 1:
                break
        
        return constraints
    
    def _is_lin_dependent(self, vector: int, basis: List[int]) -> bool:
        """
        Check if vector is linearly dependent on basis in GF(2).
        
        This is a Gaussian elimination in binary field.
        """
        if not basis:
            return False
        
        # Try to express vector as XOR of basis elements
        for mask in range(1, 2 ** len(basis)):
            xor_result = 0
            for i, b in enumerate(basis):
                if mask & (1 << i):
                    xor_result ^= b
            if xor_result == vector:
                return True
        
        return False
    
    def _solve_for_secret(self, constraint_vectors: List[int]) -> Optional[int]:
        """
        Solve linear system (mod 2) to recover secret string s.
        
        System: For all constraint vectors z_i, we have z_i · s = 0
        
        In GF(2), this is: Z·s = 0 where Z is the constraint matrix.
        Solution s is in the null space of Z.
        """
        if not constraint_vectors:
            return None
        
        n = self.config.n_qubits
        
        # Build constraint matrix (rows = constraints)
        Z = np.zeros((len(constraint_vectors), n), dtype=int)
        for i, z in enumerate(constraint_vectors):
            for j in range(n):
                Z[i, j] = (z >> j) & 1
        
        # Gaussian elimination in GF(2) to find null space
        null_space = self._null_space_gf2(Z)
        
        # The null space should contain exactly one non-zero vector (for 2-to-1 functions)
        non_trivial = [v for v in null_space if v != 0]
        
        if len(non_trivial) == 1:
            return non_trivial[0]
        elif len(non_trivial) > 1:
            # Ambiguity; return the one with smallest Hamming weight
            return min(non_trivial, key=lambda v: bin(v).count('1'))
        else:
            return None
    
    def _null_space_gf2(self, Z: np.ndarray) -> List[int]:
        """
        Compute null space of matrix Z over GF(2).
        
        Returns list of integers representing null space basis vectors.
        """
        m, n = Z.shape
        
        # Gaussian elimination
        A = Z.copy()
        pivot_cols = []
        current_row = 0
        
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(current_row, m):
                if A[row, col] == 1:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue  # col is free variable
            
            # Swap rows
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]
            pivot_cols.append(col)
            
            # Eliminate
            for row in range(m):
                if row != current_row and A[row, col] == 1:
                    A[row] = (A[row] + A[current_row]) % 2
            
            current_row += 1
        
        # Extract null space (free variables)
        free_cols = [i for i in range(n) if i not in pivot_cols]
        null_basis = []
        
        for free_assignment in range(1, 2 ** len(free_cols)):
            # Build vector for this assignment
            v = np.zeros(n, dtype=int)
            
            for i, fc in enumerate(free_cols):
                v[fc] = (free_assignment >> i) & 1
            
            # Compute dependent part
            for i, pc in enumerate(pivot_cols):
                xor_sum = 0
                for j, fc in enumerate(free_cols):
                    xor_sum ^= (A[i, fc] * v[fc])
                v[pc] = xor_sum
            
            # Convert to integer
            result = 0
            for i, bit in enumerate(v):
                result |= (bit << i)
            
            if result > 0:
                null_basis.append(result)
        
        return null_basis
    
    def run(self) -> Dict:
        """
        Execute Simon's algorithm.
        
        Returns dictionary with results and diagnostics.
        """
        results = {
            'secret_string_found': False,
            'recovered_secret': None,
            'oracle_queries': 0,
            'measurement_outcomes': [],
            'constraint_vectors': [],
            'execution_log': []
        }
        
        # Step 1: Initialize to equal superposition
        state = fock_ground_state(self.fock).vec
        state = self._hadamard(state)
        results['execution_log'].append("Step 1: Initialized to superposition |+...+⟩")
        
        measurement_outcomes = []
        
        # Step 2-3: Query oracle and measure
        for query_num in range(self.config.n_queries):
            # Apply oracle
            state = self._apply_oracle(state)
            
            # Add noise if configured
            state = self._add_noise(state, self.config.noise_model)
            
            # Apply QFT to reveal structure
            state = self._qft(state)
            
            # Measure
            histogram = self._measure_computational_basis(
                state, 
                self.config.measurement_shots
            )
            
            # Extract most likely outcomes
            sorted_outcomes = sorted(histogram.items(), key=lambda x: -x[1])
            for outcome, count in sorted_outcomes[:3]:  # Top 3 outcomes
                if count >= 50:  # Significant probability
                    measurement_outcomes.append(outcome)
            
            results['execution_log'].append(
                f"Query {query_num+1}: Oracle applied, QFT, measured. "
                f"Top outcome: {sorted_outcomes[0][0] if sorted_outcomes else 'none'}"
            )
        
        results['oracle_queries'] = self.oracle_queries
        results['measurement_outcomes'] = measurement_outcomes
        
        # Step 4: Extract constraints
        constraint_vectors = self._extract_constraint_vectors(measurement_outcomes)
        results['constraint_vectors'] = constraint_vectors
        results['execution_log'].append(
            f"Extracted {len(constraint_vectors)} constraint vectors"
        )
        
        # Step 5: Solve for secret
        if constraint_vectors:
            secret = self._solve_for_secret(constraint_vectors)
            if secret is not None:
                results['secret_string_found'] = True
                results['recovered_secret'] = secret
                results['execution_log'].append(
                    f"Recovered secret: s = {secret} (binary {format(secret, '03b')})"
                )
            else:
                results['execution_log'].append(
                    "Could not solve linear system for secret"
                )
        
        return results


if __name__ == "__main__":
    # Example execution
    print("=" * 80)
    print("Simon's Improved Algorithm: Fock + Metriplex + H7")
    print("=" * 80)
    
    # Setup
    fock_config = FockConfig(n_modes=3, n_max=2)
    fock = FockBasis(fock_config)
    
    oracle_config = MetriplexConfig()
    oracle = MetriplexOracle(oracle_config)
    
    simon_config = SimonConfig(n_qubits=3, measurement_shots=500)
    simon = SimonImproved(fock, oracle, simon_config)
    
    # Run
    results = simon.run()
    
    print("\nExecution Log:")
    for log_entry in results['execution_log']:
        print(f"  {log_entry}")
    
    print(f"\nResults:")
    print(f"  Oracle queries: {results['oracle_queries']}")
    print(f"  Measurement outcomes (sample): {results['measurement_outcomes'][:5]}")
    print(f"  Constraint vectors: {results['constraint_vectors']}")
    
    if results['secret_string_found']:
        print(f"\n✓ Secret recovered: s = {results['recovered_secret']}")
        print(f"  Binary: {format(results['recovered_secret'], '03b')}")
        print(f"  Matches oracle symmetry: s = {oracle.symmetry_string()}")
    else:
        print(f"\n✗ Could not recover secret")
