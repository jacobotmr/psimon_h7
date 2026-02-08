"""
PSimon Framework: Unified Second Quantization + Simon's Algorithm
Complete Integration of Fock Space, Metriplex Dynamics, and H7 Conservation

Framework Layer: High-level API for constructing quantum-classical hybrid systems
that exploit second quantization and hidden symmetry discovery.

Author: Jacobo Tlacaelel Mina Rodríguez
License: MIT
Version: 0.1.0 (Beta)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import warnings

from physics.fock_basis import FockBasis, FockConfig, FockStateVector, fock_ground_state, fock_single_photon
from physics.metriplex_oracle import MetriplexOracle, MetriplexConfig, H7Conservation
from quantum.simon_improved import SimonImproved, SimonConfig


class FrameworkMode(Enum):
    """Execution modes for the framework."""
    CLASSICAL_SIMULATION = "classical_simulation"  # Full classical simulation (deterministic)
    NISQ_SIMULATION = "nisq_simulation"           # Noisy simulation (realistic)
    HARDWARE_READY = "hardware_ready"             # Generate hardware instructions


@dataclass
class PSimonConfig:
    """Master configuration for PSimon framework."""
    
    # Quantum resources
    fock_config: FockConfig = field(default_factory=lambda: FockConfig(n_modes=3, n_max=3))
    
    # Oracle/metriplex
    metriplex_config: MetriplexConfig = field(default_factory=MetriplexConfig)
    
    # Simon's algorithm
    simon_config: SimonConfig = field(default_factory=lambda: SimonConfig(n_qubits=3))
    
    # Framework-level
    mode: FrameworkMode = FrameworkMode.CLASSICAL_SIMULATION
    verbose: bool = True
    random_seed: Optional[int] = None


class PSimon:
    """
    Unified framework for second quantization quantum computing.
    
    Integrates:
    1. Fock space with configurable truncation
    2. Metriplex momentum oracle with energy profiling
    3. Simon's algorithm for symmetry discovery
    4. H7 entanglement conservation laws
    
    Usage pattern:
        framework = PSimon(config)
        result = framework.run_simon_search()
        framework.analyze_results(result)
    """
    
    def __init__(self, config: PSimonConfig = None):
        """Initialize PSimon framework with configuration."""
        if config is None:
            config = PSimonConfig()
        
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Build quantum resources
        self.fock = FockBasis(config.fock_config)
        self.oracle = MetriplexOracle(config.metriplex_config)
        self.simon = SimonImproved(self.fock, self.oracle, config.simon_config)
        
        # Diagnostics
        self.execution_history: List[Dict] = []
        self._log("Framework initialized")
    
    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.config.verbose:
            print(f"[PSimon] {message}")
    
    def run_simon_search(self) -> Dict[str, Any]:
        """
        Execute Simon's algorithm to discover hidden symmetries.
        
        Returns comprehensive result dictionary with:
        - Recovered secret string
        - Oracle query statistics
        - Measurement outcomes
        - Convergence diagnostics
        """
        self._log(f"Starting Simon's search in mode: {self.config.mode.value}")
        
        # Run Simon
        result = self.simon.run()
        
        # Enrich result with framework diagnostics
        result['framework_config'] = {
            'fock_dimension': self.fock.dim,
            'oracle_momentum_range': self.oracle.config.momentum_range,
            'execution_mode': self.config.mode.value
        }
        
        # Store in history
        self.execution_history.append(result)
        
        return result
    
    def construct_nucleon_system(self) -> Dict[str, Any]:
        """
        Construct fermion-boson system mapping nucleons to Fock states.
        
        Maps:
        - uuu (up triplet) → |3,0,0⟩ (3 excitations in mode 0)
        - ddd (down triplet) → |0,3,0⟩ (3 excitations in mode 1)
        - udu → symmetric superposition of modes 0,1
        - dud → antisymmetric superposition of modes 0,1
        
        Returns state vectors and their relationships under H7 conservation.
        """
        self._log("Constructing nucleon system")
        
        result = {}
        
        # Single-quark basis states (per mode)
        state_u = [3, 0, 0]  # Up triplet: 3 bosonic excitations
        state_d = [0, 3, 0]  # Down triplet: 3 bosonic excitations
        state_mixed = [1, 1, 1]  # Mixed: 1 per mode
        
        # Construct state vectors
        result['uuu'] = self.fock.state_vector(tuple(state_u))
        result['ddd'] = self.fock.state_vector(tuple(state_d))
        result['udu'] = self._create_superposition([state_u, state_d], [1, 1])
        result['dud'] = self._create_superposition([state_u, state_d], [1, -1])
        result['mixed'] = self.fock.state_vector(tuple(state_mixed))
        
        # Check H7 conservation for each
        h7_info = {}
        for name, vec in result.items():
            # Map to 3-qubit representation
            qubit_vec = self._fock_to_qubit(vec)
            conserved = H7Conservation.verify_conservation_invariant(qubit_vec)
            h7_info[name] = {
                'h7_conserved': conserved,
                'qubit_representation': qubit_vec
            }
        
        result['h7_conservation'] = h7_info
        
        self._log(f"Nucleon system constructed: {list(result.keys())}")
        return result
    
    def _create_superposition(self, states_list: List[Tuple[int, ...]], 
                            coefficients: List[float]) -> np.ndarray:
        """Create coherent superposition of Fock states."""
        vec = np.zeros(self.fock.dim, dtype=complex)
        
        for state_tuple, coeff in zip(states_list, coefficients):
            state_vec = self.fock.state_vector(tuple(state_tuple))
            vec += coeff * state_vec
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        
        return vec
    
    def _fock_to_qubit(self, fock_state: np.ndarray) -> np.ndarray:
        """
        Map Fock state to 3-qubit computational basis.
        
        This is a projection: we take maximum occupation number in any mode
        and encode as binary.
        """
        if len(fock_state) != self.fock.dim:
            raise ValueError(f"State dimension mismatch: {len(fock_state)} != {self.fock.dim}")
        
        # Extract measurement basis probabilities
        probs = np.abs(fock_state)**2
        
        # For each basis state, compute its "qubit signature"
        qubit_state = np.zeros(8, dtype=complex)  # 3-qubit = 8 computational states
        
        for idx, prob in enumerate(probs):
            occupation = self.fock.basis_states[idx]
            # Map occupation to qubit index via total occupation
            qubit_idx = sum(occupation) % 8
            qubit_state[qubit_idx] += prob * fock_state[idx]
        
        return qubit_state
    
    def verify_h7_conservation(self) -> Dict[str, bool]:
        """
        Verify that quantum evolution preserves H7 conservation law.
        
        Tests whether entanglement pairing |x⟩ ↔ |7⊕x⟩ is maintained
        under the oracle and Simon's operations.
        """
        self._log("Verifying H7 conservation invariant")
        
        results = {}
        
        # Test ground state
        gs = fock_ground_state(self.fock)
        qubit_gs = self._fock_to_qubit(gs.vec)
        results['ground_state'] = H7Conservation.verify_conservation_invariant(qubit_gs)
        
        # Test superposition states
        nucleon_system = self.construct_nucleon_system()
        for name, h7_info in nucleon_system['h7_conservation'].items():
            results[name] = h7_info['h7_conserved']
        
        # Summary
        all_conserved = all(results.values())
        self._log(f"H7 conservation: {sum(results.values())}/{len(results)} states conserved")
        
        return results
    
    def analyze_results(self, result: Dict) -> str:
        """
        Human-readable analysis of Simon's algorithm results.
        
        Returns formatted report.
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("SIMON'S ALGORITHM: RESULTS ANALYSIS")
        report.append("=" * 80)
        
        # Execution summary
        report.append(f"\nOracle Queries: {result.get('oracle_queries', '?')}")
        report.append(f"Secret Found: {'✓' if result.get('secret_string_found') else '✗'}")
        
        if result.get('recovered_secret') is not None:
            secret = result['recovered_secret']
            oracle_secret = self.oracle.symmetry_string()
            report.append(f"Recovered Secret: s = {secret} (binary {format(secret, '03b')})")
            report.append(f"Oracle Secret: s = {oracle_secret} (binary {format(oracle_secret, '03b')})")
            
            if secret == oracle_secret:
                report.append(f"✓ MATCH: Algorithm correctly discovered hidden structure")
            else:
                report.append(f"✗ MISMATCH: Algorithm did not match oracle secret")
        
        # Constraints
        if result.get('constraint_vectors'):
            report.append(f"\nConstraint Vectors: {len(result['constraint_vectors'])}")
            for i, cv in enumerate(result['constraint_vectors']):
                report.append(f"  z_{i} = {cv} (binary {format(cv, '03b')})")
        
        # Execution log
        if result.get('execution_log'):
            report.append(f"\nExecution Log:")
            for log_entry in result['execution_log']:
                report.append(f"  {log_entry}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def export_config_json(self) -> str:
        """Export framework configuration as JSON."""
        fock_dict = asdict(self.config.fock_config)
        fock_dict['mode_type'] = fock_dict['mode_type'].value
        
        simon_dict = asdict(self.config.simon_config)
        simon_dict.pop('truncation_monitor', None)
        
        config_dict = {
            'fock': fock_dict,
            'metriplex': {
                'momentum_range': self.config.metriplex_config.momentum_range,
                'energy_profile': self.config.metriplex_config.energy_profile.value,
                'collision_groups': self.config.metriplex_config.collision_groups
            },
            'simon': simon_dict,
            'framework_mode': self.config.mode.value
        }
        return json.dumps(config_dict, indent=2)
    
    def get_framework_info(self) -> Dict:
        """Get comprehensive framework information."""
        return {
            'version': '0.1.0',
            'mode': self.config.mode.value,
            'quantum_resources': {
                'fock_dimension': self.fock.dim,
                'n_modes': self.fock.n_modes,
                'max_occupation_per_mode': self.fock.n_max
            },
            'oracle_config': {
                'momentum_range': self.oracle.config.momentum_range,
                'hidden_symmetry_s': self.oracle.symmetry_string(),
                'collision_groups': self.oracle.config.collision_groups
            },
            'h7_conservation': {
                'conservation_constant': H7Conservation.CONSERVATION_CONSTANT,
                'pairing_table': H7Conservation.pairing_table()
            },
            'execution_history_length': len(self.execution_history)
        }


# Convenience factory functions
def create_default_framework() -> PSimon:
    """Create framework with sensible defaults."""
    return PSimon()


def create_hardware_optimized_framework(n_qubits: int = 5) -> PSimon:
    """Create framework optimized for NISQ hardware constraints."""
    config = PSimonConfig(
        fock_config=FockConfig(n_modes=min(3, n_qubits//2), n_max=2),
        simon_config=SimonConfig(n_qubits=min(n_qubits, 8), noise_model='depolarizing'),
        mode=FrameworkMode.NISQ_SIMULATION
    )
    return PSimon(config)


def create_nucleon_explorer() -> PSimon:
    """Create framework specialized for nucleon/quark systems."""
    config = PSimonConfig(
        fock_config=FockConfig(n_modes=3, n_max=3),
        simon_config=SimonConfig(n_qubits=3, measurement_shots=2000),
        mode=FrameworkMode.CLASSICAL_SIMULATION
    )
    return PSimon(config)


if __name__ == "__main__":
    print("PSimon Framework: Demonstration")
    print("=" * 80)
    
    # Create framework
    framework = create_default_framework()
    print(framework.export_config_json())
    
    print("\n" + "=" * 80)
    print("Running Simon's Algorithm")
    print("=" * 80)
    
    result = framework.run_simon_search()
    print(framework.analyze_results(result))
    
    print("\n" + "=" * 80)
    print("Constructing Nucleon System")
    print("=" * 80)
    
    nucleons = framework.construct_nucleon_system()
    print(f"Nucleon states: {list(nucleons.keys())}")
    print(f"H7 Conservation Check:")
    for name, info in nucleons['h7_conservation'].items():
        print(f"  {name}: {'✓' if info['h7_conserved'] else '✗'}")
    
    print("\n" + "=" * 80)
    print("Framework Information")
    print("=" * 80)
    
    info = framework.get_framework_info()
    for key, value in info.items():
        print(f"{key}: {value}")
