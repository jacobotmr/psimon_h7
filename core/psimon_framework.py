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
    """Execution modes for the PSimon Processor."""
    TOPOLOGICAL_PROCESSOR = "topological_processor"  # Ideal topological evolution
    STOCHASTIC_POTENTIAL = "stochastic_potential"    # Noisy simulation
    HARDWARE_READY = "hardware_ready"                # Generate hardware instructions


@dataclass
class PSimonConfig:
    """Master configuration for PSimon framework."""
    
    # Quantum resources (Default: 3-bit H7 Topological Processor)
    fock_config: FockConfig = field(default_factory=lambda: FockConfig(n_modes=3, n_max=1, use_gray_code=False))
    
    # Oracle/metriplex
    metriplex_config: MetriplexConfig = field(default_factory=MetriplexConfig)
    
    # Simon's algorithm (Default: 3-qubit H7 Search)
    simon_config: SimonConfig = field(default_factory=lambda: SimonConfig(n_qubits=3, gray_code_enabled=False))
    
    # Framework-level
    mode: FrameworkMode = FrameworkMode.TOPOLOGICAL_PROCESSOR
    verbose: bool = True
    random_seed: Optional[int] = None


class PSimon:
    """
    Unified PSimon Topological Processor.
    
    Integrates:
    1. Fock space foundations for potential fields
    2. Metriplex topological oracle with cyclic energy wrapping
    3. H7 symmetry processor (Simon's algorithm)
    4. H7 entanglement conservation laws (s = 7)
    
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
        Construct nucleon system mapping quarks to segmented Fock states.
        
        H7 Processor Mapping (Wrapping 1-6, 2-5, 3-4):
        - uud/ddu → |1,1,0⟩ + |0,0,1⟩ (H7 pair 6-1)
        - ddu/uud → |0,1,1⟩ + |1,0,0⟩ (H7 pair 3-4)
        - udu/dud → |1,0,1⟩ + |0,1,0⟩ (H7 pair 5-2)
        - uuu/ddd → |1,1,1⟩ + |0,0,0⟩ (H7 pair 7-0, Truncated)
        
        Returns state vectors as topological potentials (H7 pairs).
        """
        self._log("Constructing nucleon system in H7 Processor")
        
        result = {}
        
        # Base moments
        states = {
            'uud': (1, 1, 0), # 6
            'ddu': (0, 1, 1), # 3
            'udu': (1, 0, 1), # 5
            'dud': (0, 1, 0), # 2
            'uuu': (1, 1, 1), # 7
            'ddd': (0, 0, 0)  # 0
        }
        
        # Helper to create H7 pairs (Topological Entanglement)
        def h7_pair(occ):
            v1 = self.fock.state_vector(occ)
            idx1 = self.fock.state_to_index[occ]
            idx2 = 7 ^ idx1
            occ2 = self.fock.index_to_state[idx2]
            v2 = self.fock.state_vector(occ2)
            return (v1 + v2) / np.sqrt(2)

        for name, occupation in states.items():
            try:
                result[name] = h7_pair(occupation)
            except (ValueError, KeyError):
                self._log(f"Warning: could not construct {name}")
        
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
        
        H7 Processor Mapping: The Fock basis index (for n_max=1)
        directly corresponds to the 3-bit computational basis.
        """
        if len(fock_state) == 8:
            return fock_state.copy() # Direct mapping for H7 Processor

        if len(fock_state) != self.fock.dim:
            raise ValueError(f"State dimension mismatch")
        
        qubit_state = np.zeros(8, dtype=complex)
        
        for idx, val in enumerate(fock_state):
            if abs(val) < 1e-10: continue
            occupation = self.fock.basis_states[idx]
            # Map occupation bitstring to integer value
            qubit_idx = 0
            for bit in occupation:
                qubit_idx = (qubit_idx << 1) | (int(bit) & 1)
            qubit_state[qubit_idx % 8] += val
        
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


def create_stochastic_framework(n_qubits: int = 3) -> PSimon:
    """Create processor with stochastic potential modulation."""
    config = PSimonConfig(
        fock_config=FockConfig(n_modes=n_qubits, n_max=1, use_gray_code=False),
        simon_config=SimonConfig(n_qubits=n_qubits, noise_model='depolarizing', gray_code_enabled=False),
        mode=FrameworkMode.STOCHASTIC_POTENTIAL
    )
    return PSimon(config)


def create_nucleon_explorer() -> PSimon:
    """Create framework specialized for H7 nucleon/quark systems."""
    config = PSimonConfig(
        fock_config=FockConfig(n_modes=3, n_max=1, use_gray_code=False),
        simon_config=SimonConfig(n_qubits=3, measurement_shots=2000, gray_code_enabled=False),
        mode=FrameworkMode.TOPOLOGICAL_PROCESSOR
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
