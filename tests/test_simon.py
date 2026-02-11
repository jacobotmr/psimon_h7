import pytest
import numpy as np
from physics.fock_basis import FockBasis, FockConfig
from physics.metriplex_oracle import MetriplexOracle, MetriplexConfig
from quantum.simon_improved import SimonImproved, SimonConfig

def test_simon_initialization():
    fock = FockBasis(FockConfig(n_modes=2))
    oracle = MetriplexOracle(MetriplexConfig(momentum_range=(1, 4)))
    config = SimonConfig(n_qubits=2)
    simon = SimonImproved(fock, oracle, config)
    assert simon.config.n_qubits == 2

def test_simon_hadamard():
    fock = FockBasis(FockConfig(n_modes=1)) # 1 qubit = 2 states
    oracle = MetriplexOracle()
    simon = SimonImproved(fock, oracle, SimonConfig(n_qubits=1))
    
    # Ground state |0>
    state = np.zeros(2)
    state[0] = 1.0
    
    # Apply Hadamard: |0> -> (|0> + |1>) / sqrt(2)
    h_state = simon._hadamard(state)
    assert np.abs(h_state[0] - 1.0/np.sqrt(2)) < 1e-9
    assert np.abs(h_state[1] - 1.0/np.sqrt(2)) < 1e-9

def test_simon_run_ideal():
    # Small system for fast testing
    fock = FockBasis(FockConfig(n_modes=3, n_max=1, use_gray_code=False))
    oracle = MetriplexOracle(MetriplexConfig(momentum_range=(0, 7)))
    config = SimonConfig(n_qubits=3, n_queries=5, noise_model="ideal", gray_code_enabled=False)
    simon = SimonImproved(fock, oracle, config)
    
    result = simon.run()
    assert result['oracle_queries'] > 0
    # For a deterministic oracle in ideal conditions, it should find the secret
    assert result['secret_string_found'] == True
    assert result['recovered_secret'] == oracle.symmetry_string()
