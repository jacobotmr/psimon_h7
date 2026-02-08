import pytest
import numpy as np
from physics.metriplex_oracle import MetriplexOracle, MetriplexConfig, EnergyProfile

def test_oracle_initialization():
    config = MetriplexConfig(momentum_range=(1, 4))
    oracle = MetriplexOracle(config)
    assert oracle.p_min == 1
    assert oracle.p_max == 4
    assert len(oracle.energy_map) == 4

def test_oracle_forward_mapping():
    config = MetriplexConfig(momentum_range=(1, 3))
    oracle = MetriplexOracle(config)
    
    group, output, energy = oracle.forward(1)
    assert group in ['A', 'B']
    assert isinstance(output, np.ndarray)
    assert 0 <= energy <= 1.0

def test_oracle_out_of_range():
    oracle = MetriplexOracle()
    with pytest.raises(ValueError):
        oracle.forward(100)

def test_oracle_hidden_string():
    oracle = MetriplexOracle()
    s = oracle.symmetry_string()
    # In Metriplex oracle, p and p XOR s should belong to the same collision group
    for p in range(oracle.p_min, oracle.p_max + 1):
        p_partner = p ^ s
        if oracle.p_min <= p_partner <= oracle.p_max:
            group1, _, _ = oracle.forward(p)
            group2, _, _ = oracle.forward(p_partner)
            assert group1 == group2
