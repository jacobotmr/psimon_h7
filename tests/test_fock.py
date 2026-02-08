import pytest
import numpy as np
from physics.fock_basis import FockBasis, FockConfig, OccupationMode

def test_fock_initialization():
    config = FockConfig(n_modes=2, n_max=2, mode_type=OccupationMode.BOSONIC)
    fock = FockBasis(config)
    assert fock.dim == 9  # (2+1)^2
    assert len(fock.basis_states) == 9

def test_fock_operators():
    config = FockConfig(n_modes=1, n_max=5)
    fock = FockBasis(config)
    
    # Creation operator
    a_dag = fock.get_creation_op(0)
    state_0 = np.zeros(6)
    state_0[0] = 1.0
    
    state_1 = a_dag @ state_0
    assert np.abs(state_1[1] - 1.0) < 1e-9
    
    # Annihilation operator
    a = fock.get_annihilation_op(0)
    state_back_0 = a @ state_1
    assert np.abs(state_back_0[0] - 1.0) < 1e-9

def test_fock_number_operator():
    config = FockConfig(n_modes=1, n_max=5)
    fock = FockBasis(config)
    n_op = fock.number_operator(0)
    
    for n in range(6):
        state = np.zeros(6)
        state[n] = 1.0
        result = n_op @ state
        assert np.abs(result[n] - n) < 1e-9

def test_gray_code_mapping():
    config = FockConfig(n_modes=2, n_max=3, use_gray_code=True)
    fock = FockBasis(config)
    
    # 3 in binary is 011, Gray code of 3 is 010 (which is 2)
    for i in range(fock.dim):
        gray = fock.to_gray_code(i)
        binary = fock.from_gray_code(gray)
        assert binary == i
