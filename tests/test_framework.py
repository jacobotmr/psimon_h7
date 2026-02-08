import pytest
import os
from core.psimon_framework import PSimon, PSimonConfig, FrameworkMode

def test_framework_initialization():
    framework = PSimon()
    assert framework.config is not None
    assert framework.fock is not None
    assert framework.oracle is not None
    assert framework.simon is not None

def test_framework_run_search():
    config = PSimonConfig(mode=FrameworkMode.CLASSICAL_SIMULATION)
    framework = PSimon(config)
    result = framework.run_simon_search()
    
    assert 'recovered_secret' in result
    assert 'oracle_queries' in result
    assert len(framework.execution_history) == 1

def test_framework_nucleon_system():
    framework = PSimon()
    nucleons = framework.construct_nucleon_system()
    assert 'uuu' in nucleons
    assert 'ddd' in nucleons
    assert len(nucleons) >= 4

def test_framework_export_config():
    framework = PSimon()
    config_json = framework.export_config_json()
    assert 'fock' in config_json
    assert 'metriplex' in config_json or 'oracle' in config_json
    assert 'simon' in config_json
