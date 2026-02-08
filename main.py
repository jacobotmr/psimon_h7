"""
PSimon Framework: Unified Entry Point
================================================================================
This script serves as the central command-line interface for the PSimon Framework,
integrating Fock space foundations, Metriplex dynamics, and Simon's Algorithm.

Author: Jacobo Tlacaelel Mina Rodríguez
Version: 0.1.1
"""

import sys
import argparse
import time
from core.psimon_framework import PSimon, PSimonConfig, FrameworkMode
from physics.isotope_h7_convergence import IsotopeToH7Convergence
from physics.chiral_fermionic_system import ChiralEncoder, ChiralAnalyzer, demonstrate_chiral_system
from physics.isotopic_beta_fermion_system import demonstrate_system as demonstrate_beta_decay
from models.algolia_indexer import demonstrate_algolia
from models.cognitive_engine import demonstrate_cognitive_system

def run_framework_demo():
    print("\n" + "=" * 80)
    print("PSimon Framework: Core Demonstration")
    print("=" * 80)
    
    start_time = time.time()
    config = PSimonConfig(mode=FrameworkMode.CLASSICAL_SIMULATION)
    framework = PSimon(config)
    
    print("\n[Executing Simon's search...]")
    result = framework.run_simon_search()
    print(framework.analyze_results(result))
    
    print("\n[Nuclear System Analysis...]")
    nucleons = framework.construct_nucleon_system()
    print(f"Nucleon states constructed: {list(nucleons.keys())}")
    
    elapsed = time.time() - start_time
    print(f"\n[Verification Complete in {elapsed:.4f}s]")

def run_convergence_demo():
    print("\n" + "=" * 80)
    print("PSimon: Isotope to H7 Convergence Analysis")
    print("=" * 80)
    start_time = time.time()
    convergence = IsotopeToH7Convergence()
    convergence.converge_all()
    print(convergence.export_json()[:1000] + "...\n[Report truncated, see data/isotope_h7_convergence.json]")
    elapsed = time.time() - start_time
    print(f"\n[Convergence Analysis Complete in {elapsed:.4f}s]")

def main():
    parser = argparse.ArgumentParser(description="PSimon Framework CLI")
    parser.add_argument("--demo", action="store_true", help="Run the core PSimon demonstration")
    parser.add_argument("--convergence", action="store_true", help="Run Isotope to H7 convergence analysis")
    parser.add_argument("--beta", action="store_true", help="Run Beta Decay simulation demo")
    parser.add_argument("--chiral", action="store_true", help="Run Chiral Fermionic system demo")
    parser.add_argument("--search", action="store_true", help="Run Algolia search (7th Art) demo")
    parser.add_argument("--cognitive", action="store_true", help="Run Cognitive Metriplectic Engine demo")
    parser.add_argument("--gui", action="store_true", help="Launch Streamlit Dashboard")
    parser.add_argument("--all", action="store_true", help="Run all available demonstrations")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    total_start = time.time()
    
    if args.demo or args.all:
        run_framework_demo()
    
    if args.convergence or args.all:
        run_convergence_demo()
        
    if args.beta or args.all:
        start_beta = time.time()
        demonstrate_beta_decay()
        print(f"\n[Beta Decay Simulation Complete in {time.time() - start_beta:.4f}s]")
        
    if args.chiral or args.all:
        start_chiral = time.time()
        demonstrate_chiral_system()
        print(f"\n[Chiral System Analysis Complete in {time.time() - start_chiral:.4f}s]")

    if args.search or args.all:
        demonstrate_algolia()

    if args.cognitive or args.all:
        demonstrate_cognitive_system()

    if args.gui:
        import os
        print("\n" + "=" * 80)
        print("LAUNCHING STREAMLIT DASHBOARD...")
        print("=" * 80)
        # Use the absolute path to the venv streamlit
        venv_path = os.path.join(os.getcwd(), ".venv", "bin", "streamlit")
        os.system(f"{venv_path} run frontend.py")

    if args.demo or args.convergence or args.beta or args.chiral or args.search or args.cognitive or args.all:
        print("\n" + "=" * 80)
        print(f"TOTAL EXECUTION TIME: {time.time() - total_start:.4f}s")
        print("=" * 80)

if __name__ == "__main__":
    main()
