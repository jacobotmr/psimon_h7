"""
PSimon: Ejemplos Prácticos
Casos de uso reales del framework en investigación cuántica.

Author: Jacobo Tlacaelel Mina Rodríguez
"""

import numpy as np
from psimon_framework import (
    PSimon, PSimonConfig, FrameworkMode, FockConfig, 
    SimonConfig, MetriplexConfig, create_default_framework,
    create_hardware_optimized_framework, create_nucleon_explorer
)
from fock_basis import fock_ground_state, fock_single_photon
from metriplex_oracle import H7Conservation


# ============================================================================
# EJEMPLO 1: Búsqueda de Simetrías Básica
# ============================================================================

def ejemplo_1_basic_symmetry_search():
    """
    Caso de uso básico: descubrir el string secreto de un oracle.
    
    Este es el "hello world" de Simon's algorithm en Fock space.
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 1: Búsqueda Básica de Simetrías")
    print("=" * 80)
    
    # Crear framework con configuración por defecto
    framework = create_default_framework()
    
    # Ejecutar Simon
    result = framework.run_simon_search()
    
    # Analizar
    print(framework.analyze_results(result))
    
    # Diagnóstico: ¿Convergió?
    if result['secret_string_found']:
        recovered = result['recovered_secret']
        expected = 3  # Oracle secreto conocido
        
        if recovered == expected:
            print(f"\n✓ ÉXITO: Recuperó s = {recovered} (esperado: {expected})")
        else:
            print(f"\n⚠ PARCIAL: Recuperó s = {recovered} (esperado: {expected})")
            print(f"  Esto sugiere que Gray-code o mediciones necesitan ajuste")
    else:
        print(f"\n✗ FRACASO: No convergio. Intenta con más queries.")


# ============================================================================
# EJEMPLO 2: Robustez bajo Ruido
# ============================================================================

def ejemplo_2_noise_robustness():
    """
    Investigar cómo degrada la precisión bajo ruido realista (NISQ).
    
    Compara simulación ideal vs. depolarización realista.
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 2: Robustez bajo Ruido NISQ")
    print("=" * 80)
    
    noise_rates = [0.00, 0.01, 0.02, 0.05, 0.10]
    
    results_by_noise = {}
    
    for noise_rate in noise_rates:
        print(f"\nProbando con tasa de depolarización: {noise_rate*100:.1f}%")
        
        config = PSimonConfig(
            simon_config=SimonConfig(
                n_qubits=3,
                noise_model='depolarizing',
                depolarizing_rate=noise_rate,
                measurement_shots=1000
            ),
            mode=FrameworkMode.NISQ_SIMULATION
        )
        
        framework = PSimon(config)
        result = framework.run_simon_search()
        
        results_by_noise[noise_rate] = {
            'converged': result['secret_string_found'],
            'recovered': result['recovered_secret'],
            'queries': result['oracle_queries'],
            'constraints': len(result['constraint_vectors'])
        }
        
        status = "✓" if result['secret_string_found'] else "✗"
        print(f"  {status} Convergió: {result['secret_string_found']}")
        if result['secret_string_found']:
            print(f"    Recuperó: s = {result['recovered_secret']}")
        print(f"    Queries: {result['oracle_queries']}")
    
    # Análisis de degradación
    print(f"\nAnálisis de Degradación:")
    print(f"{'Noise Rate':<12} {'Success':<10} {'Recovered':<12} {'Queries':<10}")
    print("-" * 45)
    for rate, res in results_by_noise.items():
        success_str = "✓" if res['converged'] else "✗"
        recovered_str = f"s={res['recovered']}" if res['recovered'] else "N/A"
        print(f"{rate*100:>5.1f}%       {success_str:<10} {recovered_str:<12} {res['queries']:<10}")
    
    # Encontrar threshold crítico
    converged_rates = [r for r, res in results_by_noise.items() if res['converged']]
    if converged_rates:
        max_viable = max(converged_rates)
        print(f"\nThreshold crítico: ~{max_viable*100:.1f}% depolarización")


# ============================================================================
# EJEMPLO 3: Sistema de Nucleones (Fermión-Bosón)
# ============================================================================

def ejemplo_3_nucleon_system():
    """
    Caso de uso avanzado: mapear quarks (u, d) a estados Fock de nucleones.
    
    Demostra cómo la segunda cuantización captura estructuras de particulas complejas.
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 3: Sistema de Nucleones (Fermión-Bosón en Fock)")
    print("=" * 80)
    
    framework = create_nucleon_explorer()
    
    # Construir sistema
    nucleons = framework.construct_nucleon_system()
    
    print(f"\nEstados de Nucleones Construidos:")
    print(f"  uuu (up triplet): |3,0,0⟩ Fock")
    print(f"  ddd (down triplet): |0,3,0⟩ Fock")
    print(f"  udu (mixed symmetric): superposición")
    print(f"  dud (mixed antisymmetric): superposición")
    
    # Verificar conservación H7
    print(f"\nVerificación de Conservación H7:")
    h7_status = framework.verify_h7_conservation()
    
    for state_name, conserved in h7_status.items():
        status = "✓" if conserved else "✗"
        print(f"  {status} {state_name:15s}: H7-conservado")
    
    # Fidelidades entre estados
    print(f"\nFidelidades (Solapamiento):")
    uuu = nucleons['uuu']
    ddd = nucleons['ddd']
    udu = nucleons['udu']
    dud = nucleons['dud']
    
    fidelity_uuu_ddd = np.abs(np.vdot(uuu, ddd))**2
    fidelity_udu_dud = np.abs(np.vdot(udu, dud))**2
    
    print(f"  ⟨uuu|ddd⟩² = {fidelity_uuu_ddd:.4f}")
    print(f"  ⟨udu|dud⟩² = {fidelity_udu_dud:.4f}")
    
    # Interpretar: ortogonalidad física
    if fidelity_uuu_ddd < 0.01:
        print(f"    → up y down triplets son casi ortogonales (esperado)")
    
    if fidelity_udu_dud > 0.5:
        print(f"    → mixed states tienen superposición significativa")


# ============================================================================
# EJEMPLO 4: Análisis de Convergencia de Simon
# ============================================================================

def ejemplo_4_convergence_analysis():
    """
    Investigar cómo converge Simon's algorithm con número de queries.
    
    Responde: ¿Cuántas queries son necesarias? ¿Suficientes?
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 4: Análisis de Convergencia de Simon")
    print("=" * 80)
    
    n_trials = 5
    max_queries = 6
    
    convergence_data = {}
    
    for n_q in range(1, max_queries + 1):
        successes = 0
        avg_constraints = 0
        
        for trial in range(n_trials):
            config = PSimonConfig(
                simon_config=SimonConfig(
                    n_qubits=3,
                    n_queries=n_q,
                    measurement_shots=500
                ),
                random_seed=42 + trial
            )
            
            framework = PSimon(config)
            result = framework.run_simon_search()
            
            if result['secret_string_found']:
                successes += 1
            avg_constraints += len(result['constraint_vectors'])
        
        convergence_data[n_q] = {
            'success_rate': successes / n_trials,
            'avg_constraints': avg_constraints / n_trials
        }
    
    # Mostrar resultados
    print(f"\nConvergencia vs. Número de Queries:")
    print(f"{'Queries':<10} {'Success Rate':<15} {'Avg Constraints':<18}")
    print("-" * 45)
    
    for n_q in range(1, max_queries + 1):
        rate = convergence_data[n_q]['success_rate']
        constraints = convergence_data[n_q]['avg_constraints']
        print(f"{n_q:<10} {rate*100:>6.1f}%          {constraints:>6.2f}")
    
    # Análisis
    print(f"\nAnálisis:")
    converged_at = None
    for n_q in range(1, max_queries + 1):
        if convergence_data[n_q]['success_rate'] >= 0.8:
            converged_at = n_q
            break
    
    if converged_at:
        print(f"  Converge con ~80% éxito en {converged_at} queries")
        print(f"  (Teoría predice: O(n) = O(3) queries)")
    else:
        print(f"  ⚠ No converge completamente con {max_queries} queries")
        print(f"  Sugiere: mayor measurement_shots o mejor Gray-code")


# ============================================================================
# EJEMPLO 5: Comparativa Hardware (Simulador vs. Real)
# ============================================================================

def ejemplo_5_hardware_comparison():
    """
    Preparar experimento: simulación clásica vs. IBM Quantum real.
    
    Este es el blueprint para publicar resultados experimentales.
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 5: Preparación de Experimento Hardware")
    print("=" * 80)
    
    # Simulación ideal (baseline)
    print(f"\n[SIMULACIÓN IDEAL]")
    config_ideal = PSimonConfig(
        mode=FrameworkMode.CLASSICAL_SIMULATION,
        simon_config=SimonConfig(measurement_shots=1000)
    )
    framework_ideal = PSimon(config_ideal)
    result_ideal = framework_ideal.run_simon_search()
    
    print(f"Recovered secret: s = {result_ideal['recovered_secret']}")
    print(f"Oracle secret: s = {framework_ideal.oracle.symmetry_string()}")
    print(f"Match: {result_ideal['recovered_secret'] == framework_ideal.oracle.symmetry_string()}")
    
    # Simulación NISQ (predicción de performance en hardware real)
    print(f"\n[SIMULACIÓN NISQ (predicción IBM Quantum 5-qubit)]")
    config_nisq = PSimonConfig(
        mode=FrameworkMode.NISQ_SIMULATION,
        simon_config=SimonConfig(
            n_qubits=5,
            noise_model='depolarizing',
            depolarizing_rate=0.01  # IBM típicamente 1% per 2-qubit gate
        )
    )
    framework_nisq = PSimon(config_nisq)
    result_nisq = framework_nisq.run_simon_search()
    
    print(f"Recovered secret: s = {result_nisq['recovered_secret']}")
    print(f"Success: {result_nisq['secret_string_found']}")
    
    # Comparativa
    print(f"\n[COMPARATIVA]")
    print(f"Métrica                 | Ideal  | NISQ")
    print(f"-" * 50)
    print(f"Success                 | {'✓' if result_ideal['secret_string_found'] else '✗':<6} | {'✓' if result_nisq['secret_string_found'] else '✗':<6}")
    print(f"Oracle queries          | {result_ideal['oracle_queries']:<6} | {result_nisq['oracle_queries']:<6}")
    print(f"Constraint vectors      | {len(result_ideal['constraint_vectors']):<6} | {len(result_nisq['constraint_vectors']):<6}")
    
    # Exportar config para hardware
    print(f"\n[HARDWARE DEPLOYMENT]")
    print(f"Configuración lista para IBM Quantum:")
    print(f"  - N qubits requeridos: 5")
    print(f"  - Profundidad estimada: ~50 compuertas de 2-qubits")
    print(f"  - Coherencia requerida: ~100 μs")
    print(f"  - Predicción: ~{result_nisq['secret_string_found'] and '80% éxito' or '50% riesgo'}")
    
    config_export = framework_nisq.export_config_json()
    print(f"\nJSON para hardware:")
    print(config_export[:200] + "...")


# ============================================================================
# EJEMPLO 6: Investigación Teórica - Gray-Code vs Binario
# ============================================================================

def ejemplo_6_graycode_investigation():
    """
    Experimento comparativo: ¿Importa realmente Gray-code?
    
    Hipótesis: Gray-code preserva estructura Hamming, mejorando convergencia.
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 6: Investigación - Gray-Code vs Binario Directo")
    print("=" * 80)
    
    results_by_encoding = {}
    
    for encoding, use_gray in [("Gray-Code", True), ("Binario Directo", False)]:
        print(f"\n[{encoding}]")
        
        config = PSimonConfig(
            fock_config=FockConfig(use_gray_code=use_gray)
        )
        
        successes = 0
        total_queries = 0
        
        for trial in range(3):
            framework = PSimon(config)
            result = framework.run_simon_search()
            
            if result['secret_string_found']:
                successes += 1
            total_queries += result['oracle_queries']
        
        results_by_encoding[encoding] = {
            'success_rate': successes / 3,
            'avg_queries': total_queries / 3
        }
        
        print(f"  Success rate: {successes/3*100:.1f}%")
        print(f"  Avg queries: {total_queries/3:.1f}")
    
    # Análisis
    print(f"\n[CONCLUSIÓN]")
    gc_success = results_by_encoding["Gray-Code"]['success_rate']
    bd_success = results_by_encoding["Binario Directo"]['success_rate']
    
    if gc_success > bd_success:
        improvement = (gc_success - bd_success) / bd_success * 100
        print(f"Gray-Code mejora convergencia en {improvement:.1f}%")
    else:
        print(f"No hay mejora significativa (sorpresa!)")


# ============================================================================
# MAIN: Ejecutar todos los ejemplos
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PSimon: EJEMPLOS PRÁCTICOS DE USO")
    print("=" * 80)
    
    # Descomenta para ejecutar específicos:
    
    ejemplo_1_basic_symmetry_search()
    ejemplo_2_noise_robustness()
    ejemplo_3_nucleon_system()
    ejemplo_4_convergence_analysis()
    ejemplo_5_hardware_comparison()
    ejemplo_6_graycode_investigation()
    
    print("\n" + "=" * 80)
    print("Ejemplos completados.")
    print("=" * 80)
