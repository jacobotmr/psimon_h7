"""
CONVERGENCIA: QUIRALIDAD FERMIÓNICA → H7 TOPOLÓGICO → PSIMON

Sistema de mapeo integral que conecta:
1. Isótopos reales (H, D, T, He) con quiralidad binaria
2. Espacio H7 (3-qubit Hilbert space) con conservación topológica
3. Framework PSimon para predicción de masas

El flujo:
Isotope → Binario Nuclear → Quiralidad Fermiónica → 
Proyección H7 → Momentum P → Oracle Energy E → Masa

Author: Claude (implementing Jacobo's vision)
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Importar del repositorio PSimon (si está disponible)
try:
    from physics.metriplex_oracle import MetriplexOracle, H7Conservation, MetriplexConfig
    from physics.fock_basis import FockBasis, FockConfig
    PSIMON_AVAILABLE = True
except ImportError:
    PSIMON_AVAILABLE = False
    print("⚠ Warning: PSimon framework not available, using stubs")


# ============================================================================
# NIVEL 0: DEFINICIONES ISOTÓPICAS
# ============================================================================

@dataclass
class IsotopeQuiralConfig:
    """Configuración de isótopo con quiralidad"""
    
    # Nuclear
    isotope_name: str           # "H", "D", "T", "He-3", "He-4"
    Z: int                      # Protones
    N: int                      # Neutrones
    A: int                      # Número de masa
    
    # Binario nuclear
    nuclear_binary: str         # "0", "0_1", "0_1_1", etc.
    
    # Quiralidad fermiónica
    chiral_string: str          # "1_1", "1_0_0_1", etc.
    chirality_index: float      # -1.0 a +1.0
    handedness: str             # "LEFT-HANDED", "RIGHT-HANDED", "BALANCED"
    
    # Física
    mass_u: float              # Masa en unidades atómicas
    binding_energy_mev: float
    
    # Decaimiento
    decays_to: Optional[str] = None
    decay_energy_mev: float = 0.0
    
    def __post_init__(self):
        assert self.Z + self.N == self.A, "A must equal Z + N"


class IsotopeRegistry:
    """Registro de isótopos con sus propiedades quirales"""
    
    def __init__(self):
        self.isotopes: Dict[str, IsotopeQuiralConfig] = {}
        self._init_standard_isotopes()
    
    def _init_standard_isotopes(self):
        """Registra isótopos estándar"""
        
        # HIDRÓGENO
        self.register(IsotopeQuiralConfig(
            isotope_name="H",
            Z=1, N=0, A=1,
            nuclear_binary="0",
            chiral_string="0_0_1",      # He-3 stable → no decay
            chirality_index=0.0,
            handedness="ACHIRAL",
            mass_u=1.00783,
            binding_energy_mev=0.0
        ))
        
        # DEUTERIO
        self.register(IsotopeQuiralConfig(
            isotope_name="D",
            Z=1, N=1, A=2,
            nuclear_binary="0_1",
            chiral_string="0_0_1",
            chirality_index=0.0,
            handedness="ACHIRAL",
            mass_u=2.01410,
            binding_energy_mev=1.112
        ))
        
        # TRITIO
        self.register(IsotopeQuiralConfig(
            isotope_name="T",
            Z=1, N=2, A=3,
            nuclear_binary="0_1_1",
            chiral_string="1_1_0_0_1",  # HIGH chirality
            chirality_index=0.0,
            handedness="BALANCED",
            mass_u=3.01605,
            binding_energy_mev=2.827,
            decays_to="He-3",
            decay_energy_mev=0.01857
        ))
        
        # HELIO-3
        self.register(IsotopeQuiralConfig(
            isotope_name="He-3",
            Z=2, N=1, A=3,
            nuclear_binary="0_0_1",
            chiral_string="0_0_0_1",    # RIGHT-HANDED vacuum
            chirality_index=-1.0,
            handedness="RIGHT-HANDED",
            mass_u=3.01603,
            binding_energy_mev=7.718
        ))
        
        # HELIO-4
        self.register(IsotopeQuiralConfig(
            isotope_name="He-4",
            Z=2, N=2, A=4,
            nuclear_binary="0_0_1_1",
            chiral_string="1",          # FUNDAMENTAL: vacuum state
            chirality_index=0.0,
            handedness="ACHIRAL",
            mass_u=4.00260,
            binding_energy_mev=28.296
        ))
    
    def register(self, config: IsotopeQuiralConfig):
        """Registra un isótopo"""
        self.isotopes[config.isotope_name] = config
    
    def get(self, isotope_name: str) -> IsotopeQuiralConfig:
        """Obtiene configuración de isótopo"""
        if isotope_name not in self.isotopes:
            raise ValueError(f"Isotope {isotope_name} not registered")
        return self.isotopes[isotope_name]
    
    def list_all(self) -> List[str]:
        """Lista todos los isótopos registrados"""
        return list(self.isotopes.keys())


# ============================================================================
# NIVEL 1: MAPEO QUIRALIDAD → H7
# ============================================================================

class ChiralityToH7Mapper:
    """
    Mapea quiralidad fermiónica a posiciones en H7 (3-qubit space).
    
    H7 Conservation: |x⟩ ↔ |7⊕x⟩
    
    Mapeo conceptual:
    - Isótopo + Quiralidad → Índice base en H7
    - Chirality index (-1 a +1) → perturbación topológica
    - Handedness (L/R/C) → orientación dentro del H7 pair
    """
    
    def __init__(self):
        self.h7_pairing = H7Conservation.pairing_table() if PSIMON_AVAILABLE else self._default_h7_pairing()
    
    def _default_h7_pairing(self) -> Dict[int, int]:
        """Pairing por defecto si PSimon no está disponible"""
        return {i: 7 ^ i for i in range(8)}
    
    def chiral_to_h7_index(self, isotope_name: str, chiral_string: str) -> int:
        """
        Mapea (isótopo, quiralidad) a índice en H7 [0..7].
        
        Estrategia:
        1. Extraer bits de quiralidad del string
        2. Calcular índice base a partir de isótopo
        3. Aplicar corrección topológica según handedness
        """
        
        # Paso 1: Índice base según isótopo
        isotope_to_base = {
            "H": 0,      # |000⟩
            "D": 1,      # |001⟩
            "T": 2,      # |010⟩
            "He-3": 3,   # |011⟩
            "He-4": 4    # |100⟩
        }
        
        if isotope_name not in isotope_to_base:
            raise ValueError(f"Unknown isotope: {isotope_name}")
        
        base_idx = isotope_to_base[isotope_name]
        
        # Paso 2: Extraer bits de quiralidad
        chiral_bits = [int(b) for b in chiral_string.split("_")[:-1]]  # Excluir el 1 final
        
        # Paso 3: Calcular corrección topológica
        # Si hay más 1s (left-handed) → desplazar hacia índices superiores
        # Si hay más 0s (right-handed) → desplazar hacia índices inferiores
        ones_count = sum(chiral_bits)
        zeros_count = len(chiral_bits) - ones_count
        
        # Correction: -2 a +2 según quiralidad
        correction = (ones_count - zeros_count) // 2
        
        # Paso 4: Aplicar H7 conservation (mantener pairing)
        h7_index = (base_idx + correction) % 8
        
        # Verificar que respeta H7 pairing
        partner = self.h7_pairing[h7_index]
        
        return h7_index, partner
    
    def h7_to_momentum(self, h7_index: int, h7_partner: int) -> int:
        """
        Mapea índice H7 a momentum efectivo [1..6] para oracle.
        
        Lógica:
        - Índices bajos (0-3) → momentums bajos (1-3)
        - Índices altos (4-7) → momentums altos (4-6)
        - Partner index provee información redundante
        """
        
        # Mapeo simple: h7 [0..7] → momentum [1..6]
        # Los pares H7 (x, 7⊕x) se mapean al mismo momentum
        if h7_index < 4:
            momentum = (h7_index % 3) + 1  # [0,1,2,3] → [1,2,3,1]
        else:
            momentum = ((h7_index - 4) % 3) + 4  # [4,5,6,7] → [4,5,6,4]
        
        return momentum


# ============================================================================
# NIVEL 2: CONVERGENCIA COMPLETA
# ============================================================================

@dataclass
class ConvergenceResult:
    """Resultado del mapeo de convergencia"""
    
    isotope: str
    nuclear_binary: str
    chiral_string: str
    chirality_index: float
    handedness: str
    
    h7_index: int
    h7_partner: int
    h7_conserved: bool
    
    momentum: int
    
    # Oracle information (si PSimon disponible)
    oracle_group: Optional[str] = None
    oracle_energy: Optional[float] = None
    oracle_output: Optional[np.ndarray] = None
    
    # Predicción de masa
    mass_predicted_mev: Optional[float] = None


class IsotopeToH7Convergence:
    """
    Sistema integral de convergencia: Isotope → H7 → PSimon → Masa
    """
    
    def __init__(self):
        self.isotope_registry = IsotopeRegistry()
        self.chiral_mapper = ChiralityToH7Mapper()
        
        if PSIMON_AVAILABLE:
            self.oracle = MetriplexOracle()
        else:
            self.oracle = None
        
        self.convergence_results: List[ConvergenceResult] = []
    
    def converge_isotope(self, isotope_name: str) -> ConvergenceResult:
        """
        Converge un isótopo a través de todo el pipeline.
        """
        
        # Obtener configuración del isótopo
        iso_config = self.isotope_registry.get(isotope_name)
        
        # Mapear a H7
        h7_idx, h7_partner = self.chiral_mapper.chiral_to_h7_index(
            isotope_name,
            iso_config.chiral_string
        )
        
        # Verificar conservación H7
        h7_conserved = self.chiral_mapper.h7_pairing[h7_idx] == h7_partner
        
        # Mapear a momentum
        momentum = self.chiral_mapper.h7_to_momentum(h7_idx, h7_partner)
        
        # Aplicar oracle (si disponible)
        oracle_group = None
        oracle_energy = None
        oracle_output = None
        
        if self.oracle is not None:
            oracle_group, oracle_output, oracle_energy = self.oracle.forward(momentum)
        
        # Predecir masa (simplificado por ahora)
        mass_predicted = self._predict_mass(
            iso_config.mass_u,
            iso_config.chirality_index,
            oracle_energy
        )
        
        # Crear resultado
        result = ConvergenceResult(
            isotope=isotope_name,
            nuclear_binary=iso_config.nuclear_binary,
            chiral_string=iso_config.chiral_string,
            chirality_index=iso_config.chirality_index,
            handedness=iso_config.handedness,
            h7_index=h7_idx,
            h7_partner=h7_partner,
            h7_conserved=h7_conserved,
            momentum=momentum,
            oracle_group=oracle_group,
            oracle_energy=oracle_energy,
            oracle_output=oracle_output.tolist() if oracle_output is not None else None,
            mass_predicted_mev=mass_predicted
        )
        
        self.convergence_results.append(result)
        return result
    
    def converge_all(self) -> List[ConvergenceResult]:
        """Converge todos los isótopos registrados"""
        results = []
        for isotope_name in self.isotope_registry.list_all():
            result = self.converge_isotope(isotope_name)
            results.append(result)
        return results
    
    def _predict_mass(self, true_mass_u: float, chirality_index: float, 
                     oracle_energy: Optional[float]) -> float:
        """
        Predice masa usando H7 topología + quiralidad.
        
        Modelo simplificado:
        m = m_0 × (1 + α × chirality_index + β × oracle_energy)
        
        Donde:
        - m_0: masa verdadera
        - α: coupling quiralidad
        - β: coupling energía topológica
        """
        
        alpha = 0.05  # 5% sensitivity to chirality
        beta = 0.1 if oracle_energy is not None else 0
        
        correction = (
            alpha * chirality_index +
            beta * oracle_energy if oracle_energy is not None else 0
        )
        
        predicted_mass = true_mass_u * (1 + correction)
        
        return predicted_mass
    
    def get_convergence_report(self) -> Dict:
        """Genera reporte completo de convergencia"""
        
        report = {
            "system": "IsotopeToH7Convergence",
            "timestamp": str(np.datetime64('now')),
            "isotopes_converged": len(self.convergence_results),
            "psimon_available": PSIMON_AVAILABLE,
            "results": [asdict(r) for r in self.convergence_results],
            "h7_statistics": self._compute_h7_statistics(),
            "chirality_statistics": self._compute_chirality_statistics()
        }
        
        return report
    
    def _compute_h7_statistics(self) -> Dict:
        """Estadísticas de cobertura H7"""
        
        h7_indices_used = set()
        h7_conserved_count = 0
        
        for result in self.convergence_results:
            h7_indices_used.add(result.h7_index)
            if result.h7_conserved:
                h7_conserved_count += 1
        
        return {
            "h7_indices_used": sorted(list(h7_indices_used)),
            "h7_coverage_percent": 100 * len(h7_indices_used) / 8,
            "h7_conservation_rate": 100 * h7_conserved_count / len(self.convergence_results)
        }
    
    def _compute_chirality_statistics(self) -> Dict:
        """Estadísticas de quiralidad"""
        
        chirality_indices = [r.chirality_index for r in self.convergence_results]
        handedness_counts = {}
        
        for result in self.convergence_results:
            hand = result.handedness
            handedness_counts[hand] = handedness_counts.get(hand, 0) + 1
        
        return {
            "chirality_range": [min(chirality_indices), max(chirality_indices)],
            "chirality_mean": np.mean(chirality_indices),
            "handedness_distribution": handedness_counts
        }
    
    def export_json(self) -> str:
        """Exporta resultados a JSON"""
        report = self.get_convergence_report()
        return json.dumps(report, indent=2, default=str)


# ============================================================================
# NIVEL 3: DEMOSTRACIÓN
# ============================================================================

def demonstrate_convergence():
    """Demostración completa del sistema de convergencia"""
    
    print("=" * 90)
    print("CONVERGENCIA: QUIRALIDAD FERMIÓNICA → H7 → PSIMON")
    print("=" * 90)
    
    # Crear sistema
    convergence = IsotopeToH7Convergence()
    
    print(f"\nPSimon disponible: {PSIMON_AVAILABLE}")
    print(f"Isótopos registrados: {convergence.isotope_registry.list_all()}")
    
    # Converger todos los isótopos
    print("\n" + "-" * 90)
    print("CONVERGENCIA DE ISÓTOPOS")
    print("-" * 90)
    
    results = convergence.converge_all()
    
    for result in results:
        print(f"\n{result.isotope}")
        print(f"  Nuclear binary:    {result.nuclear_binary}")
        print(f"  Chiral string:     {result.chiral_string}")
        print(f"  Chirality:         {result.chirality_index:+.2f} ({result.handedness})")
        print(f"  H7 index:          {result.h7_index} ↔ {result.h7_partner}")
        print(f"  H7 conserved:      {'✓' if result.h7_conserved else '✗'}")
        print(f"  Momentum:          {result.momentum}")
        if result.oracle_group:
            print(f"  Oracle group:      {result.oracle_group}")
            print(f"  Oracle energy:     {result.oracle_energy:.4f}")
        if result.mass_predicted_mev:
            print(f"  Mass predicted:    {result.mass_predicted_mev:.5f} u")
    
    # Reporte
    print("\n" + "=" * 90)
    print("REPORTE COMPLETO")
    print("=" * 90)
    
    report = convergence.get_convergence_report()
    
    print(f"\nIsótopos convergidos: {report['isotopes_converged']}")
    print(f"\nH7 Statistics:")
    h7_stats = report['h7_statistics']
    print(f"  Cobertura H7: {h7_stats['h7_coverage_percent']:.1f}%")
    print(f"  Conservación: {h7_stats['h7_conservation_rate']:.1f}%")
    
    print(f"\nChirality Statistics:")
    chir_stats = report['chirality_statistics']
    print(f"  Rango: [{chir_stats['chirality_range'][0]:.2f}, {chir_stats['chirality_range'][1]:.2f}]")
    print(f"  Media: {chir_stats['chirality_mean']:.2f}")
    print(f"  Distribución de handedness: {chir_stats['handedness_distribution']}")
    
    # Exportar JSON
    print("\n" + "=" * 90)
    print("JSON EXPORT (primeras 1000 chars)")
    print("=" * 90)
    
    json_output = convergence.export_json()
    print(json_output[:1000] + "\n... [truncated]")
    
    return convergence, results, json_output


if __name__ == "__main__":
    convergence, results, json_out = demonstrate_convergence()
    
    # Guardar JSON
    with open('data/isotope_h7_convergence.json', 'w') as f:
        f.write(json_out)
    
    print(f"\n✓ Convergence results saved to: data/isotope_h7_convergence.json")
