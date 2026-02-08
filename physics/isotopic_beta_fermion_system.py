"""
ISOTOPIC NUCLEAR STRUCTURE → BETA DECAY → FERMIONIC OUTPUT

Mapeo de isótopos de hidrógeno (H, D, T) y helio
Con decaimiento beta (n → p + e⁻ + ν̄ₑ)
Salida en estados fermiónicos binarios
Con inferencia bayesiana para corrección de errores
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# NIVEL 0: DEFINICIONES BÁSICAS
# ============================================================================

class ParticleType(Enum):
    """Tipos de fermiones"""
    PROTON = 0       # p⁺
    NEUTRON = 1      # n⁰
    ELECTRON = 2     # e⁻
    POSITRON = 3     # e⁺
    ANTINEUTRINO = 4 # ν̄ₑ


@dataclass
class Nucleon:
    """Representación de nucleón individual"""
    type: str  # 'proton' (0) o 'neutron' (1)
    bit_representation: int  # 0 para protón, 1 para neutrón
    
    @property
    def is_neutron(self) -> bool:
        return self.bit_representation == 1
    
    def __str__(self):
        return "n" if self.is_neutron else "p"


@dataclass
class IsotopeNuclearState:
    """Estado nuclear de un isótopo"""
    name: str           # "H", "D", "T", "He-3", "He-4"
    Z: int              # Número de protones
    N: int              # Número de neutrones
    A: int              # Número de masa (Z + N)
    
    nuclear_string: str  # Ej: "0_1" para deuterio (1p, 1n)
    mass_u: float        # Masa en unidades atómicas
    binding_energy_mev: float
    
    def __post_init__(self):
        """Validación automática"""
        assert self.Z + self.N == self.A, "A debe ser Z + N"
    
    def get_nucleon_list(self) -> List[Nucleon]:
        """Retorna lista de nucleones como objetos"""
        nucleons = []
        nucleons.extend([Nucleon(type='proton', bit_representation=0) for _ in range(self.Z)])
        nucleons.extend([Nucleon(type='neutron', bit_representation=1) for _ in range(self.N)])
        return nucleons
    
    def to_binary_string(self) -> str:
        """Convierte estructura nuclear a string binario"""
        # Orden: primero protones (0), luego neutrones (1)
        return '0' * self.Z + '1' * self.N


# ============================================================================
# NIVEL 1: DEFINICIÓN DE ISÓTOPOS
# ============================================================================

ISOTOPES_NUCLEAR_DB = {
    "H": IsotopeNuclearState(
        name="H",
        Z=1,  # 1 protón
        N=0,  # 0 neutrones
        A=1,
        nuclear_string="0",
        mass_u=1.00783,
        binding_energy_mev=0.0  # Núcleo más simple, no hay binding energy
    ),
    
    "D": IsotopeNuclearState(
        name="D",
        Z=1,  # 1 protón
        N=1,  # 1 neutrón
        A=2,
        nuclear_string="0_1",
        mass_u=2.01410,
        binding_energy_mev=1.112
    ),
    
    "T": IsotopeNuclearState(
        name="T",
        Z=1,  # 1 protón
        N=2,  # 2 neutrones
        A=3,
        nuclear_string="0_1_1",
        mass_u=3.01605,
        binding_energy_mev=2.827
    ),
    
    "He-3": IsotopeNuclearState(
        name="He-3",
        Z=2,  # 2 protones
        N=1,  # 1 neutrón
        A=3,
        nuclear_string="0_0_1",
        mass_u=3.01603,
        binding_energy_mev=7.718
    ),
    
    "He-4": IsotopeNuclearState(
        name="He-4",
        Z=2,  # 2 protones
        N=2,  # 2 neutrones
        A=4,
        nuclear_string="0_0_1_1",
        mass_u=4.00260,
        binding_energy_mev=28.296
    )
}


# ============================================================================
# NIVEL 2: DECAIMIENTO BETA (n → p + e⁻ + ν̄ₑ)
# ============================================================================

@dataclass
class BetaDecayProcess:
    """Proceso de decaimiento beta"""
    initial_isotope: str      # "T" o "He-3"
    products_isotope: str     # Isótopo resultante después de decaimiento
    emitted_electron: bool    # True si emite e⁻ (beta minus)
    emitted_antineutrino: bool # True si emite ν̄ₑ
    
    Q_value_mev: float       # Energía liberada en MeV


# Tablas de decaimiento beta
BETA_DECAY_PROCESSES = {
    "T": BetaDecayProcess(
        initial_isotope="T",
        products_isotope="He-3",
        emitted_electron=True,
        emitted_antineutrino=True,
        Q_value_mev=0.01857  # Tritio → He-3 + e⁻ + ν̄ₑ
    ),
    
    "He-3": BetaDecayProcess(
        initial_isotope="He-3",
        products_isotope="He-3",  # No cambia para He-3 en este contexto
        emitted_electron=False,
        emitted_antineutrino=False,
        Q_value_mev=0.0  # He-3 es estable (beta plus es muy débil)
    )
}


@dataclass
class FermionicDecayOutput:
    """Output fermiónico después de decaimiento beta"""
    parent_nucleus: str           # Isótopo padre
    daughter_nucleus: str         # Isótopo resultante
    
    parent_binary_string: str     # String binario del núcleo padre
    daughter_binary_string: str   # String binario del núcleo hija
    
    electron_output: bool         # True si se emite e⁻
    antineutrino_output: bool     # True si se emite ν̄ₑ
    
    # Output binario en registro fermiónico
    fermionic_output_string: str  # Ej: "101" = [electron][antineutrino][photon]
    
    decay_energy_mev: float


class BetaDecaySimulator:
    """Simula decaimiento beta y genera outputs fermiónicos"""
    
    def __init__(self, nuclear_db: Dict[str, IsotopeNuclearState]):
        self.db = nuclear_db
        self.decay_processes = BETA_DECAY_PROCESSES
    
    def simulate_decay(self, isotope_name: str) -> FermionicDecayOutput:
        """
        Simula decaimiento beta de un isótopo.
        
        Flujo:
        1. Obtener isótopo padre
        2. Un neutrón → protón (n cambia a p)
        3. Emitir e⁻ + ν̄ₑ
        4. Generar output fermiónico binario
        """
        
        if isotope_name not in self.decay_processes:
            raise ValueError(f"No decay process defined for {isotope_name}")
        
        decay_process = self.decay_processes[isotope_name]
        
        parent = self.db[decay_process.initial_isotope]
        daughter_name = decay_process.products_isotope
        daughter = self.db[daughter_name]
        
        # Strings nucleares
        parent_binary = parent.to_binary_string()
        daughter_binary = daughter.to_binary_string()
        
        # Output fermiónico: [e⁻][ν̄ₑ]
        # En este caso, ambos se emiten en decaimiento beta minus
        fermionic_bits = []
        fermionic_bits.append(1 if decay_process.emitted_electron else 0)
        fermionic_bits.append(1 if decay_process.emitted_antineutrino else 0)
        
        fermionic_output = ''.join(map(str, fermionic_bits))
        
        return FermionicDecayOutput(
            parent_nucleus=decay_process.initial_isotope,
            daughter_nucleus=daughter_name,
            parent_binary_string=parent_binary,
            daughter_binary_string=daughter_binary,
            electron_output=decay_process.emitted_electron,
            antineutrino_output=decay_process.emitted_antineutrino,
            fermionic_output_string=fermionic_output,
            decay_energy_mev=decay_process.Q_value_mev
        )


# ============================================================================
# NIVEL 3: INFERENCIA BAYESIANA PARA CORRECCIÓN DE ERRORES
# ============================================================================

class BayesianIsotopeCorrector:
    """
    Infiere el isótopo correcto a partir de observaciones ruidosas.
    
    Utiliza:
    - Prior: probabilidad de cada isótopo
    - Likelihood: probabilidad de observación | isótopo
    - Posterior: probabilidad de isótopo | observación (Bayes)
    """
    
    def __init__(self, nuclear_db: Dict[str, IsotopeNuclearState]):
        self.db = nuclear_db
        
        # Prior: distribución de probabilidad de encontrar cada isótopo
        # (En naturaleza, H es ~99.98%, D es ~0.02%, T es radiactivo)
        self.prior = {
            "H": 0.9998,
            "D": 0.0002,
            "T": 1e-10,      # Radiactivo, muy raro
            "He-3": 0.00013,
            "He-4": 0.99987
        }
        
        # Likelihood models (cómo los observables varían con el isótopo)
        self.observation_models = {
            "mass_mev": self._get_mass_model(),
            "binding_energy": self._get_binding_energy_model(),
            "decay_signature": self._get_decay_model()
        }
    
    def _get_mass_model(self) -> Dict:
        """Modelo de observación: masa nuclear"""
        return {
            isotope: {
                "mean": iso.mass_u,
                "std": iso.mass_u * 0.001  # 0.1% error típico
            }
            for isotope, iso in self.db.items()
        }
    
    def _get_binding_energy_model(self) -> Dict:
        """Modelo de observación: energía de binding"""
        return {
            isotope: {
                "mean": iso.binding_energy_mev,
                "std": max(0.01, iso.binding_energy_mev * 0.01)  # 1% error
            }
            for isotope, iso in self.db.items()
        }
    
    def _get_decay_model(self) -> Dict:
        """Modelo de observación: firma de decaimiento beta"""
        return {
            isotope: {
                "emits_electron": "T" in isotope or isotope == "T",
                "emits_antineutrino": "T" in isotope or isotope == "T"
            }
            for isotope in self.db.keys()
        }
    
    def gaussian_likelihood(self, observation: float, mean: float, std: float) -> float:
        """Calcula P(observación | media, std) usando distribución gaussiana"""
        if std == 0:
            return 1.0 if observation == mean else 0.0
        
        exponent = -((observation - mean) ** 2) / (2 * std ** 2)
        return np.exp(exponent) / (std * np.sqrt(2 * np.pi))
    
    def infer_isotope(self, observed_mass_u: float, 
                     observed_binding_energy: float = None) -> Dict:
        """
        Usa inferencia bayesiana para determinar el isótopo más probable.
        
        Input:
            observed_mass_u: Masa medida (en unidades atómicas)
            observed_binding_energy: Energía de binding medida (opcional)
        
        Output:
            {
                'inferred_isotope': str,
                'confidence': float (0-1),
                'posterior_dist': Dict[str, float],
                'alternatives': List[Tuple[str, float]]
            }
        """
        
        posteriors = {}
        
        for isotope in self.db.keys():
            # P(isótopo) = prior
            prior_prob = self.prior.get(isotope, 1e-6)
            
            # P(observación | isótopo)
            mass_model = self.observation_models['mass_mev'][isotope]
            mass_likelihood = self.gaussian_likelihood(
                observation=observed_mass_u,
                mean=mass_model['mean'],
                std=mass_model['std']
            )
            
            # Si se proporcionó binding energy, incluirlo
            combined_likelihood = mass_likelihood
            if observed_binding_energy is not None:
                be_model = self.observation_models['binding_energy'][isotope]
                be_likelihood = self.gaussian_likelihood(
                    observation=observed_binding_energy,
                    mean=be_model['mean'],
                    std=be_model['std']
                )
                combined_likelihood *= be_likelihood
            
            # P(isótopo | observación) ∝ P(observación | isótopo) × P(isótopo)
            posterior_prob = combined_likelihood * prior_prob
            posteriors[isotope] = posterior_prob
        
        # Normalizar
        total_posterior = sum(posteriors.values())
        posteriors = {k: v / total_posterior for k, v in posteriors.items()}
        
        # Encontrar máximo
        best_isotope = max(posteriors, key=posteriors.get)
        confidence = posteriors[best_isotope]
        
        # Alternativas ordenadas
        alternatives = sorted(
            posteriors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'inferred_isotope': best_isotope,
            'confidence': confidence,
            'posterior_distribution': posteriors,
            'alternatives': alternatives[:3],
            'recommendation': 'HIGH CONFIDENCE' if confidence > 0.8 else 'UNCERTAIN'
        }
    
    def detect_corruption(self, observed_binary_string: str, 
                         true_isotope: str) -> Dict:
        """
        Detecta si el string binario ha sido corrupto (bits flipped).
        
        Input:
            observed_binary_string: String binario observado (posiblemente corrupto)
            true_isotope: Isótopo conocido (ground truth)
        
        Output:
            {
                'error_rate': float,
                'errors': List[Dict],
                'corrected_string': str,
                'confidence': float
            }
        """
        
        true_binary = self.db[true_isotope].to_binary_string()
        
        if len(observed_binary_string) != len(true_binary):
            return {
                'error': 'Length mismatch',
                'observed_len': len(observed_binary_string),
                'true_len': len(true_binary)
            }
        
        errors = []
        for position, (observed_bit, true_bit) in enumerate(
            zip(observed_binary_string, true_binary)
        ):
            if observed_bit != true_bit:
                errors.append({
                    'position': position,
                    'found': observed_bit,
                    'should_be': true_bit,
                    'nucleon_type': 'neutron' if true_bit == '1' else 'proton'
                })
        
        error_rate = len(errors) / len(true_binary) if len(true_binary) > 0 else 0
        
        # Confianza en corrección basada en tasa de error
        if error_rate == 0:
            confidence = 1.0
        elif error_rate < 0.1:
            confidence = 0.99
        elif error_rate < 0.2:
            confidence = 0.9
        elif error_rate < 0.5:
            confidence = 0.7
        else:
            confidence = 0.3
        
        return {
            'error_rate': error_rate,
            'num_errors': len(errors),
            'errors': errors,
            'corrected_string': true_binary,
            'confidence': confidence,
            'recommendation': 'RELIABLE' if confidence > 0.9 else 'CHECK MANUALLY'
        }


# ============================================================================
# NIVEL 4: GENERADOR DE JSON ESTRUCTURADO
# ============================================================================

def generate_system_json() -> str:
    """Genera JSON completo del sistema isotópico"""
    
    system_structure = {
        "nuclear_system": {
            "description": "Mapeo de isótopos de hidrógeno (H, D, T) y helio con decaimiento beta",
            "isotopes": {}
        },
        
        "binary_encoding": {
            "description": "Codificación binaria: 0=protón, 1=neutrón",
            "rules": {
                "H": {
                    "Z": 1,
                    "N": 0,
                    "binary_string": "0",
                    "nucleons": ["p"],
                    "interpretation": "1 protón, 0 neutrones"
                },
                "D": {
                    "Z": 1,
                    "N": 1,
                    "binary_string": "0_1",
                    "nucleons": ["p", "n"],
                    "interpretation": "1 protón, 1 neutrón"
                },
                "T": {
                    "Z": 1,
                    "N": 2,
                    "binary_string": "0_1_1",
                    "nucleons": ["p", "n", "n"],
                    "interpretation": "1 protón, 2 neutrones (radiactivo, decae a He-3)"
                },
                "He-3": {
                    "Z": 2,
                    "N": 1,
                    "binary_string": "0_0_1",
                    "nucleons": ["p", "p", "n"],
                    "interpretation": "2 protones, 1 neutrón (producto de decaimiento T)"
                },
                "He-4": {
                    "Z": 2,
                    "N": 2,
                    "binary_string": "0_0_1_1",
                    "nucleons": ["p", "p", "n", "n"],
                    "interpretation": "2 protones, 2 neutrones (núcleo de helio estable)"
                }
            }
        },
        
        "beta_decay_processes": {
            "description": "Decaimiento beta: n → p + e⁻ + ν̄ₑ",
            "processes": {}
        },
        
        "fermionic_output": {
            "description": "Output binario fermiónico después de decaimiento",
            "fermion_bits": {
                "bit_0": "electron (e⁻)",
                "bit_1": "antineutrino (ν̄ₑ)"
            }
        }
    }
    
    # Llenar isótopos
    for iso_name, iso_obj in ISOTOPES_NUCLEAR_DB.items():
        system_structure["nuclear_system"]["isotopes"][iso_name] = {
            "name": iso_obj.name,
            "Z": iso_obj.Z,
            "N": iso_obj.N,
            "A": iso_obj.A,
            "mass_u": iso_obj.mass_u,
            "binding_energy_mev": iso_obj.binding_energy_mev,
            "binary_string": iso_obj.to_binary_string()
        }
    
    # Llenar procesos de decaimiento
    for iso_name, decay_proc in BETA_DECAY_PROCESSES.items():
        system_structure["beta_decay_processes"]["processes"][iso_name] = {
            "parent": decay_proc.initial_isotope,
            "daughter": decay_proc.products_isotope,
            "emits_electron": decay_proc.emitted_electron,
            "emits_antineutrino": decay_proc.emitted_antineutrino,
            "Q_value_mev": decay_proc.Q_value_mev,
            "reaction": "n → p + e⁻ + ν̄ₑ" if decay_proc.emitted_electron else "stable"
        }
    
    return json.dumps(system_structure, indent=2)


# ============================================================================
# NIVEL 5: DEMOSTRACIÓN Y VALIDACIÓN
# ============================================================================

def demonstrate_system():
    """Demostración completa del sistema"""
    
    print("=" * 80)
    print("SISTEMA ISOTÓPICO NUCLEAR CON DECAIMIENTO BETA Y FERMIONES")
    print("=" * 80)
    
    # ────────────────────────────────────────────────────────────────────────
    # 1. MOSTRAR ESTRUCTURAS NUCLEARES
    # ────────────────────────────────────────────────────────────────────────
    print("\n1. ESTRUCTURAS NUCLEARES BINARIAS")
    print("-" * 80)
    
    for iso_name, iso_obj in ISOTOPES_NUCLEAR_DB.items():
        print(f"\n{iso_name} (A={iso_obj.A}, Z={iso_obj.Z}, N={iso_obj.N})")
        print(f"  Binary:  {iso_obj.to_binary_string()}")
        print(f"  Mass:    {iso_obj.mass_u:.5f} u")
        print(f"  Binding: {iso_obj.binding_energy_mev:.3f} MeV")
        print(f"  Nucléons: {' + '.join([str(n) for n in iso_obj.get_nucleon_list()])}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 2. SIMULAR DECAIMIENTO BETA
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. DECAIMIENTO BETA (n → p + e⁻ + ν̄ₑ)")
    print("-" * 80)
    
    decay_simulator = BetaDecaySimulator(ISOTOPES_NUCLEAR_DB)
    
    for isotope_to_decay in ["T", "He-3"]:
        print(f"\n{isotope_to_decay} DECAY:")
        try:
            decay_output = decay_simulator.simulate_decay(isotope_to_decay)
            
            print(f"  Padre:          {decay_output.parent_nucleus}")
            print(f"  Binario padre:  {decay_output.parent_binary_string}")
            print(f"  Hija:           {decay_output.daughter_nucleus}")
            print(f"  Binario hija:   {decay_output.daughter_binary_string}")
            print(f"  Emite e⁻:       {decay_output.electron_output}")
            print(f"  Emite ν̄ₑ:      {decay_output.antineutrino_output}")
            print(f"  Output fermión: {decay_output.fermionic_output_string}")
            print(f"  Q (energía):    {decay_output.decay_energy_mev:.5f} MeV")
        except ValueError as e:
            print(f"  {e}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 3. INFERENCIA BAYESIANA
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. INFERENCIA BAYESIANA (Identificación de isótopo)")
    print("-" * 80)
    
    corrector = BayesianIsotopeCorrector(ISOTOPES_NUCLEAR_DB)
    
    # Caso 1: Observación perfecta de deuterio
    print("\nCaso 1: Masa observada = 2.01410 u (deuterio)")
    result1 = corrector.infer_isotope(observed_mass_u=2.01410)
    print(f"  Isótopo inferido: {result1['inferred_isotope']}")
    print(f"  Confianza:        {result1['confidence']:.4f}")
    print(f"  Top 3 alternativas:")
    for isotope, prob in result1['alternatives']:
        print(f"    {isotope}: {prob:.4f}")
    
    # Caso 2: Observación ruidosa
    print("\nCaso 2: Masa observada = 2.014 u (con ruido)")
    result2 = corrector.infer_isotope(observed_mass_u=2.014)
    print(f"  Isótopo inferido: {result2['inferred_isotope']}")
    print(f"  Confianza:        {result2['confidence']:.4f}")
    
    # Caso 3: Con energía de binding
    print("\nCaso 3: Masa = 3.016 u + Binding = 7.718 MeV (He-3)")
    result3 = corrector.infer_isotope(
        observed_mass_u=3.016,
        observed_binding_energy=7.718
    )
    print(f"  Isótopo inferido: {result3['inferred_isotope']}")
    print(f"  Confianza:        {result3['confidence']:.4f}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 4. DETECCIÓN DE ERRORES
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("4. DETECCIÓN Y CORRECCIÓN DE ERRORES")
    print("-" * 80)
    
    # String correcto para deuterio: "01"
    true_isotope = "D"
    true_binary = ISOTOPES_NUCLEAR_DB[true_isotope].to_binary_string()
    
    print(f"\nIsótopo verdadero: {true_isotope}")
    print(f"String binario verdadero: {true_binary}")
    
    # Corromper el string (cambiar 1 bit)
    corrupted = list(true_binary)
    corrupted[0] = '1'  # Cambiar primer bit (protón → neutrón)
    corrupted_str = ''.join(corrupted)
    
    print(f"String observado (corrupto): {corrupted_str}")
    
    error_report = corrector.detect_corruption(corrupted_str, true_isotope)
    
    if 'error' not in error_report:
        print(f"  Tasa de error:    {error_report['error_rate']:.1%}")
        print(f"  Número de errores: {error_report['num_errors']}")
        if error_report['errors']:
            for err in error_report['errors']:
                print(f"    Posición {err['position']}: {err['found']} → {err['should_be']} ({err['nucleon_type']})")
        print(f"  Confianza corrección: {error_report['confidence']:.2%}")
        print(f"  Recomendación: {error_report['recommendation']}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 5. MOSTRAR JSON COMPLETO
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. ESTRUCTURA JSON COMPLETA")
    print("-" * 80)
    
    json_output = generate_system_json()
    print(json_output)


if __name__ == "__main__":
    demonstrate_system()
    
    # Guardar JSON a archivo
    json_output = generate_system_json()
    with open('data/isotopic_nuclear_system.json', 'w') as f:
        f.write(json_output)
    
    print("\n✓ JSON guardado en: data/isotopic_nuclear_system.json")
