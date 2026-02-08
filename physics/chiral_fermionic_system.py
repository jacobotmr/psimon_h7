"""
QUIRALIZACIÓN DE SALIDA FERMIÓNICA

Output fermiónico → Estados quirales binarios
Estructura: [bits variables]_1 (siempre termina en 1)

Ejemplos:
- Decaimiento con e⁻: → 1_1 (electron, antineutrino, quiral marker)
- Decaimiento sin e⁻: → 1_0_0_1 (sin electron, patrón específico, quiral marker)
- Diferentes caminos de decaimiento → diferentes patrones quirales

La terminación en 1 actúa como "cierre quiral" o "paridad topológica"
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# NIVEL 0: DEFINICIÓN DE QUIRALIDADES
# ============================================================================

class Chirality(Enum):
    """Tipos de quiralidad fermiónica"""
    LEFT = "L"    # Zurdo (left-handed)
    RIGHT = "R"   # Diestro (right-handed)
    CENTER = "C"  # Centro (achiral)


@dataclass
class ChiralFermionicState:
    """Estado fermiónico quiral"""
    isotope: str                    # Isótopo padre
    product_isotope: str            # Isótopo hijo
    
    # Estructura del output
    fermionic_output: str           # Output original (ej: "11" para e⁺ν̄)
    
    # Conversión quiral
    chiral_string: str              # String quiral (ej: "1_1_0_0_1")
    chirality_type: Chirality       # Tipo de quiralidad (L, R, C)
    
    # Análisis
    bit_pattern: List[int]          # Patrón de bits descompuesto
    middle_bits: str                # Bits en el medio (sin terminación)
    chiral_marker: int              # Siempre debe ser 1 (terminación)
    
    # Energía
    decay_energy_mev: float
    decay_channel: str              # Descripción del canal de decaimiento
    
    # Fase topológica
    topological_phase: float        # Fase acumulada en ciclo quiral


class ChiralEncoder:
    """
    Convierte outputs fermiónicos a strings quirales.
    
    Reglas:
    1. Todos los strings terminan en 1 (marker quiral)
    2. El patrón en el medio codifica el tipo de decaimiento
    3. La quiralidad (L/R/C) determina el encadenamiento
    """
    
    def __init__(self):
        """Inicializa tabla de codificación quiral"""
        
        # Mapeo: decaimiento → patrón quiral
        self.chiral_codebook = {
            # Decaimientos de TRITIO
            "T_beta_minus": {
                "description": "T → He-3 + e⁻ + ν̄ₑ (beta minus)",
                "fermionic_output": "1_1",  # electrón + antineutrino
                "chiral_patterns": [
                    {
                        "pattern": "1_1",  # Quiralidad simple
                        "chirality": Chirality.LEFT,
                        "middle_bits": "1",
                        "interpretation": "e⁻ (left-handed)"
                    },
                    {
                        "pattern": "1_0_1",  # Quiralidad media
                        "chirality": Chirality.CENTER,
                        "middle_bits": "1_0",
                        "interpretation": "e⁻ con rotación central"
                    },
                    {
                        "pattern": "1_0_0_1",  # Quiralidad derecha
                        "chirality": Chirality.RIGHT,
                        "middle_bits": "1_0_0",
                        "interpretation": "e⁻ (right-handed)"
                    },
                    {
                        "pattern": "1_1_0_0_1",  # Quiralidad compleja
                        "chirality": Chirality.LEFT,
                        "middle_bits": "1_1_0_0",
                        "interpretation": "e⁻ + ν̄ₑ (full chiral)"
                    },
                    {
                        "pattern": "1_0_1_0_1",  # Quiralidad oscilante
                        "chirality": Chirality.RIGHT,
                        "middle_bits": "1_0_1_0",
                        "interpretation": "e⁻ oscillating mode"
                    }
                ]
            },
            
            # Decaimientos de HELIO-3
            "He3_stable": {
                "description": "He-3 (stable, no decay)",
                "fermionic_output": "0_0",  # sin partículas
                "chiral_patterns": [
                    {
                        "pattern": "1",  # Quiralidad mínima
                        "chirality": Chirality.CENTER,
                        "middle_bits": "",
                        "interpretation": "vacuum (achiral)"
                    },
                    {
                        "pattern": "0_1",  # Quiralidad con ausencia
                        "chirality": Chirality.LEFT,
                        "middle_bits": "0",
                        "interpretation": "no electron (left)"
                    },
                    {
                        "pattern": "0_0_1",  # Quiralidad doble ausencia
                        "chirality": Chirality.RIGHT,
                        "middle_bits": "0_0",
                        "interpretation": "no particles (right)"
                    },
                    {
                        "pattern": "0_0_0_1",  # Quiralidad triple ausencia
                        "chirality": Chirality.CENTER,
                        "middle_bits": "0_0_0",
                        "interpretation": "vacuum deepest state"
                    }
                ]
            },
            
            # Decaimientos de DEUTERIO (hypothetical)
            "D_virtual_decay": {
                "description": "D (virtual decay, muy raro)",
                "fermionic_output": "0_1",  # solo antineutrino
                "chiral_patterns": [
                    {
                        "pattern": "0_1",
                        "chirality": Chirality.LEFT,
                        "middle_bits": "0",
                        "interpretation": "ν̄ₑ only (left-handed)"
                    },
                    {
                        "pattern": "0_0_1",
                        "chirality": Chirality.RIGHT,
                        "middle_bits": "0_0",
                        "interpretation": "ν̄ₑ shielded (right-handed)"
                    },
                    {
                        "pattern": "0_1_0_1",
                        "chirality": Chirality.CENTER,
                        "middle_bits": "0_1_0",
                        "interpretation": "ν̄ₑ oscillating (center)"
                    }
                ]
            }
        }
        
        # Asignación de fases topológicas
        self.topological_phases = {
            Chirality.LEFT: 0.5 * np.pi,      # π/2
            Chirality.RIGHT: 1.5 * np.pi,     # 3π/2
            Chirality.CENTER: np.pi            # π
        }
    
    def get_chiral_patterns(self, decay_type: str) -> List[Dict]:
        """Retorna todos los patrones quirales para un tipo de decaimiento"""
        if decay_type not in self.chiral_codebook:
            raise ValueError(f"Unknown decay type: {decay_type}")
        return self.chiral_codebook[decay_type]["chiral_patterns"]
    
    def encode_to_chiral(self, decay_type: str, pattern_index: int = 0) -> ChiralFermionicState:
        """
        Convierte un decaimiento a estado fermiónico quiral.
        
        Args:
            decay_type: Tipo de decaimiento (ej: "T_beta_minus")
            pattern_index: Índice del patrón quiral a usar (0, 1, 2, ...)
        
        Returns:
            ChiralFermionicState con estructura quiral completa
        """
        
        decay_info = self.chiral_codebook[decay_type]
        patterns = decay_info["chiral_patterns"]
        
        if pattern_index >= len(patterns):
            raise ValueError(f"Pattern index {pattern_index} out of range")
        
        selected_pattern = patterns[pattern_index]
        chiral_string = selected_pattern["pattern"]
        chirality = selected_pattern["chirality"]
        middle_bits = selected_pattern["middle_bits"]
        
        # Validar que termine en 1
        assert chiral_string.endswith("1"), f"Chiral string must end in 1: {chiral_string}"
        
        # Extraer bits como lista
        bit_pattern = [int(b) for b in chiral_string.split("_")]
        
        # Mapeo de decay_type a isótopos
        isotope_map = {
            "T_beta_minus": ("T", "He-3"),
            "He3_stable": ("He-3", "He-3"),
            "D_virtual_decay": ("D", "tritium-like")
        }
        
        isotope, product = isotope_map[decay_type]
        
        # Energía
        decay_energies = {
            "T_beta_minus": 0.01857,
            "He3_stable": 0.0,
            "D_virtual_decay": 0.000001  # Extremadamente raro
        }
        
        topological_phase = self.topological_phases[chirality]
        
        return ChiralFermionicState(
            isotope=isotope,
            product_isotope=product,
            fermionic_output=decay_info["fermionic_output"],
            chiral_string=chiral_string,
            chirality_type=chirality,
            bit_pattern=bit_pattern,
            middle_bits=middle_bits,
            chiral_marker=1,  # Siempre 1
            decay_energy_mev=decay_energies[decay_type],
            decay_channel=decay_info["description"],
            topological_phase=topological_phase
        )
    
    def all_chiral_variants(self, decay_type: str) -> List[ChiralFermionicState]:
        """Retorna todos los estados quirales para un decaimiento"""
        patterns = self.get_chiral_patterns(decay_type)
        return [
            self.encode_to_chiral(decay_type, i)
            for i in range(len(patterns))
        ]


# ============================================================================
# NIVEL 1: ANÁLISIS DE QUIRALIDAD
# ============================================================================

class ChiralAnalyzer:
    """
    Analiza propiedades quirales de los estados fermiónicos.
    """
    
    @staticmethod
    def get_handedness(chiral_string: str) -> str:
        """
        Determina si el patrón es left-handed, right-handed o achiral
        basado en la estructura de bits.
        
        Heurística:
        - Si tiene más 1s en medio → more "material" → más quiral
        - Si tiene más 0s en medio → more "emptiness" → menos quiral
        """
        
        # Remover terminación obligatoria (1)
        middle = chiral_string[:-2]  # Quitar último "_1"
        
        if not middle:  # Solo "1"
            return "ACHIRAL"
        
        bits = [int(b) for b in middle.split("_")]
        ones = sum(bits)
        zeros = len(bits) - ones
        
        if ones > zeros:
            return "LEFT-HANDED"
        elif zeros > ones:
            return "RIGHT-HANDED"
        else:
            return "CENTER (Balanced)"
    
    @staticmethod
    def calculate_chirality_index(chiral_string: str) -> float:
        """
        Calcula índice de quiralidad cuantitativo (-1 a +1).
        
        -1: Completamente right-handed (todos 0s)
        0: Achiral (balanceado)
        +1: Completamente left-handed (todos 1s)
        """
        
        middle = chiral_string[:-2]
        if not middle:
            return 0.0
        
        bits = [int(b) for b in middle.split("_")]
        ones = sum(bits)
        zeros = len(bits) - ones
        
        if len(bits) == 0:
            return 0.0
        
        chirality_index = (ones - zeros) / len(bits)
        return chirality_index
    
    @staticmethod
    def topological_winding_number(chiral_strings: List[str]) -> int:
        """
        Calcula número de enrollamiento topológico para una secuencia
        de estados quirales.
        
        Cuenta transiciones: 0→1 y 1→0 en la secuencia
        """
        
        winding = 0
        for i in range(len(chiral_strings) - 1):
            # Obtener bits finales (antes del "...1")
            current_bit = chiral_strings[i][-2]  # Penúltimo carácter
            next_bit = chiral_strings[i + 1][-2]
            
            if current_bit != next_bit:
                winding += 1
        
        return winding


# ============================================================================
# NIVEL 2: GENERADOR DE JSON QUIRAL
# ============================================================================

def generate_chiral_json(encoder: ChiralEncoder) -> str:
    """Genera JSON con todos los estados quirales"""
    
    chiral_system = {
        "chiral_fermionic_system": {
            "description": "Salida fermiónica quiralizada - todos terminan en 1",
            "chirality_definition": {
                "LEFT": {
                    "symbol": "L",
                    "meaning": "Left-handed (more 1s in pattern)",
                    "topological_phase": str(0.5 * np.pi) + " rad (π/2)"
                },
                "RIGHT": {
                    "symbol": "R",
                    "meaning": "Right-handed (more 0s in pattern)",
                    "topological_phase": str(1.5 * np.pi) + " rad (3π/2)"
                },
                "CENTER": {
                    "symbol": "C",
                    "meaning": "Achiral/Balanced",
                    "topological_phase": str(np.pi) + " rad (π)"
                }
            },
            "termination_rule": "All strings must end with ...1 (chiral marker)",
            "decay_processes": {}
        }
    }
    
    # Procesar cada tipo de decaimiento
    for decay_type in ["T_beta_minus", "He3_stable", "D_virtual_decay"]:
        variants = encoder.all_chiral_variants(decay_type)
        
        chiral_system["chiral_fermionic_system"]["decay_processes"][decay_type] = {
            "description": encoder.chiral_codebook[decay_type]["description"],
            "fermionic_output": encoder.chiral_codebook[decay_type]["fermionic_output"],
            "chiral_variants": []
        }
        
        for variant in variants:
            analyzer = ChiralAnalyzer()
            chirality_index = analyzer.calculate_chirality_index(variant.chiral_string)
            handedness = analyzer.get_handedness(variant.chiral_string)
            
            chiral_system["chiral_fermionic_system"]["decay_processes"][decay_type]["chiral_variants"].append({
                "chiral_string": variant.chiral_string,
                "chirality_type": variant.chirality_type.value,
                "middle_bits": variant.middle_bits if variant.middle_bits else "[empty]",
                "chiral_marker": variant.chiral_marker,
                "handedness": handedness,
                "chirality_index": float(chirality_index),
                "bit_pattern": variant.bit_pattern,
                "decay_channel": variant.decay_channel,
                "topological_phase_rad": float(variant.topological_phase),
                "topological_phase_fraction": f"{variant.topological_phase / np.pi:.2f}π"
            })
    
    return json.dumps(chiral_system, indent=2)


# ============================================================================
# NIVEL 3: DEMOSTRACIÓN
# ============================================================================

def demonstrate_chiral_system():
    """Demostración completa del sistema quiral"""
    
    print("=" * 90)
    print("SISTEMA FERMIÓNICO QUIRAL - TODOS TERMINAN EN 1")
    print("=" * 90)
    
    encoder = ChiralEncoder()
    analyzer = ChiralAnalyzer()
    
    # ────────────────────────────────────────────────────────────────────────
    # 1. MOSTRAR CODEBOOK QUIRAL
    # ────────────────────────────────────────────────────────────────────────
    print("\n1. CODEBOOK QUIRAL COMPLETO")
    print("-" * 90)
    
    for decay_type, decay_info in encoder.chiral_codebook.items():
        print(f"\n{decay_type}:")
        print(f"  Descripción: {decay_info['description']}")
        print(f"  Output fermiónico: {decay_info['fermionic_output']}")
        print(f"\n  Patrones quirales:")
        
        for i, pattern in enumerate(decay_info["chiral_patterns"]):
            print(f"    [{i}] {pattern['pattern']:20s} | Chirality: {pattern['chirality'].value:1s} | {pattern['interpretation']}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 2. CODIFICAR DECAIMIENTOS A QUIRALES
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("2. CODIFICACIÓN A ESTADOS QUIRALES")
    print("-" * 90)
    
    for decay_type in ["T_beta_minus", "He3_stable"]:
        print(f"\n{decay_type}:")
        variants = encoder.all_chiral_variants(decay_type)
        
        for variant in variants:
            chirality_idx = analyzer.calculate_chirality_index(variant.chiral_string)
            handedness = analyzer.get_handedness(variant.chiral_string)
            
            print(f"\n  Chiral string: {variant.chiral_string:20s}")
            print(f"    Chirality type: {variant.chirality_type.value}")
            print(f"    Handedness: {handedness}")
            print(f"    Chirality index: {chirality_idx:+.2f}")
            print(f"    Middle bits: {variant.middle_bits if variant.middle_bits else '[empty/vacuum]'}")
            print(f"    Topological phase: {variant.topological_phase:.4f} rad = {variant.topological_phase/np.pi:.2f}π")
            print(f"    Bit pattern: {variant.bit_pattern}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 3. ANÁLISIS DE QUIRALIDAD
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("3. ANÁLISIS DE QUIRALIDAD")
    print("-" * 90)
    
    test_strings = [
        "1_1",
        "1_0_1",
        "1_0_0_1",
        "1_1_0_0_1",
        "1_0_1_0_1",
        "1",
        "0_1",
        "0_0_1"
    ]
    
    print("\nChiral string analysis:")
    print(f"{'String':20s} | {'Handedness':20s} | {'Chirality Index':15s} | {'Bits':15s}")
    print("-" * 90)
    
    for s in test_strings:
        handedness = analyzer.get_handedness(s)
        chirality_idx = analyzer.calculate_chirality_index(s)
        middle = s[:-2] if "_1" in s else s
        
        print(f"{s:20s} | {handedness:20s} | {chirality_idx:+6.2f}        | {middle:15s}")
    
    # ────────────────────────────────────────────────────────────────────────
    # 4. NÚMERO DE ENROLLAMIENTO TOPOLÓGICO
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("4. NÚMERO DE ENROLLAMIENTO TOPOLÓGICO")
    print("-" * 90)
    
    # Secuencia de transiciones
    sequence = encoder.all_chiral_variants("T_beta_minus")
    chiral_strings = [v.chiral_string for v in sequence]
    
    print(f"\nSecuencia de estados quirales (T → He-3):")
    for i, s in enumerate(chiral_strings):
        print(f"  [{i}] {s}")
    
    winding = analyzer.topological_winding_number(chiral_strings)
    print(f"\nNúmero de enrollamiento topológico: {winding}")
    print(f"Interpretación: {winding} transiciones entre quiralidades en la secuencia")
    
    # ────────────────────────────────────────────────────────────────────────
    # 5. MOSTRAR JSON QUIRAL
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("5. ESTRUCTURA JSON QUIRAL")
    print("-" * 90)
    
    json_output = generate_chiral_json(encoder)
    print(json_output[:2000] + "\n... [truncated for display]")
    
    # ────────────────────────────────────────────────────────────────────────
    # 6. TABLA COMPARATIVA
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("6. TABLA COMPARATIVA: ORIGINAL vs QUIRAL")
    print("-" * 90)
    
    print(f"\n{'Decaimiento':<20s} | {'Original':<20s} | {'Chiral Variants':<50s}")
    print("-" * 90)
    
    original_outputs = {
        "T_beta_minus": "1_1",
        "He3_stable": "0_0",
        "D_virtual": "0_1"
    }
    
    for decay_type, original in original_outputs.items():
        encoder_key = decay_type if decay_type in encoder.chiral_codebook else "T_beta_minus"
        variants = encoder.all_chiral_variants(encoder_key)
        chiral_strs = [v.chiral_string for v in variants]
        
        print(f"{decay_type:<20s} | {original:<20s} | {', '.join(chiral_strs)}")
    
    return json_output


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    json_output = demonstrate_chiral_system()
    
    # Guardar JSON
    with open('data/chiral_fermionic_system.json', 'w') as f:
        f.write(json_output)
    
    print("\n\n✓ JSON quiral guardado en: data/chiral_fermionic_system.json")
