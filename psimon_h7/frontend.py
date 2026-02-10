"""
PSimon Framework: Streamlit Professional Dashboard
================================================================================
A premium, touch-sensitive, and reactive interface for the PSimon Framework, 
integrating quantum dynamics, nuclear states, and cognitive metriplectic evolution.

Author: Jacobo Tlacaelel Mina Rodríguez
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any

# Framework Imports
from physics.isotopic_beta_fermion_system import ISOTOPES_NUCLEAR_DB
from models.cognitive_engine import CognitiveEngine
from core.psimon_framework import PSimon

# Page Config
st.set_page_config(
    page_title="PSimon Framework Console",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Mandato Metriplético Aesthetics)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚛️ PSimon Framework")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Control Layer", ["Isotope Explorer", "Cognitive Engine", "Beta Decay Simulation"])

st.sidebar.markdown("---")
st.sidebar.info("Modulated by Golden Operator $O_n$")

# --- Main Logic ---

if mode == "Isotope Explorer":
    st.title("Isotope Nuclear Mapping")
    st.write("Exploration of nuclear states and binding energies using Metriplectic metrics.")

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_iso = st.selectbox("Select Isotope", list(ISOTOPES_NUCLEAR_DB.keys()))
        data = ISOTOPES_NUCLEAR_DB[selected_iso]
        
        st.metric("Mass (u)", f"{data.mass_u:.5f}")
        st.metric("Binding Energy", f"{data.binding_energy_mev:.3f} MeV")
        st.markdown(f"**Nuclear String:** `{data.nuclear_string}`")
        st.markdown(f"**Structure (Z/N):** {data.Z}p, {data.N}n")

    with col2:
        # Radial visualization of nucleons
        phi_vals = np.linspace(0, 2*np.pi, 100)
        fig = go.Figure()
        
        # Protons
        for i in range(data.Z):
            r = 0.5 + 0.1 * i
            fig.add_trace(go.Scatterpolar(
                r=[r], theta=[i * 360/data.Z],
                mode='markers', name=f"Proton {i+1}",
                marker=dict(size=14, color='#ff0055', line=dict(width=1.5, color='white'))
            ))
            
        # Neutrons
        for i in range(data.N):
            r = 0.8 + 0.1 * i
            fig.add_trace(go.Scatterpolar(
                r=[r], theta=[(i+0.5) * 360/data.N] if data.N > 0 else [0],
                mode='markers', name=f"Neutron {i+1}",
marker=dict(size=12, color='#00d4ff')
            ))
            
        fig.update_layout(
            template="plotly_dark",
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)),
            margin=dict(r=10, t=10, l=10, b=10)
        )
        st.plotly_chart(fig, width='stretch')

elif mode == "Cognitive Engine":
    st.title("Cognitive Metriplectic Engine")
    st.write("Real-time observation of recursive learning and temporal coherence breaking.")

    if st.button("🚀 Trigger Synchronous Learning Cycle"):
        engine = CognitiveEngine(state_dim=8)
        
        # Simulated data stream
        data_stream = [np.random.randn(8) + 1j * np.random.randn(8) for _ in range(5)]
        for d in data_stream: d /= np.linalg.norm(d)
        
        with st.status("Solving Recursive Dynamics...", expanded=True) as status:
            final_state, logs = engine.learn(data_stream, iterations=200)
            status.update(label="System Reached Attractor State ✅", state="complete")
            
        df_logs = pd.DataFrame(logs)
        
        # Plots
        fig_on = px.line(df_logs, x="iteration", y="o_n", title="Golden Operator $O_n$ Modulation",
                         color_discrete_sequence=["#00d4ff"])
        fig_on.update_layout(template="plotly_dark")
        st.plotly_chart(fig_on, width='stretch')
        
        fig_coh = px.area(df_logs, x="iteration", y="coherence", title="Cognitive Coherence Levels",
                          color_discrete_sequence=["#ff0055"])
        fig_coh.update_layout(template="plotly_dark")
        st.plotly_chart(fig_coh, width='stretch')

elif mode == "Beta Decay Simulation":
    st.title("Fermionic Beta Decay Simulation")
    st.write("Dynamic mapping of n → p + e⁻ + ν̄ₑ using the PSimon Oracle.")
    
    st.info("Continuous learning mode active. Real-time Chiral Coherence Analysis.")

    # Chiral Illustration Container
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Nuclear Transition")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Beta_negative_decay.svg/1200px-Beta_negative_decay.svg.png", 
                 caption="n → p + e⁻ + ν̄ₑ (Standard Model)", width=400)
    
    with col2:
        st.subheader("Chiral Correspondence (Analogía Rigurosa)")
        st.markdown("""
        > **Carvone Isomerism Example:**
        > - **(R)-(-)-Carvone**: Spearmint smell (Left-handed coherence)
        > - **(S)-(+)-Carvone**: Caraway smell (Right-handed coherence)
        """)
        st.info("The PSimon Oracle maps nuclear parities to these topological structures.")

    if st.button("🚀 Trigger Decay Cascade"):
        from physics.chiral_fermionic_system import ChiralEncoder, ChiralAnalyzer
        
        with st.status("Solving Parity-Violating Dynamics...", expanded=True) as status:
            st.write("Initializing Fock-Chiral basis...")
            time.sleep(0.5)
            st.write("Applying Metriplex Oracle (Dissipative Relaxation)...")
            time.sleep(0.8)
            st.write("Discovering Hidden Symmetry with Simon-H7...")
            
            # Real simulation trigger
            encoder = ChiralEncoder()
            # Simulate a "T_beta_minus" decay
            variant = encoder.encode_to_chiral("T_beta_minus", pattern_index=np.random.randint(0, 3))
            
            time.sleep(0.5)
            status.update(label="Decay String Localized!", state="complete")
        
        # Display Results
        st.success(f"Output localized: `{variant.chiral_string}`")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Chirality Type", variant.chirality_type.value)
            st.metric("Topological Phase", f"{variant.topological_phase/np.pi:.2f}π")
            
        with res_col2:
            idx = ChiralAnalyzer.calculate_chirality_index(variant.chiral_string)
            st.metric("Chirality Index", f"{idx:+.2f}")
            status_text = ChiralAnalyzer.get_handedness(variant.chiral_string)
            st.markdown(f"**Coherence State:** `{status_text}`")
            
        # Carvone Analogy Output
        if "LEFT" in status_text or variant.chirality_type.value == "L":
            st.warning("🥬 **Coherence Detected**: Similar to (R)-(-)-Carvone (Spearmint Topology)")
        elif "RIGHT" in status_text or variant.chirality_type.value == "R":
            st.warning("🥐 **Coherence Detected**: Similar to (S)-(+)-Carvone (Caraway Topology)")
        else:
            st.info("⚪ **Coherence Detected**: Vacuum/Achiral State")

st.sidebar.markdown("---")
st.sidebar.caption("PSimon v4.1.0-metriplectic")
