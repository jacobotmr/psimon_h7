"""
Algolia Indexer Model: Nuclear Fermion Edition
================================================================================
This component handles data indexing using Algolia, adapted for the PSimon 
Framework to index nuclear fermion data structures (Isotopes, Mass, Energy).

Author: Jacobo Tlacaelel Mina Rodríguez
"""

import time
import requests
import os
from typing import Dict, List, Any, Optional
from algoliasearch.search.client import SearchClientSync

# Import from framework physics
try:
    from physics.isotopic_beta_fermion_system import ISOTOPES_NUCLEAR_DB
except ImportError:
    ISOTOPES_NUCLEAR_DB = {}

class AlgoliaIndexer:
    """
    Search and Indexing model using Algolia.
    Maps data storage to a 'Metric' dissipative flow in information space.
    """
    
    def __init__(self, app_id: str, api_key: str):
        self.app_id = app_id
        self.api_key = api_key
        self.client = SearchClientSync(app_id, api_key)
        
    def fetch_nuclear_data(self) -> List[Dict[str, Any]]:
        """Extract nuclear data records from the framework's database."""
        records = []
        for key, state in ISOTOPES_NUCLEAR_DB.items():
            # Convert dataclass to dict and add objectID for Algolia
            record = {
                "objectID": state.name,
                "name": state.name,
                "Z": state.Z,
                "N": state.N,
                "A": state.A,
                "nuclear_string": state.nuclear_string,
                "mass_u": state.mass_u,
                "binding_energy_mev": state.binding_energy_mev,
                "category": "Nuclear Isotope"
            }
            records.append(record)
        return records
        
    def index_data(self, index_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index records into Algolia.
        Includes execution time logging for the 'Metric' process.
        """
        start_time = time.time()
        
        print(f"[Algolia] Indexing {len(records)} nuclear records into '{index_name}'...")
        
        # Save records
        result = self.client.save_objects(
            index_name=index_name,
            objects=records,
        )
        
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "index": index_name,
            "count": len(records),
            "execution_time_sec": elapsed,
            "algolia_response": result
        }

def demonstrate_algolia():
    """Demostración del modelo de búsqueda (Isótopos Nucleares)"""
    print("\n" + "=" * 80)
    print("ALGOLIA SEARCH MODEL: NUCLEAR FERMIONS")
    print("=" * 80)
    
    # API Keys provided by environment variables
    APP_ID = os.getenv("ALGOLIA_APP_ID", "2JA5JZEL4C")
    API_KEY = os.getenv("ALGOLIA_API_KEY", "45a52e90d076c0286030e9eeb3afeae6")
    INDEX_NAME = os.getenv("ALGOLIA_INDEX_NAME", "nuclear_fermion_index")
    
    try:
        indexer = AlgoliaIndexer(APP_ID, API_KEY)
        
        print("\n[Extracting Fermionic Data from Physics Engine...]")
        records = indexer.fetch_nuclear_data()
        print(f"✓ Retrieved {len(records)} isotopes: {', '.join([r['name'] for r in records])}")
        
        print("\n[Indexing data...]")
        result = indexer.index_data(INDEX_NAME, records)
        
        print(f"\n✓ Status: {result['status']}")
        print(f"✓ Managed records: {result['count']}")
        print(f"✓ Metric Execution Time: {result['execution_time_sec']:.4f}s")
        
    except Exception as e:
        print(f"\n✗ Error in Algolia process: {e}")
        print("Note: Verify your internet connection and API credentials.")

if __name__ == "__main__":
    demonstrate_algolia()
