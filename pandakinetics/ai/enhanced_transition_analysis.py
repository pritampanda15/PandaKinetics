# =============================================================================
# pandakinetics/ai/enhanced_transition_analysis.py - Enhanced Transition State Analysis
# =============================================================================

"""
Enhanced transition state analysis using Boltz-2 inspired approaches

This module provides advanced analysis of transition states and kinetic pathways
using machine learning approaches inspired by Boltz-2's structural predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from pathlib import Path
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Some features will be limited.")


class TransitionStateAnalyzer:
    """
    Enhanced transition state analyzer using ML approaches
    
    Provides Boltz-2 inspired analysis of:
    - Transition state stability
    - Pathway feasibility
    - Energy barrier predictions
    - Residence time estimates
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"TransitionStateAnalyzer initialized on {self.device}")
    
    def analyze_transition_states(
        self,
        poses: List[Dict[str, Any]],
        ligand_smiles: str,
        boltz_results: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of transition states
        
        Args:
            poses: List of molecular poses/conformations
            ligand_smiles: Ligand SMILES string
            boltz_results: Optional Boltz-2 affinity results
            
        Returns:
            Comprehensive analysis results
        """
        
        logger.info(f"Analyzing {len(poses)} transition states...")
        
        # Analyze individual states
        state_analyses = []
        for i, pose in enumerate(poses):
            analysis = self._analyze_single_state(pose, i, ligand_smiles)
            state_analyses.append(analysis)
        
        # Pathway analysis
        pathway_analysis = self._analyze_transition_pathways(state_analyses, boltz_results)
        
        # Energy landscape analysis
        energy_analysis = self._analyze_energy_landscape(state_analyses)
        
        # Kinetic predictions
        kinetic_predictions = self._predict_enhanced_kinetics(
            state_analyses, pathway_analysis, boltz_results
        )
        
        # Compile comprehensive results
        comprehensive_analysis = {
            'individual_states': state_analyses,
            'pathway_analysis': pathway_analysis,
            'energy_landscape': energy_analysis,
            'kinetic_predictions': kinetic_predictions,
            'summary_metrics': self._calculate_summary_metrics(
                state_analyses, pathway_analysis, energy_analysis
            )
        }
        
        logger.info("✅ Transition state analysis completed")
        return comprehensive_analysis
    
    def _analyze_single_state(
        self,
        pose: Dict[str, Any],
        state_id: int,
        ligand_smiles: str
    ) -> Dict[str, Any]:
        """Analyze a single transition state"""
        
        coords = pose.get('coordinates', [])
        energy = pose.get('energy', 0.0)
        
        # Geometric analysis
        geometric_features = self._extract_geometric_features(coords)
        
        # Chemical analysis
        chemical_features = self._extract_chemical_features(ligand_smiles)
        
        # Stability prediction
        stability_score = self._predict_state_stability(
            geometric_features, chemical_features, energy
        )
        
        # Druggability assessment
        druggability = self._assess_druggability(chemical_features)
        
        return {
            'state_id': state_id,
            'energy': energy,
            'stability_score': stability_score,
            'druggability': druggability,
            'geometric_features': geometric_features,
            'chemical_features': chemical_features,
            'coordinates': coords
        }
    
    def _extract_geometric_features(self, coordinates: List) -> Dict[str, float]:
        """Extract geometric features from coordinates"""
        
        if not coordinates or len(coordinates) == 0:
            return {
                'compactness': 0.0,
                'asphericity': 0.0,
                'radius_of_gyration': 0.0,
                'coordination_number': 0.0
            }
        
        coords = np.array(coordinates)
        center = np.mean(coords, axis=0)
        
        # Radius of gyration
        distances = np.linalg.norm(coords - center, axis=1)
        rg = np.sqrt(np.mean(distances**2))
        
        # Compactness (inverse of spread)
        max_distance = np.max(distances)
        compactness = rg / max_distance if max_distance > 0 else 0.0
        
        # Asphericity (shape anisotropy)
        cov_matrix = np.cov(coords.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        
        if eigenvals[0] > 0:
            asphericity = 1.5 * ((eigenvals[0] - eigenvals[1])**2 + 
                               (eigenvals[0] - eigenvals[2])**2 + 
                               (eigenvals[1] - eigenvals[2])**2) / (eigenvals.sum()**2)
        else:
            asphericity = 0.0
        
        # Coordination number (average number of close neighbors)
        coordination_number = self._calculate_coordination_number(coords)
        
        return {
            'compactness': float(compactness),
            'asphericity': float(asphericity),
            'radius_of_gyration': float(rg),
            'coordination_number': float(coordination_number)
        }
    
    def _calculate_coordination_number(self, coords: np.ndarray, cutoff: float = 3.0) -> float:
        """Calculate average coordination number"""
        
        n_atoms = len(coords)
        if n_atoms < 2:
            return 0.0
        
        total_neighbors = 0
        for i in range(n_atoms):
            neighbors = 0
            for j in range(n_atoms):
                if i != j:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < cutoff:
                        neighbors += 1
            total_neighbors += neighbors
        
        return total_neighbors / n_atoms
    
    def _extract_chemical_features(self, ligand_smiles: str) -> Dict[str, float]:
        """Extract chemical features from SMILES"""
        
        if not RDKIT_AVAILABLE:
            return self._get_dummy_chemical_features()
        
        try:
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                return self._get_dummy_chemical_features()
            
            features = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'lipinski_violations': self._count_lipinski_violations(mol)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract chemical features: {e}")
            return self._get_dummy_chemical_features()
    
    def _get_dummy_chemical_features(self) -> Dict[str, float]:
        """Get dummy chemical features when RDKit is not available"""
        return {
            'molecular_weight': 300.0,
            'logp': 2.0,
            'hbd': 2,
            'hba': 4,
            'tpsa': 60.0,
            'rotatable_bonds': 5,
            'aromatic_rings': 1,
            'heavy_atoms': 20,
            'formal_charge': 0,
            'lipinski_violations': 0
        }
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule violations"""
        
        violations = 0
        
        # Rule of 5 violations
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations
    
    def _predict_state_stability(
        self,
        geometric_features: Dict[str, float],
        chemical_features: Dict[str, float],
        energy: float
    ) -> float:
        """Predict stability of transition state"""
        
        # Simple stability prediction based on features
        # Higher compactness and lower energy typically indicate more stable states
        
        compactness = geometric_features.get('compactness', 0.0)
        coordination = geometric_features.get('coordination_number', 0.0)
        
        # Normalize energy (assuming typical range -15 to 0 kcal/mol)
        normalized_energy = min(1.0, max(0.0, (energy + 15) / 15))
        
        # Combine features for stability score
        stability_score = (
            0.3 * compactness +
            0.2 * (coordination / 6.0) +  # Normalize coordination
            0.5 * (1.0 - normalized_energy)  # Lower energy = higher stability
        )
        
        return min(1.0, max(0.0, stability_score))
    
    def _assess_druggability(self, chemical_features: Dict[str, float]) -> Dict[str, Any]:
        """Assess druggability of the molecule"""
        
        # Lipinski Rule of 5 assessment
        lipinski_score = 5 - chemical_features.get('lipinski_violations', 5)
        
        # Additional drug-like properties
        mw = chemical_features.get('molecular_weight', 0)
        logp = chemical_features.get('logp', 0)
        tpsa = chemical_features.get('tpsa', 0)
        
        # Drug-likeness score
        drug_likeness = 0.0
        if 150 <= mw <= 500:
            drug_likeness += 0.25
        if -2 <= logp <= 5:
            drug_likeness += 0.25
        if 20 <= tpsa <= 130:
            drug_likeness += 0.25
        if chemical_features.get('rotatable_bonds', 0) <= 10:
            drug_likeness += 0.25
        
        return {
            'lipinski_score': lipinski_score,
            'drug_likeness_score': drug_likeness,
            'is_drug_like': drug_likeness >= 0.75,
            'lipinski_compliant': lipinski_score >= 4
        }
    
    def _analyze_transition_pathways(
        self,
        state_analyses: List[Dict[str, Any]],
        boltz_results: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze transition pathways between states"""
        
        n_states = len(state_analyses)
        
        # Energy barriers between states
        energy_matrix = np.zeros((n_states, n_states))
        stability_matrix = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    energy_i = state_analyses[i]['energy']
                    energy_j = state_analyses[j]['energy']
                    stability_i = state_analyses[i]['stability_score']
                    stability_j = state_analyses[j]['stability_score']
                    
                    # Energy barrier approximation
                    energy_barrier = abs(energy_j - energy_i) + 2.0  # Add transition cost
                    energy_matrix[i, j] = energy_barrier
                    
                    # Stability-based transition probability
                    stability_matrix[i, j] = (stability_i + stability_j) / 2.0
        
        # Identify most likely pathways
        most_stable_states = sorted(
            range(n_states), 
            key=lambda i: state_analyses[i]['stability_score'], 
            reverse=True
        )[:3]
        
        # Dominant pathway analysis
        dominant_pathways = self._find_dominant_pathways(
            energy_matrix, stability_matrix, most_stable_states
        )
        
        return {
            'energy_barriers': energy_matrix.tolist(),
            'stability_transitions': stability_matrix.tolist(),
            'most_stable_states': most_stable_states,
            'dominant_pathways': dominant_pathways,
            'pathway_diversity': self._calculate_pathway_diversity(energy_matrix),
            'boltz_enhanced': boltz_results is not None
        }
    
    def _find_dominant_pathways(
        self,
        energy_matrix: np.ndarray,
        stability_matrix: np.ndarray,
        stable_states: List[int]
    ) -> List[Dict[str, Any]]:
        """Find dominant transition pathways"""
        
        pathways = []
        
        for i, start_state in enumerate(stable_states[:2]):  # Top 2 stable states
            for end_state in stable_states[i+1:]:
                # Direct pathway
                direct_barrier = energy_matrix[start_state, end_state]
                direct_stability = stability_matrix[start_state, end_state]
                
                pathway = {
                    'start_state': start_state,
                    'end_state': end_state,
                    'type': 'direct',
                    'energy_barrier': float(direct_barrier),
                    'stability_score': float(direct_stability),
                    'pathway_length': 1
                }
                
                pathways.append(pathway)
        
        # Sort by energy barrier (lower is better)
        pathways.sort(key=lambda p: p['energy_barrier'])
        
        return pathways[:5]  # Return top 5 pathways
    
    def _calculate_pathway_diversity(self, energy_matrix: np.ndarray) -> float:
        """Calculate diversity of transition pathways"""
        
        # Standard deviation of energy barriers
        barriers = energy_matrix[energy_matrix > 0]  # Exclude diagonal
        
        if len(barriers) > 0:
            diversity = np.std(barriers) / np.mean(barriers)
        else:
            diversity = 0.0
        
        return float(diversity)
    
    def _analyze_energy_landscape(self, state_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the energy landscape of transition states"""
        
        energies = [state['energy'] for state in state_analyses]
        stabilities = [state['stability_score'] for state in state_analyses]
        
        # Energy statistics
        energy_stats = {
            'min_energy': float(np.min(energies)),
            'max_energy': float(np.max(energies)),
            'mean_energy': float(np.mean(energies)),
            'energy_range': float(np.max(energies) - np.min(energies)),
            'energy_std': float(np.std(energies))
        }
        
        # Stability statistics
        stability_stats = {
            'mean_stability': float(np.mean(stabilities)),
            'max_stability': float(np.max(stabilities)),
            'min_stability': float(np.min(stabilities)),
            'stability_std': float(np.std(stabilities))
        }
        
        # Energy landscape characteristics
        landscape_features = {
            'ruggedness': float(np.std(energies) / (np.max(energies) - np.min(energies) + 1e-6)),
            'funnel_like': float(1.0 - np.corrcoef(energies, stabilities)[0, 1]),
            'multi_modal': len(self._find_local_minima(energies)) > 1
        }
        
        return {
            'energy_statistics': energy_stats,
            'stability_statistics': stability_stats,
            'landscape_features': landscape_features,
            'global_minimum_state': int(np.argmin(energies)),
            'most_stable_state': int(np.argmax(stabilities))
        }
    
    def _find_local_minima(self, energies: List[float], window: int = 3) -> List[int]:
        """Find local minima in energy profile"""
        
        minima = []
        n = len(energies)
        
        for i in range(window, n - window):
            is_minimum = True
            
            # Check if current point is lower than neighbors
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j != i and energies[j] <= energies[i]:
                    is_minimum = False
                    break
            
            if is_minimum:
                minima.append(i)
        
        return minima
    
    def _predict_enhanced_kinetics(
        self,
        state_analyses: List[Dict[str, Any]],
        pathway_analysis: Dict[str, Any],
        boltz_results: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Predict enhanced kinetic parameters using ML analysis"""
        
        # Extract features for kinetic prediction
        stability_scores = [state['stability_score'] for state in state_analyses]
        energies = [state['energy'] for state in state_analyses]
        
        # Basic kinetic predictions
        mean_stability = np.mean(stability_scores)
        energy_range = np.max(energies) - np.min(energies)
        
        # Predict residence time based on stability and energy landscape
        predicted_residence_time = self._predict_residence_time(
            mean_stability, energy_range, pathway_analysis
        )
        
        # Predict association/dissociation rates
        predicted_kon, predicted_koff = self._predict_rate_constants(
            mean_stability, energy_range, predicted_residence_time
        )
        
        predictions = {
            'predicted_kon': predicted_kon,
            'predicted_koff': predicted_koff,
            'predicted_kd': predicted_koff / predicted_kon,
            'predicted_residence_time': predicted_residence_time,
            'stability_based_affinity': mean_stability,
            'energy_landscape_score': 1.0 / (1.0 + energy_range / 10.0)
        }
        
        # Enhance with Boltz-2 results if available
        if boltz_results:
            predictions['boltz_enhanced'] = True
            predictions['boltz_confidence'] = boltz_results.get('confidence', 0.5)
            
            # Weight predictions by Boltz confidence
            confidence = boltz_results.get('confidence', 0.5)
            
            if 'residence_time_s' in boltz_results:
                predictions['enhanced_residence_time'] = (
                    confidence * boltz_results['residence_time_s'] +
                    (1 - confidence) * predicted_residence_time
                )
            
            if 'kd_M' in boltz_results:
                predictions['enhanced_kd'] = (
                    confidence * boltz_results['kd_M'] +
                    (1 - confidence) * predictions['predicted_kd']
                )
        
        return predictions
    
    def _predict_residence_time(
        self,
        mean_stability: float,
        energy_range: float,
        pathway_analysis: Dict[str, Any]
    ) -> float:
        """Predict residence time from structural analysis"""
        
        # Higher stability and energy barriers typically lead to longer residence times
        base_time = 1.0  # 1 second base
        
        # Stability factor (0.1 to 10x modifier)
        stability_factor = 0.1 + 9.9 * mean_stability
        
        # Energy barrier factor
        energy_factor = 1.0 + energy_range / 5.0
        
        # Pathway complexity factor
        pathway_diversity = pathway_analysis.get('pathway_diversity', 1.0)
        complexity_factor = 1.0 + pathway_diversity
        
        predicted_time = base_time * stability_factor * energy_factor * complexity_factor
        
        return float(predicted_time)
    
    def _predict_rate_constants(
        self,
        mean_stability: float,
        energy_range: float,
        residence_time: float
    ) -> Tuple[float, float]:
        """Predict kon and koff from structural features"""
        
        # koff from residence time
        koff = 1.0 / residence_time
        
        # kon prediction based on structural features
        # Higher stability typically correlates with higher association rates
        base_kon = 1e6  # M^-1 s^-1 base rate
        
        stability_factor = 0.1 + 9.9 * mean_stability
        energy_factor = 1.0 / (1.0 + energy_range / 10.0)
        
        kon = base_kon * stability_factor * energy_factor
        
        return float(kon), float(koff)
    
    def _calculate_summary_metrics(
        self,
        state_analyses: List[Dict[str, Any]],
        pathway_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate summary metrics for the analysis"""
        
        # Overall druggability
        druggability_scores = [
            state['druggability']['drug_likeness_score'] 
            for state in state_analyses
        ]
        
        mean_druggability = np.mean(druggability_scores)
        
        # Structural diversity
        stability_scores = [state['stability_score'] for state in state_analyses]
        structural_diversity = np.std(stability_scores)
        
        # Energy landscape quality
        energy_range = energy_analysis['energy_statistics']['energy_range']
        landscape_quality = 1.0 / (1.0 + energy_range / 20.0)  # Better with lower range
        
        # Pathway accessibility
        dominant_pathways = pathway_analysis.get('dominant_pathways', [])
        pathway_accessibility = len(dominant_pathways) / 5.0  # Normalize to 0-1
        
        return {
            'overall_druggability': float(mean_druggability),
            'structural_diversity': float(structural_diversity),
            'landscape_quality': float(landscape_quality),
            'pathway_accessibility': float(pathway_accessibility),
            'analysis_confidence': float(np.mean([
                mean_druggability,
                1.0 - structural_diversity,  # Lower diversity = higher confidence
                landscape_quality,
                pathway_accessibility
            ])),
            'num_states_analyzed': len(state_analyses)
        }
    
    def export_analysis(
        self,
        analysis_results: Dict[str, Any],
        output_path: Path,
        format: str = 'json'
    ):
        """Export analysis results to file"""
        
        if format == 'json':
            output_file = output_path / 'enhanced_transition_analysis.json'
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"Analysis exported to {output_file}")
        
        else:
            logger.warning(f"Export format '{format}' not supported")


# Export main classes
__all__ = [
    'TransitionStateAnalyzer'
]