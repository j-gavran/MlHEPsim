"""Invariant mass calculation utilities for Drell-Yan analysis."""

import numpy as np


def calculate_dimuon_invariant_mass(data, pt1_idx, pt2_idx, eta1_idx, eta2_idx, phi1_idx, phi2_idx):
    """
    Calculate dimuon invariant mass from kinematics.
    
    Formula: M^2 = 2*pT1*pT2*(cosh(eta1 - eta2) - cos(phi1 - phi2))
    
    Parameters
    ----------
    data : np.ndarray
        Data array containing kinematic variables
    pt1_idx, pt2_idx : int
        Indices for leading and subleading muon pT
    eta1_idx, eta2_idx : int
        Indices for leading and subleading muon eta
    phi1_idx, phi2_idx : int
        Indices for leading and subleading muon phi
        
    Returns
    -------
    np.ndarray
        Invariant mass values in GeV
    """
    pt1 = data[:, pt1_idx]
    pt2 = data[:, pt2_idx]
    eta1 = data[:, eta1_idx]
    eta2 = data[:, eta2_idx]
    phi1 = data[:, phi1_idx]
    phi2 = data[:, phi2_idx]
    
    m_squared = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
    # Avoid negative values due to numerical precision
    m_squared = np.maximum(m_squared, 0)
    return np.sqrt(m_squared)


def compute_masses_for_dataset(data, selected_features, variables, calculate_mass_fn):
    """
    Compute invariant masses for a dataset.
    
    Parameters
    ----------
    data : np.ndarray
        Data array with kinematic features
    selected_features : list
        List of selected feature names in order
    variables : dict
        Variables configuration dictionary
    calculate_mass_fn : callable
        Function to calculate invariant mass
        
    Returns
    -------
    np.ndarray
        Invariant masses in GeV
    """
    # Get feature indices
    pt_1_idx = selected_features.index('Muons_Pos_PT')
    pt_2_idx = selected_features.index('Muons_Neg_PT')
    eta_1_idx = selected_features.index('Muons_Pos_Eta')
    eta_2_idx = selected_features.index('Muons_Neg_Eta')
    phi_1_idx = selected_features.index('Muons_Pos_Phi')
    phi_2_idx = selected_features.index('Muons_Neg_Phi')
    
    return calculate_mass_fn(
        data, pt_1_idx, pt_2_idx,
        eta_1_idx, eta_2_idx, phi_1_idx, phi_2_idx
    )
