"""Muon PT negative calculation utilities for Drell-Yan analysis."""

import numpy as np


def calculate_pt_negative_muon(data, pt1_idx, eta1_idx, eta2_idx, phi1_idx, phi2_idx, mass1_idx):
    """
    Calculate PT of the negative muon from kinematics.
    
    Formula: M^2 = 2*pT1*pT2*(cosh(eta1 - eta2) - cos(phi1 - phi2))
    Rewritten: pT2 = (M^2 / (2*pT1*(cosh(eta1 - eta2) - cos(phi1 - phi2)))) if M^2 > 0 else 0
    Parameters
    ----------
    data : np.ndarray
        Data array containing kinematic variables
    pt1_idx,: int
        Index for positive muon pT
    eta1_idx, eta2_idx : int
        Indices for leading and subleading muon eta
    phi1_idx, phi2_idx : int
        Indices for leading and subleading muon phi
    mass1_idx : int
        Index for invariant mass
    Returns
    -------
    np.ndarray
        PT of negative muon in GeV
    """
    pt1 = data[:, pt1_idx]
    eta1 = data[:, eta1_idx]
    eta2 = data[:, eta2_idx]
    phi1 = data[:, phi1_idx]
    phi2 = data[:, phi2_idx]
    mass1 = data[:, mass1_idx]
    
    pt2 = (mass1**2) / (2 * pt1 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)))
    pt2 = np.where(pt2 > 0, pt2, 0)
    return pt2

def compute_pt_negative_for_dataset(data, selected_features, variables, calculate_pt_fn):
    """
    Compute PT of negative muon for a dataset.
    
    Parameters
    ----------
    data : np.ndarray
        Data array with kinematic features
    selected_features : list
        List of selected feature names in order
    variables : dict
        Variables configuration dictionary
    calculate_pt_fn : callable
        Function to calculate PT of negative muon
        
    Returns
    -------
    np.ndarray
        PT of negative muon in GeV
    """
    # Get feature indices
    pt_pos_idx = selected_features.index('Muons_Pos_PT')
    eta_lead_idx = selected_features.index('Muons_Pos_Eta')
    eta_sub_idx = selected_features.index('Muons_Neg_Eta')
    phi_lead_idx = selected_features.index('Muons_Pos_Phi')
    phi_sub_idx = selected_features.index('Muons_Neg_Phi')
    mass1_idx = selected_features.index('Muons_Minv_MuMu')

    return calculate_pt_fn(
        data, pt_pos_idx, eta_lead_idx, eta_sub_idx, phi_lead_idx, phi_sub_idx, mass1_idx
    )
