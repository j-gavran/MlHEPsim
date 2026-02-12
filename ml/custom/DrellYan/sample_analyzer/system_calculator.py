""" System-level kinematic variable calculations for Drell-Yan analysis. """
import numpy as np


def calculate_system_variables(pt1, eta1, phi1, pt2, eta2, phi2):
    """
    Calculate system-level kinematic variables (PT, eta, phi, rapidity) for a two-particle system.
    
    Parameters
    ----------
    pt1, eta1, phi1 : np.ndarray
        Transverse momentum, pseudorapidity, and azimuthal angle of particle 1 (positive muon)
    pt2, eta2, phi2 : np.ndarray
        Transverse momentum, pseudorapidity, and azimuthal angle of particle 2 (negative muon)
        
    Returns
    -------
    Z_pt, Z_eta, Z_phi, Z_Y, cos_theta_star: np.ndarray
        Transverse momentum, pseudorapidity, azimuthal angle, rapidity, and Collins-Soper angle of the system
    """
    # Convert to Cartesian
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)

    # System momentum
    Z_pt = np.sqrt((px1 + px2)**2 + (py1 + py2)**2)
    Z_phi = np.arctan2(py1 + py2, px1 + px2)

    # System pz
    pz1 = pt1 * np.sinh(eta1)
    pz2 = pt2 * np.sinh(eta2)
    pz_sys = pz1 + pz2

    # System eta
    Z_eta = np.arcsinh(pz_sys / Z_pt)

    # For rapidity, need energy (assume massless for high PT)
    E1 = pt1 * np.cosh(eta1)
    E2 = pt2 * np.cosh(eta2)
    E_sys = E1 + E2

    Z_Y = 0.5 * np.log((E_sys + pz_sys) / (E_sys - pz_sys))

    # Collins-soper angle
    px_sys = px1 + px2
    py_sys = py1 + py2

    M = np.sqrt(E_sys**2 - px_sys**2 - py_sys**2 - pz_sys**2)   #invariant mass

    # Boost to CM frame
    beta_x = px_sys / E_sys
    beta_y = py_sys / E_sys
    beta_z = pz_sys / E_sys
    gamma = E_sys / M

    # Boost negative muon (convention)
    px_neg, py_neg, pz_neg, E_neg = px2, py2, pz2, E2

    # Lorentz boost
    px_CM = px_neg - gamma * beta_x * E_neg
    py_CM = py_neg - gamma * beta_y * E_neg
    pz_CM = pz_neg - gamma * beta_z * E_neg
    E_CM = gamma * (E_neg - beta_x*px_neg - beta_y*py_neg - beta_z*pz_neg)
    
    # Collins-Soper z-axis in lab frame
    # CS axis = bisector of beam directions in CM
    # For pp collider at rest: simplified to beam axis direction
    
    # Magnitude of momentum in CM
    p_CM = np.sqrt(px_CM**2 + py_CM**2 + pz_CM**2)
    
    # cos(theta*) = pz component / |p|
    cos_theta_star = pz_CM / p_CM

    return Z_pt, Z_eta, Z_phi, Z_Y, cos_theta_star

def compute_system_variables_for_dataset(data, selected_features, variables, calculate_system_variables):
    """
    Compute system-level kinematic variables for a dataset.
    
    Parameters
    ----------
    variables : dict
        Variables configuration dictionary
    calculate_system_variables : callable
        Function to calculate system variables
    Returns
    -------
    Z_pt, Z_eta, Z_phi, Z_Y : np.ndarray
        System-level kinematic variables
    """
    # Get feature indices
    pt_1_idx = selected_features.index('Muons_Pos_PT')
    pt_2_idx = selected_features.index('Muons_Neg_PT')
    eta_1_idx = selected_features.index('Muons_Pos_Eta')
    eta_2_idx = selected_features.index('Muons_Neg_Eta')
    phi_1_idx = selected_features.index('Muons_Pos_Phi')
    phi_2_idx = selected_features.index('Muons_Neg_Phi')
    
    return calculate_system_variables(
        data[:, pt_1_idx],
        data[:, eta_1_idx],
        data[:, phi_1_idx],
        data[:, pt_2_idx],
        data[:, eta_2_idx],
        data[:, phi_2_idx],
    )  
