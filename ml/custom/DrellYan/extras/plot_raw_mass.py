#!/usr/bin/env python
"""Plot Muons_Minv_MuMu directly from ROOT files without any cuts."""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_mass_from_root(file_paths, mass_column="Muons_Minv_MuMu"):
    """
    Load Muons_Minv_MuMu values directly from ROOT files.
    
    Parameters
    ----------
    file_paths : list
        List of ROOT file paths to load
    mass_column : str
        Name of the mass column to load
        
    Returns
    -------
    np.ndarray
        Array of all mass values
    """
    all_masses = []
    
    for file_path in file_paths:
        print(f"Loading from: {file_path}")
        with uproot.open(file_path) as root_file:
            tree = root_file["tree_Hmumu"]
            
            # Load mass values without any cuts
            mass_data = tree[mass_column].array(library="np")
            all_masses.append(mass_data)
            print(f"  Loaded {len(mass_data):,} events")
    
    # Concatenate all masses
    combined_masses = np.concatenate(all_masses)
    print(f"\nTotal events: {len(combined_masses):,}")
    
    return combined_masses


def plot_mass_distribution(masses, output_path="raw_mass_distribution.png", bins=200, x_min=0, x_max=600):
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(masses, bins=bins, range=(x_min, x_max),
                                         color='steelblue', alpha=0.7, edgecolor='black')
    
    # Formatting
    ax.set_xlabel('Dimuon Invariant Mass [GeV]', fontsize=14)
    ax.set_ylabel('Number of Events', fontsize=14)
    ax.set_title('Raw Muons_Minv_MuMu Distribution (No Cuts Applied)', fontsize=16, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def plot_mass_distribution_log(masses, output_path="raw_mass_distribution_log.png",
                               bins=200, x_min=0, x_max=600):
    """
    Plot the raw mass distribution with log scale on y-axis.
    
    Parameters
    ----------
    masses : np.ndarray
        Array of mass values
    output_path : str
        Path to save the plot
    bins : int
        Number of histogram bins
    x_min, x_max : float
        Range for x-axis (GeV)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(masses, bins=bins, range=(x_min, x_max),
                                         color='steelblue', alpha=0.7, edgecolor='black')

    # Formatting
    ax.set_xlabel('Dimuon Invariant Mass [GeV]', fontsize=14)
    ax.set_ylabel('Number of Events (log scale)', fontsize=14)
    ax.set_title('Raw Muons_Minv_MuMu Distribution - Log Scale (No Cuts Applied)', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Log-scale plot saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Raw Muons_Minv_MuMu Distribution Plotter")
    print("=" * 80)
    
    # File paths
    file_paths = [
        "/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700889.Sh_2214_Zmumu_mZ_105_ECMS_BFilter_HmumuSR_skimmed_prepared_FSR.root",
        "/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700890.Sh_2214_Zmumu_mZ_105_ECMS_CFilterBVeto_HmumuSR_skimmed_prepared_FSR.root",
        "/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700891.Sh_2214_Zmumu_mZ_105_ECMS_CVetoBVeto_HmumuSR_skimmed_prepared_FSR.root"
    ]
    
    # Load mass data
    print("\nLoading mass data from ROOT files...")
    print("-" * 80)
    masses = load_mass_from_root(file_paths)
    
    # Create plots
    print("\n" + "=" * 80)
    print("Creating Plots")
    print("=" * 80)
    
    # Linear scale plot
    print("\n[1/2] Creating linear scale plot...")
    plot_mass_distribution(masses, 
                          output_path="ml/custom/DrellYan/extras/raw_mass_distribution.png",
                          bins=200, x_min=0, x_max=600)
    
    # Log scale plot
    print("\n[2/2] Creating log scale plot...")
    plot_mass_distribution_log(masses,
                              output_path="ml/custom/DrellYan/extras/raw_mass_distribution_log.png",
                              bins=200, x_min=0, x_max=600)
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
