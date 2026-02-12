#!/usr/bin/env python
"""Main script to run Drell-Yan sample analysis."""

import sys
import os
import argparse
import glob
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive

from analyzer import DrellYanSampleAnalyzer


def get_latest_model(mlruns_dir="mlruns/models", model_pattern="MADEMOG_flow_model*"):
    """
    Find the most recently modified model in mlruns/models.
    
    Parameters
    ----------
    mlruns_dir : str
        Path to mlruns/models directory
    model_pattern : str
        Pattern to match model names (e.g., "MADEMOG_flow_model*")
    
    Returns
    -------
    str
        Name of the most recently created/modified model
    """
    mlruns_path = os.path.join(project_root, mlruns_dir)
    
    if not os.path.exists(mlruns_path):
        raise FileNotFoundError(f"MLruns directory not found: {mlruns_path}")
    
    # Find all model directories matching the pattern
    model_dirs = glob.glob(os.path.join(mlruns_path, model_pattern))
    
    if not model_dirs:
        raise FileNotFoundError(f"No models found matching pattern: {model_pattern}")
    
    # Get the most recently modified model directory
    latest_model_dir = max(model_dirs, key=os.path.getmtime)
    model_name = os.path.basename(latest_model_dir)
    
    # Get modification time for display
    mod_time = os.path.getmtime(latest_model_dir)
    from datetime import datetime
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Found {len(model_dirs)} model(s) matching pattern '{model_pattern}'")
    print(f"Using latest model: {model_name}")
    print(f"Last modified: {mod_time_str}")
    
    return model_name


def main(Generate=True):
    """Run the complete Drell-Yan analysis pipeline."""
    # Configuration
    data_dir = "/project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/data/drellyan/DRELLYAN.npy"
    variables_json = 'ml/data/drellyan/variables.json'
    
    # Load data config to get mass range
    config_path = os.path.join(project_root, 'ml/custom/DrellYan/config/flows/data_config.yaml')
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get mass range from config with 10 GeV buffer
    input_proc = data_config['data_config']['input_processing']
    mass_region = input_proc['mass_region']
    
    if mass_region == 'full_data':
        min_mass_config = input_proc['min_mass']
        max_mass_config = input_proc['max_mass']
        mass_cut_lower = None
        mass_cut_upper = None
        mass_range_min = min_mass_config
        mass_range_max = max_mass_config
        
    elif mass_region == 'sidebands':
        min_mass_config = input_proc['sideband_lower_min']
        max_mass_config = input_proc['sideband_upper_max']
        # The excluded region is between the sidebands
        mass_cut_lower = input_proc['sideband_lower_max']
        mass_cut_upper = input_proc['sideband_upper_min']
        # Overall range for unpreprocessed data (removes Z peak, keeps middle)
        mass_range_min = input_proc['sideband_lower_min']
        mass_range_max = input_proc['sideband_upper_max']
    
    mass_min = min_mass_config - 10
    mass_max = max_mass_config + 10
    
    # Automatically find the latest model
    model_name = get_latest_model(
        mlruns_dir="mlruns/models",
        model_pattern="MAFMADEMOG_flow_model*"  # Match all MAFMADEMOG models
    )
    
    # Correlation plot settings
    gridsize = 200  # Number of bins for 2D histograms in correlation plots
    
    # Initialize analyzer
    print("=" * 80)
    print("Drell-Yan Sample Analysis")
    print("=" * 80)
    analyzer = DrellYanSampleAnalyzer(
        data_dir, variables_json, model_name,
        mass_cut_lower=mass_cut_lower, mass_cut_upper=mass_cut_upper,
        mass_range_min=mass_range_min, mass_range_max=mass_range_max
    )
    
    # Generate samples
    print("\n" + "=" * 80)
    print("Step 1: Generating Samples")
    print("=" * 80)
    if Generate:
        analyzer.generate_samples(n_samples=1000000, chunks=20, debug=True)
    else: 
        analyzer.get_existing_samples()
    
    if variables_json == 'ml/data/drellyan/variables_reduced.json':
        # Compute PT negative
        print("\n" + "=" * 80)
        print("Step 2: Computing PT Negative")
        print("=" * 80)
        analyzer.compute_pt_negative()  

        # Create plots
        print("\n" + "=" * 80)
        print("Step 3: Creating Plots")
        print("=" * 80)
        
        print("\n[1/3] Creating feature comparison plots...")
        analyzer.plot_feature_comparison()

        print("\n[2/3] Creating PT negative comparison plot...")
        analyzer.plot_pt_negative_comparison()

        print("\n[3/3] Creating correlation comparison plots...")
        analyzer.plot_correlation_plots(gridsize=gridsize)

    if variables_json == 'ml/data/drellyan/variables.json':
        # Compute invariant masses
        print("\n" + "=" * 80)
        print("Step 2: Computing Invariant Masses and System Variables")
        print("=" * 80)
        analyzer.compute_masses()
        analyzer.compute_system_variables()
    
        # Create plots
        print("\n" + "=" * 80)
        print("Step 3: Creating Plots")
        print("=" * 80)
        
        print("\n[1/4] Creating feature comparison plots...")
        analyzer.plot_feature_comparison()

        print("\n[2/4] Creating system variables comparison plots (linear and log scale)...")
        analyzer.plot_system_variables_comparison(log_scale=False)
        analyzer.plot_system_variables_comparison(log_scale=True)

        print("\n[3/4] Creating invariant mass comparison plot...")
        analyzer.plot_invariant_mass(x_min = mass_min, x_max = mass_max)

        print("\n[4/4] Creating correlation comparison plots...")
        analyzer.plot_correlation_plots(gridsize=gridsize)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Drell-Yan sample analysis")
    parser.add_argument(
        "--generate", # Add this after terminal prompt to generate new samples
        action="store_true", 
        default=True,
        help="Generate new samples (default: False, use existing samples)"
    )
    # parser.add_argument(
    #     "--no-generate",
    #     dest="generate",
    #     action="store_false",
    #     help="Skip sample generation, use existing samples"
    # )
    
    args = parser.parse_args()
    main(Generate=args.generate)
