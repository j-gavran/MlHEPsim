"""Plotting utilities for Drell-Yan sample analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt


def get_range_limits(data, percentile_range=0.1):
    """Calculate range limits excluding extreme outliers."""
    lower = np.percentile(data, percentile_range)
    upper = np.percentile(data, 100 - percentile_range)
    data_range = upper - lower
    # Add padding
    lower = lower - 0.1 * data_range
    upper = upper + 0.1 * data_range
    return lower, upper


def plot_feature_comparison(real_data, generated_data, selected_features, selection, 
                           variables, output_path=None, bins=50, n_cols=3, 
                           mass_cut_lower=None, mass_cut_upper=None):
    """
    Plot comparison of all features between real and generated data with ratio plots.
    
    Parameters
    ----------
    selected_features : list
        List of selected feature names
    selection : pd.DataFrame
        Feature selection DataFrame with type information
    variables : dict
        Variables configuration dictionary
    mass_cut_lower : float, optional
        Lower boundary of mass cut region (excluded from training)
    mass_cut_upper : float, optional
        Upper boundary of mass cut region (excluded from training)
    """
    # Calculate grid dimensions
    n_features = len(selected_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure with subplots: each feature gets 2 rows (main plot + ratio)
    # Adjust vertical space to accommodate spacing
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    # Use different approach: manually calculate spacing for each pair
    # Each feature pair (main + ratio) gets its own mini-grid with proper spacing
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    
    # Create outer grid for feature rows
    outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)
    
    # Collect all ratios to determine appropriate y-axis limits
    all_ratios = []
    
    # Plot each feature
    for i, feature in enumerate(selected_features):
        row = i // n_cols
        col = i % n_cols
        
        # Create inner grid for this feature (main plot + ratio plot)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], 
                                          height_ratios=[4, 1], hspace=0.08)
        
        # Create main plot and ratio plot
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        feature_type = selection[selection['feature'] == feature]['type'].values[0]
        
        # Get real data (original physical values)
        original_feature_idx = list(variables['colnames'].keys()).index(feature)
        real_feature_data = real_data[:, original_feature_idx]
        
        # Get generated data (already inverse-transformed)
        generated_feature_data = generated_data[:, i]
        
        print(f"\n{feature} ({feature_type}):")
        print(f"  Real range:      [{real_feature_data.min():.3f}, {real_feature_data.max():.3f}]")
        print(f"  Generated range: [{generated_feature_data.min():.3f}, {generated_feature_data.max():.3f}]")
        
        # Calculate histogram range
        x_min = min(get_range_limits(real_feature_data)[0], 
                   get_range_limits(generated_feature_data)[0])
        x_max = max(get_range_limits(real_feature_data)[1], 
                   get_range_limits(generated_feature_data)[1])
        
        # Create histograms
        hist_real, bins_edges = np.histogram(real_feature_data, bins=bins, 
                                             range=(x_min, x_max), density=True)
        hist_gen, _ = np.histogram(generated_feature_data, bins=bins_edges, density=True)
        
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Main plot
        ax_main.step(bin_centers, hist_real, color='blue', label='Real Data (Unpreprocessed)', where='mid', lw=2)
        ax_main.step(bin_centers, hist_gen, color='red', label='Generated', where='mid', lw=2)
        
        # Formatting main plot
        y_max = max(max(hist_real), max(hist_gen)) * 1.1
        ax_main.set_ylabel('Density', fontsize=10)
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(0, y_max)
        ax_main.grid(True, linestyle='--', alpha=0.5)
        ax_main.legend(fontsize=8, loc='best')
        ax_main.set_title(feature, fontsize=10, pad=5)
        ax_main.tick_params(labelbottom=False, labelsize=9)
        
        # Ratio plot: Generated / Real
        # Avoid division by zero
        ratio = np.divide(hist_gen, hist_real, out=np.ones_like(hist_gen), where=hist_real!=0)
        all_ratios.extend(ratio[np.isfinite(ratio)])
        
        ax_ratio.step(bin_centers, ratio, color='black', where='mid', lw=1.5)
        ax_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax_ratio.set_ylabel('Gen/Real', fontsize=8)
        ax_ratio.set_xlim(x_min, x_max)
        ax_ratio.grid(True, linestyle='--', alpha=0.5)
        ax_ratio.tick_params(labelsize=8)
    
    # Determine appropriate y-limits for ratio plots based on actual data
    if all_ratios:
        ratio_min = max(0.85, np.percentile(all_ratios, 1) * 0.95)
        ratio_max = min(1.15, np.percentile(all_ratios, 99) * 1.05)
        
        # Set the same y-limits for all ratio plots
        # With nested GridSpec, axes are ordered: [main1, ratio1, main2, ratio2, ...]
        for i in range(n_features):
            ax_ratio = fig.axes[i * 2 + 1]  # Get ratio subplot (every odd index)
            ax_ratio.set_ylim(ratio_min, ratio_max)
    
    # Save plot
    if output_path is None:
        output_path = 'feature_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFeature comparison plot saved to: {output_path}")


def plot_system_variables_comparison(real_system_vars, generated_system_vars, output_path=None, bins=50, log_scale=False,
                                    mass_cut_lower=None, mass_cut_upper=None):
    """
    Plot 1D histogram comparison of system-level variables with ratio plots.
    
    Plots Z_PT, Z_eta, Z_phi, Z_Y, and cos(theta*) distributions comparing real vs generated data.
    
    Parameters
    ----------
    real_system_vars : tuple of np.ndarray
        Tuple containing (Z_pt, Z_eta, Z_phi, Z_Y, cos_theta_star) for real data
    generated_system_vars : tuple of np.ndarray
        Tuple containing (Z_pt, Z_eta, Z_phi, Z_Y, cos_theta_star) for generated data
    log_scale : bool
        If True, use logarithmic y-axis
    mass_cut_lower : float, optional
        Lower boundary of mass cut region (excluded from training)
    mass_cut_upper : float, optional
        Upper boundary of mass cut region (excluded from training)
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    
    # Unpack system variables
    real_Z_pt, real_Z_eta, real_Z_phi, real_Z_Y, real_cos_theta_star = real_system_vars
    gen_Z_pt, gen_Z_eta, gen_Z_phi, gen_Z_Y, gen_cos_theta_star = generated_system_vars
    
    # Define variables to plot
    variables = [
        ("Z_PT [GeV]", real_Z_pt, gen_Z_pt),
        ("Z_eta", real_Z_eta, gen_Z_eta),
        ("Z_phi", real_Z_phi, gen_Z_phi),
        ("Z_Y (rapidity)", real_Z_Y, gen_Z_Y),
        ("cos(theta*) Collins-Soper", real_cos_theta_star, gen_cos_theta_star)
    ]
    
    n_features = len(variables)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    # Create outer grid
    outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)
    
    # Collect all ratios for consistent y-axis limits
    all_ratios = []
    
    # Plot each variable
    for i, (var_name, real_data, gen_data) in enumerate(variables):
        row = i // n_cols
        col = i % n_cols
        
        # Create inner grid for this variable (main plot + ratio plot)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], 
                                          height_ratios=[4, 1], hspace=0.08)
        
        # Create main plot and ratio plot
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        print(f"\n{var_name}:")
        print(f"  Real range:      [{real_data.min():.3f}, {real_data.max():.3f}]")
        print(f"  Generated range: [{gen_data.min():.3f}, {gen_data.max():.3f}]")
        
        # Calculate histogram range using percentiles to exclude outliers
        x_min = min(np.percentile(real_data, 0.1), np.percentile(gen_data, 0.1))
        x_max = max(np.percentile(real_data, 99.9), np.percentile(gen_data, 99.9))
        data_range = x_max - x_min
        
        x_min = x_min - 0.1 * data_range
        x_max = x_max + 0.1 * data_range
        
        # Create histograms
        hist_real, bins_edges = np.histogram(real_data, bins=bins, 
                                             range=(x_min, x_max), density=True)
        hist_gen, _ = np.histogram(gen_data, bins=bins_edges, density=True)
        
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Main plot
        ax_main.step(bin_centers, hist_real, color='blue', label='Real Data (Unpreprocessed)', where='mid', lw=2)
        ax_main.step(bin_centers, hist_gen, color='red', label='Generated', where='mid', lw=2)
        
        # Formatting main plot
        y_max = max(max(hist_real), max(hist_gen)) * 1.1
        ax_main.set_ylabel('Density', fontsize=10)
        ax_main.set_xlim(x_min, x_max)
        
        if log_scale:
            ax_main.set_yscale('log')
            # Set y limits for log scale to avoid showing zeros
            y_min = min(hist_real[hist_real > 0].min() if np.any(hist_real > 0) else 1e-6,
                       hist_gen[hist_gen > 0].min() if np.any(hist_gen > 0) else 1e-6) * 0.5
            ax_main.set_ylim(y_min, y_max * 2)
        else:
            ax_main.set_ylim(0, y_max)
            
        ax_main.grid(True, linestyle='--', alpha=0.5)
        ax_main.legend(fontsize=8, loc='best')
        ax_main.set_title(var_name, fontsize=10, pad=5)
        ax_main.tick_params(labelbottom=False, labelsize=9)
        
        # Ratio plot: Generated / Real
        ratio = np.divide(hist_gen, hist_real, out=np.ones_like(hist_gen), where=hist_real!=0)
        all_ratios.extend(ratio[np.isfinite(ratio)])
        
        ax_ratio.step(bin_centers, ratio, color='black', where='mid', lw=1.5)
        ax_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax_ratio.set_ylabel('Gen/Real', fontsize=8)
        ax_ratio.set_xlim(x_min, x_max)
        ax_ratio.grid(True, linestyle='--', alpha=0.5)
        ax_ratio.tick_params(labelsize=8)
    
    # Determine appropriate y-limits for ratio plots
    if all_ratios:
        ratio_min = max(0.85, np.percentile(all_ratios, 1) * 0.95)
        ratio_max = min(1.15, np.percentile(all_ratios, 99) * 1.05)
        
        # Set the same y-limits for all ratio plots
        for i in range(n_features):
            ax_ratio = fig.axes[i * 2 + 1]  # Get ratio subplot (every odd index)
            ax_ratio.set_ylim(ratio_min, ratio_max)
    
    # Save plot
    if output_path is None:
        suffix = '_log' if log_scale else ''
        output_path = f'system_variables_comparison{suffix}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSystem variables comparison plot saved to: {output_path}")


def plot_invariant_mass(real_mass, generated_mass, output_path=None, bins=100, 
                       x_min=110, x_max=170, mass_cut_lower=None, mass_cut_upper=None):
    """
    Plot comparison of invariant mass distributions with ratio plot.
    
    Parameters
    ----------
    mass_cut_lower : float, optional
        Lower boundary of mass cut region (excluded from training)
    mass_cut_upper : float, optional
        Upper boundary of mass cut region (excluded from training)
    """
    from matplotlib.gridspec import GridSpec
    
    # Create figure with four subplots: main plot, ratio, zoomed gap, gap ratio
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 1], hspace=0.08, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_gap_ratio = fig.add_subplot(gs[1, 1], sharex=ax_gap)

    # Create bin edges for exact whole number bins
    # If bins is an integer, create evenly spaced bins at whole numbers
    if isinstance(bins, int):
        # Calculate bin width to get approximately the requested number of bins
        bin_width = (x_max - x_min) / bins
        # Round to nearest 0.5 or 1.0 for cleaner bins
        if bin_width < 1.0:
            bin_width = 0.5
        else:
            bin_width = round(bin_width)
        # Create bin edges starting exactly at x_min
        bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    else:
        bin_edges = bins
    
    # Create histograms with density normalization (area = 1)
    hist_real, bins_real = np.histogram(real_mass, bins=bin_edges, density=True)
    hist_gen, bins_gen = np.histogram(generated_mass, bins=bin_edges, density=True)
    
    # Plot main histogram
    bin_centers = (bins_real[:-1] + bins_real[1:]) / 2
    ax_main.step(bin_centers, hist_real, color='blue', label='Real Data (Unpreprocessed)', where='mid', lw=2)
    ax_main.step(bin_centers, hist_gen, color='red', label='Generated', where='mid', lw=2)
    
    # Add mass cut indicators if provided
    if mass_cut_lower is not None and mass_cut_upper is not None:
        y_max_vis = max(max(hist_real), max(hist_gen)) * 1.1
        ax_main.axvspan(mass_cut_lower, mass_cut_upper, alpha=0.15, color='gray', 
                       label=f'Excluded Region ({mass_cut_lower}-{mass_cut_upper} GeV)')
        ax_main.axvline(mass_cut_lower, color='gray', linestyle='--', lw=1.5, alpha=0.7)
        ax_main.axvline(mass_cut_upper, color='gray', linestyle='--', lw=1.5, alpha=0.7)
    
    # Formatting main plot
    y_max = max(max(hist_real), max(hist_gen)) * 1.1
    ax_main.set_ylabel('Density', fontsize=12)
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(0, y_max)
    ax_main.set_title('Dimuon Invariant Mass Distribution', fontsize=14)
    ax_main.grid(True, linestyle='--', alpha=0.7)
    ax_main.legend(fontsize=11)
    ax_main.tick_params(labelbottom=False)
    
    # Ratio plot: Generated / Real
    ratio = np.divide(hist_gen, hist_real, 
                     out=np.zeros_like(hist_gen, dtype=float), 
                     where=hist_real!=0)
    
    ax_ratio.step(bin_centers, ratio, color='black', where='mid', lw=1.5)
    ax_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7)
    ax_ratio.set_xlabel('Dimuon Invariant Mass [GeV]', fontsize=12)
    ax_ratio.set_ylabel('Gen/Real', fontsize=10)
    ax_ratio.set_xlim(x_min, x_max)
    ax_ratio.grid(True, linestyle='--', alpha=0.7)
    
    # Set reasonable y-limits for ratio plot based on data (excluding zeros)
    ratio_nonzero = ratio[ratio > 0]
    if len(ratio_nonzero) > 0:
        ratio_min = max(0, np.percentile(ratio_nonzero, 1) * 0.8)
        ratio_max = min(2, np.percentile(ratio_nonzero, 99) * 1.2)
        ax_ratio.set_ylim(ratio_min, ratio_max)
    else:
        ax_ratio.set_ylim(0, 2)
    
    # ===== RIGHT SIDE: ZOOMED GAP REGION =====
    if mass_cut_lower is not None and mass_cut_upper is not None:
        gap_margin = 2  # GeV on each side
        gap_x_min = mass_cut_lower - gap_margin
        gap_x_max = mass_cut_upper + gap_margin
        
        # Use same bin edges as full plot, but filter to gap region
        gap_mask = (bin_centers >= gap_x_min) & (bin_centers <= gap_x_max)
        gap_bin_centers = bin_centers[gap_mask]
        gap_hist_real = hist_real[gap_mask]
        gap_hist_gen = hist_gen[gap_mask]
        
        # Plot gap histogram (using same density normalization as full plot)
        ax_gap.step(gap_bin_centers, gap_hist_real, color='blue', label='Real Data', where='mid', lw=2)
        ax_gap.step(gap_bin_centers, gap_hist_gen, color='red', label='Generated', where='mid', lw=2)
        
        # Add exclusion region shading
        y_max_gap = max(max(gap_hist_real) if len(gap_hist_real) > 0 else 0, 
                        max(gap_hist_gen) if len(gap_hist_gen) > 0 else 0) * 1.1
        ax_gap.axvspan(mass_cut_lower, mass_cut_upper, alpha=0.15, color='gray')
        ax_gap.axvline(mass_cut_lower, color='gray', linestyle='--', lw=1.5, alpha=0.7)
        ax_gap.axvline(mass_cut_upper, color='gray', linestyle='--', lw=1.5, alpha=0.7)
        
        # Formatting gap plot
        ax_gap.set_ylabel('Density', fontsize=12)
        ax_gap.set_xlim(gap_x_min, gap_x_max)
        ax_gap.set_ylim(0, y_max_gap)
        ax_gap.set_title(f'Gap Region Detail ({mass_cut_lower}-{mass_cut_upper} GeV)', fontsize=14)
        ax_gap.grid(True, linestyle='--', alpha=0.7)
        ax_gap.legend(fontsize=11)
        ax_gap.tick_params(labelbottom=False)
        
        # Gap ratio plot
        gap_ratio = np.divide(gap_hist_gen, gap_hist_real,
                             out=np.zeros_like(gap_hist_gen, dtype=float),
                             where=gap_hist_real!=0)
        
        ax_gap_ratio.step(gap_bin_centers, gap_ratio, color='black', where='mid', lw=1.5)
        ax_gap_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax_gap_ratio.set_xlabel('Dimuon Invariant Mass [GeV]', fontsize=12)
        ax_gap_ratio.set_ylabel('Gen/Real', fontsize=10)
        ax_gap_ratio.set_xlim(gap_x_min, gap_x_max)
        ax_gap_ratio.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-limits for gap ratio with tighter bounds
        gap_ratio_nonzero = gap_ratio[gap_ratio > 0]
        if len(gap_ratio_nonzero) > 0:
            # Use 5-95 percentile range with moderate margins for better detail
            ratio_center = np.median(gap_ratio_nonzero)
            ratio_range = np.percentile(gap_ratio_nonzero, 95) - np.percentile(gap_ratio_nonzero, 5)
            gap_ratio_min = max(0.7, ratio_center - ratio_range * 1.5)
            gap_ratio_max = min(1.3, ratio_center + ratio_range * 1.5)
            ax_gap_ratio.set_ylim(gap_ratio_min, gap_ratio_max)
        else:
            ax_gap_ratio.set_ylim(0.7, 1.3)
    
    # Save plot
    if output_path is None:
        output_path = 'invariant_mass_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Invariant mass plot saved to: {output_path}")

def plot_pt_negative_comparison(real_pt_negative, generated_pt_negative, output_path=None, bins=100):
    """
    Plot comparison of pt negative distributions.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use percentile-based range to exclude outliers (similar to get_range_limits)
    percentile_range = 1  # Exclude bottom and top 1% outliers
    
    # Calculate range limits for both datasets
    real_lower = np.percentile(real_pt_negative, percentile_range)
    real_upper = np.percentile(real_pt_negative, 100 - percentile_range)
    gen_lower = np.percentile(generated_pt_negative, percentile_range)
    gen_upper = np.percentile(generated_pt_negative, 100 - percentile_range)
    
    # Use the combined range with some padding
    x_min = min(real_lower, gen_lower)
    x_max = max(real_upper, gen_upper)
    data_range = x_max - x_min
    x_min = x_min - 0.05 * data_range
    x_max = x_max + 0.05 * data_range
    
    # Create bin edges for exact whole number bins
    # If bins is an integer, create evenly spaced bins at whole numbers
    if isinstance(bins, int):
        # Calculate bin width to get approximately the requested number of bins
        bin_width = (x_max - x_min) / bins
        # Round to nearest 0.5 or 1.0 for cleaner bins
        if bin_width < 1.0:
            bin_width = 0.5
        else:
            bin_width = round(bin_width)
        # Create bin edges starting exactly at x_min
        bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    else:
        bin_edges = bins
    
    # Create histograms with exact bin edges
    hist_real, bins_real = np.histogram(real_pt_negative, bins=bin_edges, density=True)
    hist_gen, bins_gen = np.histogram(generated_pt_negative, bins=bin_edges, density=True)
    
    # Plot
    bin_centers_real = (bins_real[:-1] + bins_real[1:]) / 2
    bin_centers_gen = (bins_gen[:-1] + bins_gen[1:]) / 2
    ax.step(bin_centers_real, hist_real, color='blue', label='Real Data', where='mid', lw=2)
    ax.step(bin_centers_gen, hist_gen, color='red', label='Generated', where='mid', lw=2)
    
    # Formatting
    y_max = max(max(hist_real), max(hist_gen)) * 1.1
    ax.set_xlabel('PT Negative [GeV]', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title('PT Negative Distribution', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = 'pt_negative_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PT Negative plot saved to: {output_path}")

def plot_correlation_comparison(real_data, generated_data, selected_features, variables,
                                gridsize, output_path=None, mass_cut_lower=None, mass_cut_upper=None):
    """
    Plot pairwise correlations for (pt1 vs pt2), (eta1 vs eta2), (phi1 vs phi2).
    
    Creates a 3x3 grid: real data (left), generated data (middle), difference (right).
    Uses 2D histograms for visualization.
    
    Parameters
    ----------
    selected_features : list
        List of selected feature names
    variables : dict
        Variables configuration dictionary
    gridsize : int
        Number of bins for 2D histograms (default: 200)
    mass_cut_lower : float, optional
        Lower boundary of mass cut region (excluded from training)
    mass_cut_upper : float, optional
        Upper boundary of mass cut region (excluded from training)
    """
    # Determine which feature naming convention is being used
    available_features = list(variables['colnames'].keys())
    
    # Check if using Lead/Sub naming (normal) or Pos/Neg naming (reduced)
    if "Muons_Pos_PT" in available_features and "Muons_Neg_PT" in available_features:
        # Normal model with both PT features
        pairs = [
            ("Muons_Pos_PT", "Muons_Neg_PT"),
            ("Muons_Pos_Eta", "Muons_Neg_Eta"),
            ("Muons_Pos_Phi", "Muons_Neg_Phi")
        ]
    elif "Muons_Pos_PT" in available_features:
        # Reduced model with only one PT feature
        pairs = [
            ("Muons_Pos_Eta", "Muons_Neg_Eta"),
            ("Muons_Pos_Phi", "Muons_Neg_Phi")
        ]
        # Note: Cannot plot PT vs PT since we only have one PT feature
    else:
        raise ValueError(f"Could not determine feature naming convention from available features: {available_features}")
    
    # Helper to get real data column
    orig_idx_map = {f: i for i, f in enumerate(list(variables['colnames'].keys()))}
    
    def get_real_col(feature):
        return real_data[:, orig_idx_map[feature]]
    
    # Prepare data pairs
    real_pairs = []
    gen_pairs = []
    
    for feat1, feat2 in pairs:
        # Real data (original physical units)
        real_pairs.append((get_real_col(feat1), get_real_col(feat2)))
        
        # Generated data (already inverse-transformed)
        idx1 = selected_features.index(feat1)
        idx2 = selected_features.index(feat2)
        gen_pairs.append((generated_data[:, idx1], generated_data[:, idx2]))
    
    # Create figure with 3 columns: real, generated, difference
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    
    for row, ((x_real, y_real), (x_gen, y_gen), (label1, label2)) in enumerate(
        zip(real_pairs, gen_pairs, pairs)
    ):
        ax_real = axs[row, 0]
        ax_gen = axs[row, 1]
        ax_diff = axs[row, 2]
        
        # Use percentile-based range to exclude top 5% outliers
        upper_percentile = 95.0
        
        # Combine real and generated data to determine consistent range
        x_combined = np.concatenate([x_real, x_gen])
        y_combined = np.concatenate([y_real, y_gen])
        
        # Calculate range for x-axis
        x_lower = np.percentile(x_combined, 0)
        x_upper = np.percentile(x_combined, upper_percentile)
        x_range = x_upper - x_lower
        x_min = x_lower - 0.05 * x_range
        x_max = x_upper + 0.05 * x_range
        
        # Calculate range for y-axis
        y_lower = np.percentile(y_combined, 0)
        y_upper = np.percentile(y_combined, upper_percentile)
        y_range = y_upper - y_lower
        y_min = y_lower - 0.05 * y_range
        y_max = y_upper + 0.05 * y_range
        
        # Number of bins for 2D histogram
        n_bins = gridsize
        print(f"  Row {row} ({label1} vs {label2}): Using {n_bins}x{n_bins} bins")
        
        # Create 2D histograms for real and generated data
        hist_real, xedges, yedges = np.histogram2d(x_real, y_real, bins=n_bins, 
                                                   range=[[x_min, x_max], [y_min, y_max]], 
                                                   density=True)
        hist_gen, _, _ = np.histogram2d(x_gen, y_gen, bins=n_bins, 
                                       range=[[x_min, x_max], [y_min, y_max]], 
                                       density=True)
        
        # Plot real data as 2D histogram
        im_real = ax_real.imshow(hist_real.T, origin='lower', aspect='auto',
                                extent=[x_min, x_max, y_min, y_max],
                                cmap='Blues', interpolation='nearest')
        plt.colorbar(im_real, ax=ax_real, label='Density')
        
        # Plot generated data as 2D histogram
        im_gen = ax_gen.imshow(hist_gen.T, origin='lower', aspect='auto',
                              extent=[x_min, x_max, y_min, y_max],
                              cmap='Reds', interpolation='nearest')
        plt.colorbar(im_gen, ax=ax_gen, label='Density')
        
        # Difference: Real - Generated (positive means more real data, negative means more generated)
        diff = hist_real - hist_gen
        
        # Plot difference with diverging colormap
        im = ax_diff.imshow(diff.T, origin='lower', aspect='auto', 
                           extent=[x_min, x_max, y_min, y_max],
                           cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        plt.colorbar(im, ax=ax_diff, label='Real - Generated')
        
        # Formatting
        ax_real.set_title(f"Real (Unpreprocessed): {label1} vs {label2}", fontsize=12)
        ax_gen.set_title(f"Generated: {label1} vs {label2}", fontsize=12)
        ax_diff.set_title(f"Difference: {label1} vs {label2}", fontsize=12)
        
        for ax in [ax_real, ax_gen, ax_diff]:
            ax.set_xlabel(label1, fontsize=10)
            ax.set_ylabel(label2, fontsize=10)
            ax.grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = 'correlation_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation comparison plot saved to: {output_path}")


