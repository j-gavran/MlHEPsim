# Drell-Yan Sample Analysis Tools

This directory contains modular tools for analyzing Drell-Yan dimuon samples from generative models.

## File Structure

```
sample_analyzer/
├── analyzer.py              # Main DrellYanSampleAnalyzer class
├── mass_calculator.py       # Invariant mass calculation utilities
├── pt_calculator.py         # PT negative muon calculation utilities
├── plotting.py              # All plotting functions (2D histograms only)
├── run_analysis.py          # Main execution script (supports full/reduced models)
├── figures/                 # Output directory for all plots
└── README.md                # This file
```

## Quick Start

### Run Complete Analysis

```bash
python ml/custom/DrellYan/sample_analyzer/run_analysis.py
```

This will:
1. Load the trained model and real data (configure which JSON to use in the script)
2. Generate 10M samples from the model
3. Compute derived quantities (invariant mass OR PT negative, depending on model)
4. Create comparison plots:
   - Feature distributions (1D histograms with ratio plots)
   - Invariant mass OR PT negative distribution
   - 2D correlation plots with difference maps (2D histograms only, no hexbin/scatter)

### Use as a Library

```python
from ml.custom.DrellYan.sample_analyzer.analyzer import DrellYanSampleAnalyzer

# Initialize
analyzer = DrellYanSampleAnalyzer(
    data_dir="path/to/DRELLYAN.npy",
    variables_json="path/to/variables.json",  # or variables_reduced.json
    model_name="MADEMOG_flow_model_gauss_rank"
)

# Generate samples
analyzer.generate_samples(n_samples=10000000, chunks=20, debug=True)

# For full model: Compute invariant masses
analyzer.compute_masses()

# For reduced model: Compute PT of negative muon from mass
analyzer.compute_pt_negative()

# Create plots (all save to figures/ directory)
analyzer.plot_feature_comparison()
analyzer.plot_invariant_mass()  # Or plot_pt_negative_comparison() for reduced model
analyzer.plot_correlation_plots(gridsize=200)
```

## Module Documentation

### `analyzer.py`
Main analyzer class that orchestrates the analysis pipeline.

**Key Methods:**
- `generate_samples()` - Generate samples from trained model
- `compute_masses()` - Calculate invariant masses (for full model)
- `compute_pt_negative()` - Calculate PT of negative muon from mass (for reduced model)
- `plot_feature_comparison()` - Plot 1D feature distributions with ratio plots
- `plot_invariant_mass()` - Plot invariant mass distribution
- `plot_pt_negative_comparison()` - Plot PT negative distribution
- `plot_correlation_plots()` - Plot 2D correlations with difference maps

### `mass_calculator.py`
Physics calculations for dimuon invariant mass.

**Functions:**
- `calculate_dimuon_invariant_mass()` - Compute M from kinematics using M² = 2·PT₁·PT₂·(cosh(Δη) - cos(Δφ))
- `compute_masses_for_dataset()` - Helper to compute for full dataset

### `pt_calculator.py`
Physics calculations for negative muon transverse momentum.

**Functions:**
- `calculate_pt_negative_muon()` - Compute PT₂ from mass and kinematics using PT₂ = M²/(2·PT₁·(cosh(Δη) - cos(Δφ)))
- `compute_pt_negative_for_dataset()` - Helper to compute for full dataset

### `plotting.py`
All visualization functions using 2D histograms exclusively.

**Functions:**
- `plot_feature_comparison()` - Compare real vs generated features with Generated/Real ratio plots
- `plot_invariant_mass()` - Compare invariant mass distributions
- `plot_pt_negative_comparison()` - Compare PT negative distributions with outlier exclusion
- `plot_correlation_comparison()` - 2D correlation plots with difference maps (auto-detects Lead/Sub vs Pos/Neg naming)
- `get_range_limits()` - Helper for plot ranges using percentile-based outlier exclusion

## Configuration

Edit `run_analysis.py` to modify:
- Data paths
- Model configuration:
  - `variables_json`: Choose `'ml/data/drellyan/variables.json'` (full) or `'ml/data/drellyan/variables_reduced.json'` (reduced)
- Model name
- Number of samples to generate
- Plot parameters:
  - `gridsize`: Number of bins for 2D histograms in correlation plots (default: 200)
  - Bins for 1D histograms (default: 100)

## Output

All plots are saved to the `sample_analyzer/figures/` directory (auto-created):

### Full Model (`variables.json`)
- `feature_comparison.png` - Individual feature distributions with ratio plots
- `invariant_mass_comparison.png` - Dimuon mass distribution
- `correlation_comparison.png` - 2D correlation plots (PT vs PT, Eta vs Eta, Phi vs Phi)

### Reduced Model (`variables_reduced.json`)
- `feature_comparison.png` - Individual feature distributions with ratio plots
- `pt_negative_comparison.png` - PT negative muon distribution (derived from mass)
- `correlation_comparison.png` - 2D correlation plots (Eta vs Eta, Phi vs Phi only)

## Key Features

### Adaptive Correlation Plots
The correlation plotting automatically detects which feature set is being used:
- **Full model**: Plots PT₁ vs PT₂, η₁ vs η₂, φ₁ vs φ₂ using Lead/Sub naming
- **Reduced model**: Plots η₁ vs η₂, φ₁ vs φ₂ only (no PT correlation) using Pos/Neg naming

### Outlier Handling
- **Feature plots**: Use percentile-based range limits (1th to 99th percentile) to exclude extreme outliers
- **PT negative plot**: Automatically excludes outliers to focus on the bulk distribution
- **Correlation plots**: Full range shown, 2D histograms handle density naturally

### 2D Histogram Visualization
All correlation plots use 2D histograms exclusively:
- **Real data**: Blue colormap
- **Generated data**: Red colormap
- **Difference map**: Diverging red-blue colormap showing (Real - Generated) density
- Controllable resolution via `gridsize` parameter

## Notes

- **Cache disabled**: Samples are always freshly generated (important after retraining)
- **Preprocessing**: Uses Gaussian rank transformation for continuous features
- **Mass cuts**: Training data has 110-160 GeV mass window (configured in `process_drellyan_dataset.py`)
- **Feature naming**: 
  - Full model uses `Lead/Sub` convention (leading/subleading muons by PT)
  - Reduced model uses `Pos/Neg` convention (positive/negative charge muons)
- **PT calculation**: For reduced model, PT₂ is derived using: PT₂ = M²/(2·PT₁·(cosh(Δη) - cos(Δφ)))
