import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path

import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt 
import numpy as np
import json
from ml.common.nn.gen_model_sampler import GenModelSampler
from scipy import stats
from ml.common.data_utils.feature_scaling import RescalingHandler
from ml.common.data_utils.processors import Preprocessor
from ml.custom.HIGGS.process_higgs_dataset import HIGGSFeatureSelector, HIGGSNpyProcessor

# Load feature names from variables.json
with open('ml/data/higgs/variables.json', 'r') as f:
    variables = json.load(f)

# Get continuous feature names
cont_features = [name for name, type in variables['colnames'].items() 
                if type == 'cont']

# Load real HIGGS data
data_dir = "/project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/data/higgs/HIGGS.npy"
real_data = np.load(data_dir)

# Get the preprocessor and feature selector to access the scalers
npy_proc = HIGGSNpyProcessor(data_dir="ml/data/higgs/", base_file_name="HIGGS")
# Load the features dictionary
with open('ml/data/higgs/variables.json', 'r') as f:
    features = json.load(f)

# Initialize the feature selector with the same settings used during training
f_sel = HIGGSFeatureSelector(
    file_path=npy_proc.npy_file, 
    features=features,
    drop_types=['label', 'disc', 'uni'],  # Only keep continuous features
)

pre = Preprocessor(cont_rescale_type="gauss_rank")

# First get the scalers and selection from the real data
selection = f_sel._select_colnames()
real_data_selected, selection, scalers = pre(real_data, selection)

print("Feature selection details:")
print(selection)
print("\nShape information:")
print(f"Real data shape after selection: {real_data_selected.shape}")

# Initialize the sampler
sampler = GenModelSampler(
    model_names="MADEMOG_flow_model_gauss_rank", 
    save_dir="ml/data/higgs",
    file_name="HIGGS_generated"
)

# Generate a smaller number of samples
N = 10000000
generated_samples = sampler.sample(N, chunks=20)  # Get samples in dictionary
generated_data = generated_samples["MADEMOG_flow_model_gauss_rank"][0]  # Extract array from dict

print(f"Generated data shape: {generated_data.shape}")

# Initialize the rescaling handler for inverse transform
rescale_handler = RescalingHandler(selection, scalers)

# Apply inverse transform to get back to original distribution
generated_data = rescale_handler.inverse_transform(generated_data)

# Calculate grid dimensions
n_features = len(cont_features)
n_cols = 5
n_rows = (n_features + n_cols - 1) // n_cols

# Create figure with subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axs = axs.flatten()

# Function to calculate range limits
def get_range_limits(data, percentile_range=0.1):
    """Calculate range limits excluding extreme outliers"""
    lower = np.percentile(data, percentile_range)
    upper = np.percentile(data, 100 - percentile_range)
    data_range = upper - lower
    # Add padding
    lower = lower - 0.1 * data_range
    upper = upper + 0.1 * data_range
    return lower, upper

# Plot each feature
for i, (feature, ax) in enumerate(zip(cont_features, axs)):
    # Get feature data
    real_feature_data = real_data[:, i]
    generated_feature_data = generated_data[:, i]
    
    # Calculate x-axis range based on both datasets
    x_min = min(get_range_limits(real_feature_data)[0], 
                get_range_limits(generated_feature_data)[0])
    x_max = max(get_range_limits(real_feature_data)[1], 
                get_range_limits(generated_feature_data)[1])
    
    # Create histograms
    bins = 50  # You can adjust this number to control histogram resolution
    hist_real, bins_real = np.histogram(real_feature_data, bins=bins, range=(x_min, x_max), density=True)
    hist_gen, bins_gen = np.histogram(generated_feature_data, bins=bins, range=(x_min, x_max), density=True)
    
    # Plot histograms
    bin_centers_real = (bins_real[:-1] + bins_real[1:]) / 2
    bin_centers_gen = (bins_gen[:-1] + bins_gen[1:]) / 2
    ax.step(bin_centers_real, hist_real, color='blue', label='Real Data', where='mid', lw=2)
    ax.step(bin_centers_gen, hist_gen, color='red', label='Generated', where='mid', lw=2)
    
    # Calculate appropriate y limit
    y_max = max(max(hist_real), max(hist_gen)) * 1.1
    
    # Set labels and limits
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

# Remove any empty subplots
for j in range(i + 1, len(axs)):
    axs[j].remove()

plt.tight_layout()
# Save as PNG with high DPI for better quality
plt.savefig('extras/feature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()