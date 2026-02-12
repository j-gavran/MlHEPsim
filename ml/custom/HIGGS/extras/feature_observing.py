import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt 
import numpy as np
import json

# Load feature names from variables.json
with open('ml/data/higgs/variables.json', 'r') as f:
    variables = json.load(f)

# Get continuous feature names (matching metrics plots)
cont_features = [name for name, type in variables['colnames'].items() 
                if type == 'cont']

data_dir = "/project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/data/higgs/HIGGS.npy"
data = np.load(data_dir)

# Calculate grid dimensions (similar to metrics plot)
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
    feature_data = data[:, i]  # Assuming same order as variables.json
    
    # Calculate x-axis range
    x_min, x_max = get_range_limits(feature_data)
    
    # Create histogram using calculated range
    counts, bins, _ = ax.hist(feature_data, bins=50, density=True, histtype='step', 
                            color='blue', label='Data', range=(x_min, x_max))
    
    # Calculate appropriate y limit with 10% padding
    y_max = max(counts) * 1.1
    
    # Remove the automatically plotted histogram and replot it
    # (this ensures the histogram appears on top of the grid lines)
    ax.clear()
    ax.hist(feature_data, bins=bins, density=True, histtype='step', 
            color='blue', label='Data')
    
    # Set labels and limits
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_xlim(x_min, x_max)  # Adaptive x limit
    ax.set_ylim(0, y_max)      # Adaptive y limit
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove any empty subplots
for j in range(i + 1, len(axs)):
    axs[j].remove()

plt.tight_layout()
# Save as PNG with high DPI for better quality
plt.savefig('features_grid.png', dpi=300, bbox_inches='tight')
plt.close()