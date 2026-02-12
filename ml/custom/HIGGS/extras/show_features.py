import logging
import numpy as np
from process_higgs_dataset import HIGGSNpyProcessor, HIGGSFeatureSelector
from ml.common.utils.loggers import setup_logger

setup_logger()

# Initialize the data processor
npy_proc = HIGGSNpyProcessor(
    "ml/data/higgs/",
    base_file_name="HIGGS_data",
)

# Initialize feature selector with same config as in your flows
f_sel = HIGGSFeatureSelector(
    npy_proc.npy_file,
    drop_types=["uni", "disc"],
    on_train="bkg"
)

# Get the data and selection
file_path, features = npy_proc()
data = np.load(file_path)
selected_data, selection = f_sel(file_path, features)

# Print selected features and their first 10 values
selected_features = selection[selection['select'] == True]['feature'].tolist()
print("\nSelected features and their first 10 values:")
for i, feature in enumerate(selected_features):
    print(f"\nFeature: {feature}")
    print(f"First 10 values: {selected_data[:10, i]}")

