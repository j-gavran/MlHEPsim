"""Refactored Drell-Yan sample analyzer with modular structure."""

import sys
import os
# Get the project root (go up 4 levels from extras/ to MLHEPsim/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import json
import uproot
import logging
from ml.common.nn.gen_model_sampler import GenModelSampler
from ml.common.data_utils.feature_scaling import RescalingHandler
from ml.common.data_utils.processors import Preprocessor
from ml.custom.DrellYan.process_drellyan_dataset import DrellYanFeatureSelector, DrellYanNpyProcessor

# Import modular components
from mass_calculator import calculate_dimuon_invariant_mass, compute_masses_for_dataset
from pt_calculator import calculate_pt_negative_muon, compute_pt_negative_for_dataset
from system_calculator import calculate_system_variables, compute_system_variables_for_dataset
from plotting import (plot_feature_comparison, plot_invariant_mass, plot_pt_negative_comparison,
                     plot_correlation_comparison, plot_system_variables_comparison)


class DrellYanSampleAnalyzer:
    """Main analyzer class for generating and comparing Drell-Yan samples."""
    
    def __init__(self, data_dir, variables_json_path, model_name="MADEMOG_flow_model_gauss_rank", 
                 mass_cut_lower=None, mass_cut_upper=None, mass_range_min=None, mass_range_max=None):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        data_dir : str
            Path to the real data .npy file
        variables_json_path : str
            Path to the variables.json file
        model_name : str
            Name of the model to load for generation
        mass_cut_lower : float, optional
            Lower boundary of mass cut region (excluded from training)
        mass_cut_upper : float, optional
            Upper boundary of mass cut region (excluded from training)
        mass_range_min : float, optional
            Overall minimum mass to include in unpreprocessed data
        mass_range_max : float, optional
            Overall maximum mass to include in unpreprocessed data
        """
        self.data_dir = data_dir
        self.variables_json_path = variables_json_path
        self.model_name = model_name
        self.mass_cut_lower = mass_cut_lower
        self.mass_cut_upper = mass_cut_upper
        self.mass_range_min = mass_range_min
        self.mass_range_max = mass_range_max
        
        # Load variables configuration
        with open(variables_json_path, 'r') as f:
            self.variables = json.load(f)
        
        # Load real data with overall mass range cut (but not middle exclusion)
        self.real_data = self._load_unpreprocessed_data()
        
        # Load preprocessed data (WITH mass cuts) for fitting scalers
        # This ensures scalers match what the model was trained with
        if not os.path.exists(data_dir):
            logging.warning(f"Preprocessed data file not found: {data_dir}")
            logging.info("Creating preprocessed data file...")
            # This will trigger the creation of the file
            self._create_preprocessed_data()
        
        self.preprocessed_data = np.load(data_dir)
        
        # Initialize components
        self._setup_preprocessing()
        
        # Storage for generated data
        self.generated_data = None
        self.real_mass = None
        self.generated_mass = None
        self.real_system_vars = None
        self.generated_system_vars = None
    
    def _load_unpreprocessed_data(self):
        """Load raw data from ROOT files with overall mass range cut.
        
        Applies min/max mass cuts but NOT the middle exclusion region.
        Uses a cached numpy file if available for faster loading.
        """
        # Check for cached unpreprocessed data file
        cache_file = "ml/data/drellyan/DRELLYAN_unpreprocessed.npy"
        
        if os.path.exists(cache_file):
            logging.info(f"Loading unpreprocessed data from cache: {cache_file}")
            unpreprocessed_data = np.load(cache_file)
            logging.info(f"Loaded unpreprocessed data shape: {unpreprocessed_data.shape}")
            
            # Apply mass range filter if specified
            if self.mass_range_min is not None or self.mass_range_max is not None:
                unpreprocessed_data = self._apply_mass_range_filter(unpreprocessed_data)
            
            return unpreprocessed_data
        
        # Cache doesn't exist - load from ROOT files and save
        logging.info("Cache not found. Loading unpreprocessed data from ROOT files...")
        logging.info("This will be saved for faster loading next time.")
        
        # Get feature names
        features = [name for name, type_ in self.variables['colnames'].items() 
                   if type_ in ['cont', 'uni']]
        
        # File paths (same as in process_drellyan_dataset.py)
        file_paths = [
            "/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700789.Sh_2214_Zmumu_maxHTpTV2_BFilter_HmumuSR.root",
            "/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700790.Sh_2214_Zmumu_maxHTpTV2_CFilterBVeto_HmumuSR.root",
            "/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700791.Sh_2214_Zmumu_maxHTpTV2_CVetoBVeto_HmumuSR.root"
        ]
        
        all_data = []
        for file_path in file_paths:
            logging.info(f"Loading from {file_path}")
            with uproot.open(file_path) as file_drell_yan:
                tree_drell_yan = file_drell_yan["tree_Hmumu"]
                
                # Load ALL data without any mass cuts
                features_to_load = features + ["Muons_Minv_MuMu"]
                data = tree_drell_yan.arrays(features_to_load, library="np")
                
                # Stack features into matrix
                arrays = []
                for feat in features:
                    arr = np.asarray(data[feat])
                    # If dtype is 'object', it's a jagged array - extract scalar values
                    if arr.dtype == object:
                        arr = np.array([x[0] if isinstance(x, np.ndarray) else x for x in arr], dtype=np.float32)
                    arrays.append(arr)
                
                file_data = np.column_stack(arrays).astype(np.float32)
                all_data.append(file_data)
                logging.info(f"  Loaded {file_data.shape[0]} events")
        
        unpreprocessed_data = np.concatenate(all_data, axis=0)
        logging.info(f"Total unpreprocessed data shape: {unpreprocessed_data.shape}")
        
        # Apply eta cut: both muons must be within detector acceptance |eta| < 2.5
        # Assuming Muons_Pos_Eta is at index 0 and Muons_Neg_Eta is at index 1
        eta_mask = (
            (unpreprocessed_data[:, 0] >= -2.5) & (unpreprocessed_data[:, 0] <= 2.5) &  # Muons_Pos_Eta
            (unpreprocessed_data[:, 1] >= -2.5) & (unpreprocessed_data[:, 1] <= 2.5)    # Muons_Neg_Eta
        )
        unpreprocessed_data = unpreprocessed_data[eta_mask]
        logging.info(f"After eta cut |eta| < 2.5: {unpreprocessed_data.shape}")
        
        # Save to cache for next time
        np.save(cache_file, unpreprocessed_data)
        logging.info(f"Saved unpreprocessed data to cache: {cache_file}")
        
        # Apply mass range filter if specified
        if self.mass_range_min is not None or self.mass_range_max is not None:
            unpreprocessed_data = self._apply_mass_range_filter(unpreprocessed_data)
        
        return unpreprocessed_data
    
    def _apply_mass_range_filter(self, data):
        """Apply overall mass range filter to data (removes Z peak, keeps middle region)."""
        from mass_calculator import compute_masses_for_dataset, calculate_dimuon_invariant_mass
        
        # Get feature names
        features = [name for name, type_ in self.variables['colnames'].items() 
                   if type_ in ['cont', 'uni']]
        
        # Calculate mass for all events using the same method as compute_masses
        masses = compute_masses_for_dataset(
            data, features, self.variables, calculate_dimuon_invariant_mass
        )
        
        # Create filter mask
        mask = np.ones(len(masses), dtype=bool)
        if self.mass_range_min is not None:
            mask &= (masses >= self.mass_range_min)
            logging.info(f"Applying lower mass cut: >= {self.mass_range_min} GeV")
        if self.mass_range_max is not None:
            mask &= (masses <= self.mass_range_max)
            logging.info(f"Applying upper mass cut: <= {self.mass_range_max} GeV")
        
        filtered_data = data[mask]
        logging.info(f"After mass range filter: {filtered_data.shape[0]} events (removed {data.shape[0] - filtered_data.shape[0]} events)")
        
        return filtered_data
        
    def _create_preprocessed_data(self):
        """Create preprocessed data file if it doesn't exist."""
        # Get feature names
        features = [name for name, type_ in self.variables['colnames'].items() 
                   if type_ in ['cont', 'uni']]
        
        # Load config to get mass cuts
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'flows', 'data_config.yaml')
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        input_proc = data_config['data_config']['input_processing']
        
        # Initialize processor with current config
        npy_proc = DrellYanNpyProcessor(
            data_dir="ml/data/drellyan/", 
            base_file_name="DrellYan",
            list_data_features=features,
            mass_region=input_proc['mass_region'],
            min_mass=input_proc.get('min_mass', 110),
            max_mass=input_proc.get('max_mass', 160),
            sideband_lower_min=input_proc.get('sideband_lower_min', 100),
            sideband_lower_max=input_proc.get('sideband_lower_max', 122),
            sideband_upper_min=input_proc.get('sideband_upper_min', 128),
            sideband_upper_max=input_proc.get('sideband_upper_max', 180),
        )
        
        # Process and create the file
        npy_proc()
        logging.info(f"Created preprocessed data file: {npy_proc.npy_file}")
        
    def _setup_preprocessing(self):
        """Setup preprocessing pipeline and scalers.
        
        IMPORTANT: Scalers must be fitted on the SAME data distribution 
        that the model was trained on (i.e., preprocessed data with mass cuts).
        """
        # Get feature names
        features = [name for name, type_ in self.variables['colnames'].items() 
                   if type_ in ['cont', 'uni']]
        
        # Initialize processor and selector
        npy_proc = DrellYanNpyProcessor(
            data_dir="ml/data/drellyan/", 
            base_file_name="DrellYan", 
            list_data_features=features
        )
        
        f_sel = DrellYanFeatureSelector(
            file_path=npy_proc.npy_file, 
            features=self.variables,
            drop_types=['label', 'disc'],
        )
        
        # Get selection and scalers from PREPROCESSED data (with mass cuts)
        # This ensures scalers match what the model was trained with
        pre = Preprocessor(cont_rescale_type="gauss_rank")
        self.selection = f_sel._select_colnames()
        preprocessed_data_selected, self.selection, self.scalers = pre(self.preprocessed_data, self.selection)
        
        # Setup rescaling handler
        self.rescale_handler = RescalingHandler(self.selection, self.scalers)
        
        # Get selected feature names
        self.selected_features = self.selection[self.selection['select'] == True]['feature'].tolist()
        
        print("Feature selection details:")
        print(self.selection)
        print(f"\nUnpreprocessed real data shape: {self.real_data.shape}")
        print(f"Preprocessed data shape (for scalers): {self.preprocessed_data.shape}")
        print(f"Selected features: {len(self.selected_features)}")
        
    def generate_samples(self, n_samples=10000000, chunks=20, debug=False):
        """
        Generate samples from the trained model.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        chunks : int
            Number of chunks to split generation into
        debug : bool
            Whether to print debug information
            
        Returns
        -------
        np.ndarray
            Generated samples in original (physical) units
        """
        # Initialize sampler with disable_cache=True to always generate fresh samples
        sampler = GenModelSampler(
            model_names=self.model_name, 
            save_dir="ml/data/drellyan",
            file_name="DrellYan_generated",
            disable_cache=True  # When True always generate fresh samples (important after retraining)
        )
        
        # Generate samples
        print(f"\nGenerating {n_samples} samples...")
        generated_samples = sampler.sample(n_samples, chunks=chunks)
        generated_data = generated_samples[self.model_name][0]
        
        if debug:
            print("\n=== Before inverse_transform ===")
            for i, feature in enumerate(self.selected_features):
                feature_type = self.selection[self.selection['feature'] == feature]['type'].values[0]
                print(f"{feature} ({feature_type}): min={generated_data[:, i].min():.3f}, "
                      f"max={generated_data[:, i].max():.3f}, mean={generated_data[:, i].mean():.3f}, "
                      f"std={generated_data[:, i].std():.3f}")
        
        # Apply inverse transform
        self.generated_data = self.rescale_handler.inverse_transform(generated_data)
        
        # Apply same mass range filter as real data
        if self.mass_range_min is not None or self.mass_range_max is not None:
            print(f"\nApplying mass range filter to generated data...")
            original_count = self.generated_data.shape[0]
            self.generated_data = self._apply_mass_range_filter(self.generated_data)
            print(f"Generated data after mass filter: {self.generated_data.shape[0]} events (removed {original_count - self.generated_data.shape[0]})")
        
        if debug:
            print("\n=== After inverse_transform ===")
            for i, feature in enumerate(self.selected_features):
                feature_type = self.selection[self.selection['feature'] == feature]['type'].values[0]
                print(f"{feature} ({feature_type}): min={self.generated_data[:, i].min():.3f}, "
                      f"max={self.generated_data[:, i].max():.3f}, mean={self.generated_data[:, i].mean():.3f}, "
                      f"std={self.generated_data[:, i].std():.3f}")
        
        print(f"Generated data shape: {self.generated_data.shape}")
        return self.generated_data

    def get_existing_samples(self, sample_file = "/project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/data/drellyan/DrellYan_generated_MADEMOG_flow_model_gauss_rank.npy"):
        """
        Load existing generated samples from a .npy file.
        
        Parameters
        ----------
        sample_file : str
            Path to the .npy file containing generated samples
            
        Returns
        -------
        np.ndarray
            Loaded generated samples in original (physical) units
        """
        print(f"\nLoading existing samples from {sample_file}...")
        generated_data = np.load(sample_file)
        
        if generated_data.shape[1] != len(self.selected_features):
            raise ValueError("Loaded data shape does not match number of selected features.")
        
        self.generated_data = generated_data
        
        # Apply same mass range filter as real data
        if self.mass_range_min is not None or self.mass_range_max is not None:
            print(f"\nApplying mass range filter to loaded generated data...")
            original_count = self.generated_data.shape[0]
            self.generated_data = self._apply_mass_range_filter(self.generated_data)
            print(f"Generated data after mass filter: {self.generated_data.shape[0]} events (removed {original_count - self.generated_data.shape[0]})")
        
        print("\n=== Loaded existing samples ===")
        for i, feature in enumerate(self.selected_features):
            feature_type = self.selection[self.selection['feature'] == feature]['type'].values[0]
            print(f"{feature} ({feature_type}): min={self.generated_data[:, i].min():.3f}, "
                  f"max={self.generated_data[:, i].max():.3f}, mean={self.generated_data[:, i].mean():.3f}, "
                  f"std={self.generated_data[:, i].std():.3f}")
    
    def compute_masses(self):
        """Compute invariant masses for both real and generated data."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Calculate mass for generated data
        self.generated_mass = compute_masses_for_dataset(
            self.generated_data, self.selected_features, 
            self.variables, calculate_dimuon_invariant_mass
        )
        
        # Prepare real data in correct format (original physical values)
        real_data_for_mass = np.zeros((self.real_data.shape[0], len(self.selected_features)))
        for idx, feature in enumerate(self.selected_features):
            original_idx = list(self.variables['colnames'].keys()).index(feature)
            real_data_for_mass[:, idx] = self.real_data[:, original_idx]
        
        # Calculate mass for real data
        self.real_mass = compute_masses_for_dataset(
            real_data_for_mass, self.selected_features,
            self.variables, calculate_dimuon_invariant_mass
        )
        
        print(f"\n=== Invariant Mass Statistics ===")
        print(f"Real mass range:      [{self.real_mass.min():.3f}, {self.real_mass.max():.3f}] GeV")
        print(f"Generated mass range: [{self.generated_mass.min():.3f}, {self.generated_mass.max():.3f}] GeV")
        print(f"Real mass mean:       {self.real_mass.mean():.3f} GeV")
        print(f"Generated mass mean:  {self.generated_mass.mean():.3f} GeV")
        
        return self.real_mass, self.generated_mass
    
    def compute_pt_negative(self):
        """Compute PT of negative muon for both real and generated data."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Calculate PT for generated data
        self.generated_pt_negative = compute_pt_negative_for_dataset(
            self.generated_data, self.selected_features, 
            self.variables, calculate_pt_negative_muon
        )
        
        # Prepare real data in correct format (original physical values)
        real_data_for_pt = np.zeros((self.real_data.shape[0], len(self.selected_features)))
        for idx, feature in enumerate(self.selected_features):
            original_idx = list(self.variables['colnames'].keys()).index(feature)
            real_data_for_pt[:, idx] = self.real_data[:, original_idx]
        
        # Calculate PT for real data
        self.real_pt_negative = compute_pt_negative_for_dataset(
            real_data_for_pt, self.selected_features,
            self.variables, calculate_pt_negative_muon
        )
        
        print(f"\n=== PT Negative Muon Statistics ===")
        print(f"Real pt range:      [{self.real_pt_negative.min():.3f}, {self.real_pt_negative.max():.3f}] GeV")
        print(f"Generated pt range: [{self.generated_pt_negative.min():.3f}, {self.generated_pt_negative.max():.3f}] GeV")
        print(f"Real pt mean:       {self.real_pt_negative.mean():.3f} GeV")
        print(f"Generated pt mean:  {self.generated_pt_negative.mean():.3f} GeV")
        
        return self.real_pt_negative, self.generated_pt_negative

    def compute_system_variables(self):
        """Compute system-level kinematic variables for both real and generated data."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Calculate system variables for generated data
        self.generated_system_vars = compute_system_variables_for_dataset(
            self.generated_data, self.selected_features,
            self.variables, calculate_system_variables
        )
        
        # Prepare real data in correct format (original physical values)
        real_data_for_system = np.zeros((self.real_data.shape[0], len(self.selected_features)))
        for idx, feature in enumerate(self.selected_features):
            original_idx = list(self.variables['colnames'].keys()).index(feature)
            real_data_for_system[:, idx] = self.real_data[:, original_idx]
        
        # Calculate system variables for real data
        self.real_system_vars = compute_system_variables_for_dataset(
            real_data_for_system, self.selected_features,
            self.variables, calculate_system_variables
        )
        
        # Unpack for printing statistics
        real_Z_pt, real_Z_eta, real_Z_phi, real_Z_Y, real_cos_theta_star = self.real_system_vars
        gen_Z_pt, gen_Z_eta, gen_Z_phi, gen_Z_Y, gen_cos_theta_star = self.generated_system_vars
        
        print(f"\n=== System Variables Statistics ===")
        print(f"Z_PT:")
        print(f"  Real range:      [{real_Z_pt.min():.3f}, {real_Z_pt.max():.3f}] GeV")
        print(f"  Generated range: [{gen_Z_pt.min():.3f}, {gen_Z_pt.max():.3f}] GeV")
        print(f"Z_eta:")
        print(f"  Real range:      [{real_Z_eta.min():.3f}, {real_Z_eta.max():.3f}]")
        print(f"  Generated range: [{gen_Z_eta.min():.3f}, {gen_Z_eta.max():.3f}]")
        print(f"Z_phi:")
        print(f"  Real range:      [{real_Z_phi.min():.3f}, {real_Z_phi.max():.3f}]")
        print(f"  Generated range: [{gen_Z_phi.min():.3f}, {gen_Z_phi.max():.3f}]")
        print(f"Z_Y (rapidity):")
        print(f"  Real range:      [{real_Z_Y.min():.3f}, {real_Z_Y.max():.3f}]")
        print(f"  Generated range: [{gen_Z_Y.min():.3f}, {gen_Z_Y.max():.3f}]")
        print(f"cos(theta*) Collins-Soper:")
        print(f"  Real range:      [{real_cos_theta_star.min():.3f}, {real_cos_theta_star.max():.3f}]")
        print(f"  Generated range: [{gen_cos_theta_star.min():.3f}, {gen_cos_theta_star.max():.3f}]")
        
        return self.real_system_vars, self.generated_system_vars

    def plot_feature_comparison(self, output_path=None, bins=50, n_cols=3):
        """Plot comparison of all features between real and generated data."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'feature_comparison.png')
        
        plot_feature_comparison(
            self.real_data, self.generated_data, self.selected_features,
            self.selection, self.variables, output_path, bins, n_cols,
            mass_cut_lower=self.mass_cut_lower, mass_cut_upper=self.mass_cut_upper
        )
        
    def plot_invariant_mass(self, output_path=None, bins=100, x_min=100, x_max=160):
        """Plot comparison of invariant mass distributions."""
        if self.real_mass is None or self.generated_mass is None:
            raise ValueError("No mass data available. Call compute_masses() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'invariant_mass_comparison.png')
        
        plot_invariant_mass(
            self.real_mass, self.generated_mass, output_path, bins, x_min, x_max,
            mass_cut_lower=self.mass_cut_lower, mass_cut_upper=self.mass_cut_upper
        )

    def plot_pt_negative_comparison(self, output_path=None, bins=100, x_min=0, x_max=100):
        """Plot comparison of PT negative muon distributions."""
        if self.real_pt_negative is None or self.generated_pt_negative is None:
            raise ValueError("No PT negative muon data available. Call compute_pt_negative() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'pt_negative_comparison.png')
        
        plot_pt_negative_comparison(
            self.real_pt_negative, self.generated_pt_negative, output_path, bins
        )

    def plot_correlation_plots(self, output_path=None, gridsize=200):
        """Plot pairwise correlations with difference maps."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'correlation_comparison.png')
        
        plot_correlation_comparison(
            self.real_data, self.generated_data, self.selected_features,
            self.variables, gridsize, output_path,
            mass_cut_lower=self.mass_cut_lower, mass_cut_upper=self.mass_cut_upper
        )

    def plot_system_variables_comparison(self, output_path=None, bins=50, log_scale=True):
        """Plot comparison of system-level kinematic variables."""
        if self.real_system_vars is None or self.generated_system_vars is None:
            raise ValueError("No system variables available. Call compute_system_variables() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            suffix = '_log' if log_scale else ''
            output_path = os.path.join(figures_dir, f'system_variables_comparison{suffix}.png')
        
        plot_system_variables_comparison(
            self.real_system_vars, self.generated_system_vars, output_path, bins, log_scale,
            mass_cut_lower=self.mass_cut_lower, mass_cut_upper=self.mass_cut_upper
        )
