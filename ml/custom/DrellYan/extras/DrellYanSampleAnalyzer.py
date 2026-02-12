import sys
import os
# Get the project root (go up 4 levels from extras/ to MLHEPsim/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt 
import numpy as np
import json
from ml.common.nn.gen_model_sampler import GenModelSampler
from ml.common.data_utils.feature_scaling import RescalingHandler
from ml.common.data_utils.processors import Preprocessor
from ml.custom.DrellYan.process_drellyan_dataset import DrellYanFeatureSelector, DrellYanNpyProcessor


class DrellYanSampleAnalyzer:
    """Class for generating, analyzing, and comparing Drell-Yan samples."""
    
    def __init__(self, data_dir, variables_json_path, model_name="MADEMOG_flow_model_gauss_rank"):
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
        """
        self.data_dir = data_dir
        self.variables_json_path = variables_json_path
        self.model_name = model_name
        
        # Load variables configuration
        with open(variables_json_path, 'r') as f:
            self.variables = json.load(f)
        
        # Load real data
        self.real_data = np.load(data_dir)
        
        # Initialize components
        self._setup_preprocessing()
        
        # Storage for generated data
        self.generated_data = None
        self.real_mass = None
        self.generated_mass = None
        
    def _setup_preprocessing(self):
        """Setup preprocessing pipeline and scalers."""
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
        
        # Get selection and scalers from real data
        pre = Preprocessor(cont_rescale_type="gauss_rank")
        self.selection = f_sel._select_colnames()
        real_data_selected, self.selection, self.scalers = pre(self.real_data, self.selection)
        
        # Setup rescaling handler
        self.rescale_handler = RescalingHandler(self.selection, self.scalers)
        
        # Get selected feature names
        self.selected_features = self.selection[self.selection['select'] == True]['feature'].tolist()
        
        print("Feature selection details:")
        print(self.selection)
        print(f"\nReal data shape: {self.real_data.shape}")
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
            disable_cache=True  # Always generate fresh samples (important after retraining)
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
        
        if debug:
            print("\n=== After inverse_transform ===")
            for i, feature in enumerate(self.selected_features):
                feature_type = self.selection[self.selection['feature'] == feature]['type'].values[0]
                print(f"{feature} ({feature_type}): min={self.generated_data[:, i].min():.3f}, "
                      f"max={self.generated_data[:, i].max():.3f}, mean={self.generated_data[:, i].mean():.3f}, "
                      f"std={self.generated_data[:, i].std():.3f}")
        
        print(f"Generated data shape: {self.generated_data.shape}")
        return self.generated_data
    
    def calculate_invariant_mass(self, data, pt1_idx, pt2_idx, eta1_idx, eta2_idx, phi1_idx, phi2_idx):
        """
        Calculate dimuon invariant mass from kinematics.
        
        Formula: M^2 = 2*pT1*pT2*(cosh(eta1 - eta2) - cos(phi1 - phi2))
        
        Parameters
        ----------
        data : np.ndarray
            Data array containing kinematic variables
        pt1_idx, pt2_idx : int
            Indices for leading and subleading muon pT
        eta1_idx, eta2_idx : int
            Indices for leading and subleading muon eta
        phi1_idx, phi2_idx : int
            Indices for leading and subleading muon phi
            
        Returns
        -------
        np.ndarray
            Invariant mass values in GeV
        """
        pt1 = data[:, pt1_idx]
        pt2 = data[:, pt2_idx]
        eta1 = data[:, eta1_idx]
        eta2 = data[:, eta2_idx]
        phi1 = data[:, phi1_idx]
        phi2 = data[:, phi2_idx]
        
        m_squared = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
        # Avoid negative values due to numerical precision
        m_squared = np.maximum(m_squared, 0)
        return np.sqrt(m_squared)
    
    def compute_masses(self):
        """Compute invariant masses for both real and generated data."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Get feature indices
        pt_lead_idx = self.selected_features.index('Muons_PT_Lead')
        pt_sub_idx = self.selected_features.index('Muons_PT_Sub')
        eta_lead_idx = self.selected_features.index('Muons_Eta_Lead')
        eta_sub_idx = self.selected_features.index('Muons_Eta_Sub')
        phi_lead_idx = self.selected_features.index('Muons_Phi_Lead')
        phi_sub_idx = self.selected_features.index('Muons_Phi_Sub')
        
        # Calculate mass for generated data
        self.generated_mass = self.calculate_invariant_mass(
            self.generated_data, pt_lead_idx, pt_sub_idx, 
            eta_lead_idx, eta_sub_idx, phi_lead_idx, phi_sub_idx
        )
        
        # Prepare real data in correct format (original physical values)
        real_data_for_mass = np.zeros((self.real_data.shape[0], len(self.selected_features)))
        for idx, feature in enumerate(self.selected_features):
            original_idx = list(self.variables['colnames'].keys()).index(feature)
            real_data_for_mass[:, idx] = self.real_data[:, original_idx]
        
        # Calculate mass for real data
        self.real_mass = self.calculate_invariant_mass(
            real_data_for_mass, pt_lead_idx, pt_sub_idx,
            eta_lead_idx, eta_sub_idx, phi_lead_idx, phi_sub_idx
        )
        
        print(f"\n=== Invariant Mass Statistics ===")
        print(f"Real mass range:      [{self.real_mass.min():.3f}, {self.real_mass.max():.3f}] GeV")
        print(f"Generated mass range: [{self.generated_mass.min():.3f}, {self.generated_mass.max():.3f}] GeV")
        print(f"Real mass mean:       {self.real_mass.mean():.3f} GeV")
        print(f"Generated mass mean:  {self.generated_mass.mean():.3f} GeV")
        
        return self.real_mass, self.generated_mass
    
    def plot_feature_comparison(self, output_path=None, bins=50, n_cols=3):
        """
        Plot comparison of all features between real and generated data.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, saves in script directory.
        bins : int
            Number of bins for histograms
        n_cols : int
            Number of columns in the subplot grid
        """
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Calculate grid dimensions
        n_features = len(self.selected_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axs = axs.flatten() if n_features > 1 else [axs]
        
        # Plot each feature
        for i, (feature, ax) in enumerate(zip(self.selected_features, axs)):
            feature_type = self.selection[self.selection['feature'] == feature]['type'].values[0]
            
            # Get real data (original physical values)
            original_feature_idx = list(self.variables['colnames'].keys()).index(feature)
            real_feature_data = self.real_data[:, original_feature_idx]
            
            # Get generated data (already inverse-transformed)
            generated_feature_data = self.generated_data[:, i]
            
            print(f"\n{feature} ({feature_type}):")
            print(f"  Real range:      [{real_feature_data.min():.3f}, {real_feature_data.max():.3f}]")
            print(f"  Generated range: [{generated_feature_data.min():.3f}, {generated_feature_data.max():.3f}]")
            
            # Calculate histogram range
            x_min = min(self._get_range_limits(real_feature_data)[0], 
                       self._get_range_limits(generated_feature_data)[0])
            x_max = max(self._get_range_limits(real_feature_data)[1], 
                       self._get_range_limits(generated_feature_data)[1])
            
            # Create histograms
            hist_real, bins_real = np.histogram(real_feature_data, bins=bins, 
                                               range=(x_min, x_max), density=True)
            hist_gen, bins_gen = np.histogram(generated_feature_data, bins=bins, 
                                             range=(x_min, x_max), density=True)
            
            # Plot
            bin_centers_real = (bins_real[:-1] + bins_real[1:]) / 2
            bin_centers_gen = (bins_gen[:-1] + bins_gen[1:]) / 2
            ax.step(bin_centers_real, hist_real, color='blue', label='Real Data', where='mid', lw=2)
            ax.step(bin_centers_gen, hist_gen, color='red', label='Generated', where='mid', lw=2)
            
            # Formatting
            y_max = max(max(hist_real), max(hist_gen)) * 1.1
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        
        # Remove empty subplots
        for j in range(n_features, len(axs)):
            axs[j].remove()
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, 'feature_comparison.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFeature comparison plot saved to: {output_path}")
        
    def plot_invariant_mass(self, output_path=None, bins=100):
        """
        Plot comparison of invariant mass distributions.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, saves in script directory.
        bins : int
            Number of bins for histogram
        """
        if self.real_mass is None or self.generated_mass is None:
            raise ValueError("No mass data available. Call compute_masses() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        x_min = 100
        x_max = 160

        # Create histograms
        hist_real, bins_real = np.histogram(self.real_mass, bins=bins, 
                                           range=(x_min, x_max), density=True)
        hist_gen, bins_gen = np.histogram(self.generated_mass, bins=bins, 
                                         range=(x_min, x_max), density=True)
        
        # Plot
        bin_centers_real = (bins_real[:-1] + bins_real[1:]) / 2
        bin_centers_gen = (bins_gen[:-1] + bins_gen[1:]) / 2
        ax.step(bin_centers_real, hist_real, color='blue', label='Real Data', where='mid', lw=2)
        ax.step(bin_centers_gen, hist_gen, color='red', label='Generated', where='mid', lw=2)
        
        # Formatting
        y_max = max(max(hist_real), max(hist_gen)) * 1.1
        ax.set_xlabel('Dimuon Invariant Mass [GeV]', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_title('Dimuon Invariant Mass Distribution', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, 'invariant_mass_comparison.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Invariant mass plot saved to: {output_path}")

    def plot_correlation_plots(self, output_path=None, sample_size=10000000, kind='hex', gridsize=200):
        """
        Plot pairwise correlations for (pt1 vs pt2), (eta1 vs eta2), (phi1 vs phi2).
        
        Creates a 3x3 grid: real data (left), generated data (middle), difference (right).
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, saves in script directory.
        sample_size : int
            Maximum number of points to plot (randomly subsampled if larger)
        kind : {'hex', 'scatter'}
            Plot type: 'hex' for hexbin density, 'scatter' for scatter plot
        gridsize : int
            Gridsize for hexbin plots (when kind='hex')
        """
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        # Feature pairs to plot
        pairs = [
            ("Muons_PT_Lead", "Muons_PT_Sub"),
            ("Muons_Eta_Lead", "Muons_Eta_Sub"),
            ("Muons_Phi_Lead", "Muons_Phi_Sub"),
        ]
        
        # Helper to get real data column
        orig_idx_map = {f: i for i, f in enumerate(list(self.variables['colnames'].keys()))}
        
        def get_real_col(feature):
            return self.real_data[:, orig_idx_map[feature]]
        
        # Prepare data pairs
        real_pairs = []
        gen_pairs = []
        
        for feat1, feat2 in pairs:
            # Real data (original physical units)
            real_pairs.append((get_real_col(feat1), get_real_col(feat2)))
            
            # Generated data (already inverse-transformed)
            idx1 = self.selected_features.index(feat1)
            idx2 = self.selected_features.index(feat2)
            gen_pairs.append((self.generated_data[:, idx1], self.generated_data[:, idx2]))
        
        # Subsampling helper
        def subsample(x, y, n):
            N = len(x)
            if N > n:
                indices = np.random.choice(N, n, replace=False)
                return x[indices], y[indices]
            return x, y
        
        # Create figure with 3 columns: real, generated, difference
        fig, axs = plt.subplots(3, 3, figsize=(18, 14))
        
        for row, ((x_real, y_real), (x_gen, y_gen), (label1, label2)) in enumerate(
            zip(real_pairs, gen_pairs, pairs)
        ):
            # Subsample for plotting
            x_real_s, y_real_s = subsample(x_real, y_real, sample_size)
            x_gen_s, y_gen_s = subsample(x_gen, y_gen, sample_size)
            
            ax_real = axs[row, 0]
            ax_gen = axs[row, 1]
            ax_diff = axs[row, 2]
            
            # Determine common bins for difference calculation
            x_min = min(x_real.min(), x_gen.min())
            x_max = max(x_real.max(), x_gen.max())
            y_min = min(y_real.min(), y_gen.min())
            y_max = max(y_real.max(), y_gen.max())
            
            # Plot real data
            if kind == 'hex':
                hb_real = ax_real.hexbin(x_real_s, y_real_s, gridsize=gridsize, cmap='Blues', mincnt=1,
                                        extent=(x_min, x_max, y_min, y_max))
            else:
                ax_real.scatter(x_real_s, y_real_s, s=1, alpha=0.3, color='blue')
            
            # Plot generated data
            if kind == 'hex':
                hb_gen = ax_gen.hexbin(x_gen_s, y_gen_s, gridsize=gridsize, cmap='Reds', mincnt=1,
                                      extent=(x_min, x_max, y_min, y_max))
            else:
                ax_gen.scatter(x_gen_s, y_gen_s, s=1, alpha=0.3, color='red')
            
            # Calculate difference: 2D histogram (Real - Generated)
            # Use more samples for better statistics in difference plot
            n_bins = gridsize
            hist_real, xedges, yedges = np.histogram2d(x_real, y_real, bins=n_bins, 
                                                       range=[[x_min, x_max], [y_min, y_max]], 
                                                       density=True)
            hist_gen, _, _ = np.histogram2d(x_gen, y_gen, bins=n_bins, 
                                           range=[[x_min, x_max], [y_min, y_max]], 
                                           density=True)
            
            # Difference: Real - Generated (positive means more real data, negative means more generated)
            diff = hist_real - hist_gen
            
            # Plot difference with diverging colormap
            im = ax_diff.imshow(diff.T, origin='lower', aspect='auto', 
                               extent=[x_min, x_max, y_min, y_max],
                               cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            plt.colorbar(im, ax=ax_diff, label='Real - Generated')
            
            # Formatting
            ax_real.set_title(f"Real: {label1} vs {label2}", fontsize=12)
            ax_gen.set_title(f"Generated: {label1} vs {label2}", fontsize=12)
            ax_diff.set_title(f"Difference: {label1} vs {label2}", fontsize=12)
            
            for ax in [ax_real, ax_gen, ax_diff]:
                ax.set_xlabel(label1, fontsize=10)
                ax.set_ylabel(label2, fontsize=10)
                ax.grid(True, alpha=0.4, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, 'correlation_comparison.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation comparison plot saved to: {output_path}")
    
    @staticmethod
    def _get_range_limits(data, percentile_range=0.1):
        """Calculate range limits excluding extreme outliers."""
        lower = np.percentile(data, percentile_range)
        upper = np.percentile(data, 100 - percentile_range)
        data_range = upper - lower
        # Add padding
        lower = lower - 0.1 * data_range
        upper = upper + 0.1 * data_range
        return lower, upper


# Main execution
if __name__ == "__main__":
    # Configuration
    data_dir = "/project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/data/drellyan/DRELLYAN.npy"
    variables_json = 'ml/data/drellyan/variables.json'
    model_name = "MADEMOG_flow_model_gauss_rank"
    
    # Initialize analyzer
    analyzer = DrellYanSampleAnalyzer(data_dir, variables_json, model_name)
    
    # Generate samples
    analyzer.generate_samples(n_samples=10000000, chunks=20, debug=True)
    
    # Compute invariant masses
    analyzer.compute_masses()
    
    # Create plots
    analyzer.plot_feature_comparison()
    analyzer.plot_invariant_mass()
    analyzer.plot_correlation_plots()