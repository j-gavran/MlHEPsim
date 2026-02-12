import uproot
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml.common.data_utils.processors import FeatureSelector, NpyProcessor
from ml.common.data_utils.utils import url_download
from ml.common.nn.gen_model_sampler import GenModelSampler

class DrellYanNpyProcessor(NpyProcessor):
    def __init__(
        self,
        data_dir,
        list_data_features,
        base_file_name="DRELLYAN",
        keep_ratio=1.0,
        shuffle=True,
        hold_mode=False,
        use_hold=False,
        hold_ratio=0.2,
        min_mass=110,
        max_mass=160,
        mass_region="full_data",
        sideband_lower_min=100,
        sideband_lower_max=115,
        sideband_upper_min=135,
        sideband_upper_max=200,
        cut=None
    ):
        """Drell Yan dataset to .npy starting processor.

        Note
        ----
        Supports holdout mode, where data is split into two partitions. Partition 1 is used for training and partition 2
        is used for independent holdout evaluation.

        Parameters
        ----------
        keep_ratio : float, optional
            Keep only a fraction of the data, by default 1.0.
        shuffle : bool, optional
            Shuffle data loaded from starting file, by default True.
        hold_mode : bool, optional
            Holdout mode to use partition_1 of partition_2 of holdout data, by default False.
        use_hold : bool, optional
            If True use partition_1 else use partition_2, by default False.
        hold_ratio : float, optional
            Ratio of holdout data in partition_2, by default 0.2.
        min_mass : float, optional
            Minimum invariant mass cut for signal region, by default 110.
        max_mass : float, optional
            Maximum invariant mass cut for signal region, by default 160.
        mass_region : str, optional
            Mass region to use: 'full_data' or 'sidebands', by default 'full_data'.
        sideband_lower_min : float, optional
            Lower sideband minimum mass, by default 80.
        sideband_lower_max : float, optional
            Lower sideband maximum mass, by default 100.
        sideband_upper_min : float, optional
            Upper sideband minimum mass, by default 170.
        sideband_upper_max : float, optional
            Upper sideband maximum mass, by default 200.
        """
        super().__init__(data_dir, base_file_name)
        self.file_name = None
        self.keep_ratio = keep_ratio
        self.shuffle = shuffle
        self.hold_mode, self.use_hold, self.hold_ratio = hold_mode, use_hold, 1 - hold_ratio
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.mass_region = mass_region
        self.sideband_lower_min = sideband_lower_min
        self.sideband_lower_max = sideband_lower_max
        self.sideband_upper_min = sideband_upper_min
        self.sideband_upper_max = sideband_upper_max
        self.list_data_features = list_data_features
        
        # Override self.features from config instead of loading from variables.json
        # All DrellYan features are continuous variables
        self.features = {
            "colnames": {feature: "cont" for feature in list_data_features}
        }

        if self.hold_mode:
            self.hold_npy_partition_1 = self.npy_file.replace(".npy", "_hold_partition_1.npy")
            self.hold_npy_partition_2 = self.npy_file.replace(".npy", "_hold_partition_2.npy")
    
    def _select_npy_file(self):
        if self.hold_mode and not self.use_hold:
            logging.info(f"Using holdout partition 1 from {self.hold_npy_partition_1}!")
            return self.hold_npy_partition_1, self.features
        elif self.hold_mode and self.use_hold:
            logging.info(f"Using holdout partition 2 from {self.hold_npy_partition_2}!")
            return self.hold_npy_partition_2, self.features
        else:
            logging.info(f"Using {self.npy_file}!")
            return self.npy_file, self.features

    def __call__(self, *args, **kwargs):
        dataset = self.get_dataset()
        
        if dataset is None:
            logging.info(f"{self.npy_file} already exists!")
            return self._select_npy_file()

        dataset = self.process_dataset(dataset)

        if self.hold_mode:
            hold_idx = int(len(dataset) * self.hold_ratio)

            hold_dataset = dataset[:hold_idx]
            dataset = dataset[hold_idx:]

            self.make_npy_file(dataset, self.hold_npy_partition_1)
            self.make_npy_file(hold_dataset, self.hold_npy_partition_2)
        else:
            dataset = self.process_dataset(dataset)
            self.make_npy_file(dataset, self.npy_file)

        return self._select_npy_file()

    def get_dataset(self):
        """Creates DrellYan dataset dataframe if not existing yet, otherwise loads existing .npy file.
        Parameters
        ----------
        data_dir : str, optional
            Path to higgs data, by default "data/".

        Returns
        -------
        pd.DataFrame
            29 dim dataframe of all downloaded data.

        """
    
        # Always regenerate dataset to ensure mass cuts are up to date
        logging.info(f"⚠️ Regenerating dataset to apply current mass cuts")
        
        # Delete old files if they exist
        if Path(self.npy_file).is_file():
            Path(self.npy_file).unlink()
            logging.info(f"Deleted old {self.npy_file}")
        
        if self.hold_mode:
            if Path(self.hold_npy_partition_1).is_file():
                Path(self.hold_npy_partition_1).unlink()
                logging.info(f"Deleted old {self.hold_npy_partition_1}")
            if Path(self.hold_npy_partition_2).is_file():
                Path(self.hold_npy_partition_2).unlink()
                logging.info(f"Deleted old {self.hold_npy_partition_2}")

        # Create .npy file from ROOT files and return dataframe
        self.create_dataset(
            list_data_features=self.list_data_features,
            mass_region=self.mass_region,
            min_mass=self.min_mass,
            max_mass=self.max_mass,
            sideband_lower_min=self.sideband_lower_min,
            sideband_lower_max=self.sideband_lower_max,
            sideband_upper_min=self.sideband_upper_min,
            sideband_upper_max=self.sideband_upper_max,
        )
        try:
            data = np.load(self.npy_file, allow_pickle=True)
            return pd.DataFrame(data)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load {self.npy_file}")


    def create_dataset(self, list_data_features, mass_region, min_mass, max_mass,
                      sideband_lower_min, sideband_lower_max, 
                      sideband_upper_min, sideband_upper_max):
        """Creates Drell-Yan dataset from ROOT files.
        
        Parameters
        ----------
        mass_region : str
            'full_data' for full data or 'sidebands' for sideband regions
        """

        logging.info("Loading Drell-Yan dataset from ROOT files!")

        # File paths
        file_path_1 = ("/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700889.Sh_2214_Zmumu_mZ_105_ECMS_BFilter_HmumuSR_skimmed_prepared_FSR.root")
        file_path_2 = ("/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700890.Sh_2214_Zmumu_mZ_105_ECMS_CFilterBVeto_HmumuSR_skimmed_prepared_FSR.root")
        file_path_3 = ("/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700891.Sh_2214_Zmumu_mZ_105_ECMS_CVetoBVeto_HmumuSR_skimmed_prepared_FSR.root")

        file_path_4 = ("/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700789.Sh_2214_Zmumu_maxHTpTV2_BFilter_HmumuSR.root")
        file_path_5 = ("/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700790.Sh_2214_Zmumu_maxHTpTV2_CFilterBVeto_HmumuSR.root")
        file_path_6 = ("/dcache/atlas/higgs/Hmumu/RunIII/NTuple_MC23a/mc23_13p6TeV.700791.Sh_2214_Zmumu_maxHTpTV2_CVetoBVeto_HmumuSR.root")

        # Load and concatenate data from all files
        all_filtered_data = []
        counter = 0
        for file_path in [file_path_4, file_path_5, file_path_6]:
            with uproot.open(file_path) as file_drell_yan:
                tree_drell_yan = file_drell_yan["tree_Hmumu"]

                # Mass window cut - full data or sidebands
                if mass_region == "full_data":
                    mass_cut = f"(Muons_Minv_MuMu >= {min_mass}) & (Muons_Minv_MuMu <= {max_mass})"
                    if counter == 0:
                        logging.info(f"Using FULL DATA: [{min_mass}, {max_mass}] GeV")
                elif mass_region == "sidebands":
                    mass_cut = (f"((Muons_Minv_MuMu >= {sideband_lower_min}) & (Muons_Minv_MuMu <= {sideband_lower_max})) | "
                                    f"((Muons_Minv_MuMu >= {sideband_upper_min}) & (Muons_Minv_MuMu <= {sideband_upper_max}))")
                    if counter == 0:
                        logging.info(f"Using SIDEBANDS: [{sideband_lower_min}, {sideband_lower_max}] GeV and [{sideband_upper_min}, {sideband_upper_max}] GeV")
                else:
                    raise ValueError(f"Invalid mass_region: {mass_region}. Must be 'full_data' or 'sidebands'.")

                if counter == 0:
                    logging.info(f"Will apply eta cut: |eta| < 2.5 for both muons after loading")
                    counter += 1

                # Load requested features with only mass cut
                # Eta cut will be applied after flattening jagged arrays
                features_to_load = list(set(list_data_features + ["Muons_Minv_MuMu"]))
                data = tree_drell_yan.arrays(features_to_load, library="np", cut=mass_cut)
    
            # Stack features into matrix and flatten jagged arrays
            arrays = []
            for feat in list_data_features:
                arr = np.asarray(data[feat])
                # If dtype is 'object', it's a jagged array - extract the scalar values
                if arr.dtype == object:
                    arr = np.array([x[0] if isinstance(x, np.ndarray) else x for x in arr], dtype=np.float32)
                arrays.append(arr)
            
            self.x = np.column_stack(arrays).astype(np.float32)
            
            # Apply eta cut: both muons must be within detector acceptance |eta| < 2.5
            eta_mask = (
                (self.x[:, 0] >= -2.5) & (self.x[:, 0] <= 2.5) &  # Muons_Pos_Eta
                (self.x[:, 1] >= -2.5) & (self.x[:, 1] <= 2.5)    # Muons_Neg_Eta
            )
            self.x = self.x[eta_mask]
            
            all_filtered_data.append(self.x) 

        dataset = np.concatenate(all_filtered_data, axis=0)
        logging.info(f"Final Drell-Yan dataset shape after cuts: {dataset.shape}")
        
        # Shuffle the dataset before saving to ensure random sampling 
        np.random.shuffle(dataset)
        logging.info("Shuffled dataset before saving")

        np.save(self.npy_file, dataset)
        logging.info(f"saved {self.npy_file} of shape {dataset.shape}!")


    def process_dataset(self, dataset):
        logging.info("Processing dataset!")

        if self.shuffle:
            logging.info("Shuffling data!")
            dataset = dataset.sample(frac=1).reset_index(drop=True)

        if self.keep_ratio < 1.0:
            logging.info(f"Keeping only {self.keep_ratio:.2f} of data!")
            dataset = dataset[: int(len(dataset) * self.keep_ratio)]

        return dataset

    def make_npy_file(self, dataset, npy_file=None):
        if npy_file is None:
            npy_file = self.npy_file
        
        # Convert DataFrame to numpy array if needed
        if hasattr(dataset, 'values'):
            dataset = dataset.values
        
        np.save(npy_file, dataset)
        logging.info(f"Saved {npy_file} with shape {dataset.shape}")

    def download(self):
        pass


class DrellYanFeatureSelector(FeatureSelector):
    def __init__(self, file_path, n_data=None, **kwargs):
        super().__init__(file_path, **kwargs)
        self.n_data = n_data

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}!")
        data = np.load(self.file_path)

        if self.n_data is not None:
            data = data[: self.n_data]
            logging.info(f"Using {self.n_data} data points!")

        return data

    def select_features(self, data):
        return super().select_features(data)