from ml_hep_sim.data_utils.higgs.higgs_dataset import HiggsDataModule
from ml_hep_sim.data_utils.mnist.mnist_dataset import MnistDataModule
from ml_hep_sim.data_utils.toy_datasets import TOY_DATASETS, ToyDataModule


class DataModuleBuilder:
    def __init__(self, dataset_name, data_dir, data_param_dict, data_paths=None, **kwargs):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.data_param_dict = data_param_dict
        self.data_paths = data_paths

    def get_data_module(self):
        if self.dataset_name.lower() == "mnist":
            self.data_paths = [
                self.data_dir + "data/mnist/train.npy",
                self.data_dir + "data/mnist/test.npy",
            ]
            data_module = MnistDataModule(
                self.data_paths,
                **self.data_param_dict,
            )
        elif self.dataset_name.lower() == "higgs":
            self.data_paths = [
                self.data_dir + "data/higgs/HIGGS_18_feature_train.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_val.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_test.npy",
            ]
            data_module = HiggsDataModule(
                self.data_paths,
                **self.data_param_dict,
            )
        elif self.dataset_name.lower() == "higgs_bkg":
            self.data_paths = [
                self.data_dir + "data/higgs/HIGGS_18_feature_bkg_train.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_bkg_val.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_bkg_test.npy",
            ]
            data_module = HiggsDataModule(
                self.data_paths,
                **self.data_param_dict,
            )
        elif self.dataset_name.lower() == "higgs_sig":
            self.data_paths = [
                self.data_dir + "data/higgs/HIGGS_18_feature_sig_train.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_sig_val.npy",
                self.data_dir + "data/higgs/HIGGS_18_feature_sig_test.npy",
            ]
            data_module = HiggsDataModule(
                self.data_paths,
                **self.data_param_dict,
            )
        elif self.dataset_name in TOY_DATASETS:
            data_module = ToyDataModule(self.dataset_name, **self.data_param_dict)
        else:
            return None

        return data_module
