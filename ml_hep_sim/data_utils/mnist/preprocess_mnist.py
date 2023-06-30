import os
from pathlib import Path

import numpy as np
from ml_hep_sim.data_utils.mnist.python_mnist_parser.mnist.loader import MNIST

# change this if mnist folder elsewhere
data_path = str(Path(__file__).resolve().parents[3]) + "/data"

ROOT_FOLDER = data_path


def load_mnist_data(flag="training"):
    """MNIST loader helper function."""
    mndata = MNIST(os.path.join(ROOT_FOLDER, "mnist"))
    try:
        if flag == "training":
            images, labels = mndata.load_training()
        elif flag == "testing":
            images, labels = mndata.load_testing()
        else:
            raise Exception("Flag should be either training or testing.")
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images)
    images_array = np.concatenate(images_array, 0)

    labels_array = np.array(labels)

    return images_array.astype(np.uint8), labels_array


def preprocess_mnist():
    """Make MNIST npy data:

    - Download 4 ubyte.gz files from http://yann.lecun.com/exdb/mnist/.
    - Unzip into mnist folder.
    - Move unziped files into data/ directory.
    - Run this function (need https://github.com/sorki/python-mnist).

    Returns test and train npy files in data/mnist folder.

    """
    x_train, labels_train = load_mnist_data("training")
    x_train = np.reshape(x_train, [60000, 28, 28, 1])

    np.save(os.path.join("data", "mnist", "train.npy"), x_train)
    np.save(os.path.join("data", "mnist", "train_labels.npy"), labels_train)

    x_test, labels_test = load_mnist_data("testing")
    x_test = np.reshape(x_test, [10000, 28, 28, 1])

    np.save(os.path.join("data", "mnist", "test.npy"), x_test)
    np.save(os.path.join("data", "mnist", "test_labels.npy"), labels_test)


if __name__ == "__main__":
    preprocess_mnist()

    # modified from:
    # - https://github.com/daib13/TwoStageVAE/blob/master/preprocess.py
    # - https://github.com/asperti/BalancingVAE/blob/master/computed_gamma.py
