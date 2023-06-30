import logging
import time

import torch

from ml_hep_sim.pipeline.pipes import Block

# https://docs.rapids.ai/api/dask-cuda/nightly/install/


class IncBlock(Block):
    def __init__(self, x_in):
        super().__init__()
        self.x_in = x_in

        self.x = None

    def run(self, *args):
        logging.warning(f"on {self} at {time.time()}\n")
        self.setup(*args)

        time.sleep(4)

        self.x = self.x_in + 1

        logging.warning(self.__class__.__name__, time.time(), self.x, "\n")

        return self


class DoubleBlock(Block):
    def __init__(self, x_in):
        super().__init__()
        self.x_in = x_in

        self.y = None

    def run(self, *args):
        logging.warning(f"on {self} at {time.time()}")
        self.setup(*args)

        time.sleep(5)

        self.y = self.x_in * 2

        logging.warning(self.__class__.__name__, time.time(), self.y, "\n")

        return self


class SquareBlock(Block):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def run(self, *args):
        logging.warning(f"on {self} at {time.time()}")
        self.setup(*args)

        time.sleep(3)

        if self.x:
            self.x = self.x**2
            self.y = False
        if self.y:
            self.y = self.y**2
            self.x = False

        logging.warning(self.__class__.__name__, time.time(), self.x if self.x is not None else self.y, "\n")

        return self


class AddBlock(Block):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.x = x
        self.y = y

        self.results = None

    def run(self, *args):
        logging.warning(f"on {self} at {time.time()}")
        self.setup(*args)

        time.sleep(2)

        self.results = self.x + self.y

        logging.warning(self.__class__.__name__, time.time(), self.results, "\n")

        return self


class ResultsBlock(Block):
    def __init__(self, x=None):
        super().__init__()
        self.x = x
        self.results = None

    def run(self, *args):
        self.setup(*args)
        self.results = self.x
        return self


class JoinAddBlock(Block):
    def __init__(self, results=None):
        super().__init__()

        self.results = results

    def run(self, *args):
        logging.warning(f"on {self} at {time.time()}")
        self.setup(*args)

        time.sleep(1)

        self.results = sum(self.results)

        logging.warning(self.__class__.__name__, time.time(), self.results, "\n")

        return self


class TorchNormalTensorBlock(Block):
    def __init__(self, mean=0.0, std=1.0, size=(3000, 3000), b=1, device="cpu"):
        super().__init__()
        self.mean = mean
        self.std = std
        self.size = size
        self.b = b
        self.device = device

        self.X = None

    def run(self, *args):
        self.setup(*args)
        self.X = [torch.normal(mean=self.mean, std=self.std, size=self.size).to(self.device) for _ in range(self.b)]
        return self


class TorchMatrixXYBlock(Block):
    def __init__(self, idx, *args, X=None):
        super().__init__()
        self.idx = idx
        self.X = X

        self.results = None

    def run(self, *args):
        priors = self.setup(*args)

        self.results = self.X[self.idx].mm(self.X[self.idx].T)

        return self


class TorchMergeBlock(Block):
    def __init__(self, results=None, **kwargs):
        super().__init__()
        self.results = results

        self.X = None

    def run(self, *args):
        priors = self.setup(*args)

        self.X = torch.stack(self.results)

        self.flush_results = True

        return self
