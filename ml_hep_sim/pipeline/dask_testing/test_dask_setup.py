import unittest

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster

from ml_hep_sim.pipeline.dask_testing.simple_blocks import (
    AddBlock,
    DoubleBlock,
    IncBlock,
    JoinAddBlock,
    ResultsBlock,
    SquareBlock,
    TorchMatrixXYBlock,
    TorchMergeBlock,
    TorchNormalTensorBlock,
)
from ml_hep_sim.pipeline.pipes import Pipeline


class TestDaskSimpleBlockPipeline(unittest.TestCase):
    def test_dask_basic_pipeline(self):
        """
        x1----x3----
                    |----x5
        x2----x4----
        """
        client = Client(dashboard_address=":8787")

        p = Pipeline()

        x1 = IncBlock(1)

        x2 = DoubleBlock(2)

        x3 = SquareBlock()
        x3.take(x1)

        x4 = SquareBlock()
        x4.take(x2)

        x5 = AddBlock()
        x5.take(x3, x4)

        p.compose(x1, x2, x3, x4, x5)
        p.fit()

        res = p.computed

        client.close()

        self.assertEqual(res.results, 16)

    def test_dask_basic_multi_pipeline(self):
        """
        x1----
              |
        x2----|----x5
              |     |----x7
        x3----|----x6
              |
        x4----
        """
        client = Client(dashboard_address=":8787")

        p = Pipeline()

        x1 = IncBlock(1)
        x2 = IncBlock(2)

        x3 = DoubleBlock(1)
        x4 = DoubleBlock(2)

        x5 = AddBlock().take(x1, x3)
        x6 = AddBlock().take(x2, x4)

        x7 = JoinAddBlock().take(x5, x6)

        p.compose(x1, x2, x3, x4, x5, x6, x7)
        p.draw_pipeline_tree(to_graphviz_file="test")

        p.fit()

        client.close()

    def test_dask_basic_multi_results_pipeline(self):
        """
        x1----
        .     |
        .     |----y
        .     |
        xn----
        """
        client = Client(dashboard_address=":8787")

        p = Pipeline()

        xs1 = [IncBlock(i) for i in range(10)]
        xs2 = [ResultsBlock().take(x) for x in xs1]
        y = JoinAddBlock().take(*xs2)

        p.compose(*xs1, *xs2, y)
        p.fit()

        client.close()


class TestDaskTorch(unittest.TestCase):
    def test_dask_torch_cpu(self):
        cluster = LocalCluster(memory_limit="auto", n_workers=1, threads_per_worker=5, dashboard_address=":8787")
        client = Client(cluster)

        b = 5
        p = Pipeline()

        x1 = TorchNormalTensorBlock(b=b, device="cpu")
        x2 = [TorchMatrixXYBlock(i, x1).take(x1) for i in range(b)]
        x3 = TorchMergeBlock().take(*x2)

        p.compose(x1, x2, x3)
        p.fit()

        client.close()

    def test_dask_torch_cuda(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        p = Pipeline()

        x1 = TorchNormalTensorBlock(b=5, device="cuda")
        x2 = [TorchMatrixXYBlock(i, x1).take(x1) for i in range(5)]
        x3 = TorchMergeBlock().take(*x2)

        p.compose(x1, x2, x3)
        p.fit()

        client.close()


if __name__ == "__main__":
    unittest.main()
