import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from skimage import color
from skimage import io as imageio
from skimage import transform
from sklearn.utils import shuffle as util_shuffle

from ml_hep_sim.data_utils.dummy_dataset import DummyDataModule

TOY_DATASETS = [
    "swissroll",
    "circles",
    "rings",
    "moons",
    "4gaussians",
    "8gaussians",
    "pinwheel",
    "2spirals",
    "checkerboard",
    "line",
    "cos",
    "fmf_normal",
    "fmf_uniform",
    "einstein",
]


class ToyDataModule(DummyDataModule):
    def __init__(self, dataset_name, data_size, batch_size, to_gpu=False, **dl_kwargs):
        super().__init__(data_size=data_size, batch_size=batch_size, to_gpu=to_gpu, **dl_kwargs)
        self.dataset_name = dataset_name
        self.data_size = data_size

    def prepare_data(self):
        self.data = [generate_2d_data(self.dataset_name, batch_size=self.data_size)[0] for _ in range(3)]
        self.labels = [np.zeros(len(d)).astype(np.float32) for d in self.data]

        logging.warning(f"using {self.data[0].shape} data points...")


def generate_2d_data(data, rng=True, batch_size=1000):
    """https://github.com/LukasRinder/normalizing-flows/blob/master/data/toy_data.py"""
    if rng:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.max(data)

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=0.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data, np.max(data)

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]), np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])]
            ).T
            * 3.0
        )
        X = util_shuffle(X, random_state=rng)

        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), np.max(X)

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2], dtype="float32")
        return data, np.max(data)

    elif data == "4gaussians":
        scale = 4.0
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.max(dataset)

    elif data == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.max(dataset)

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return data.astype(np.float32), np.max(data)

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return np.array(x, dtype="float32"), np.max(x)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return np.array(data, dtype="float32"), np.max(data)

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
        return data, np.max(data)

    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        return data, np.max(data)

    elif data == "fmf_normal":
        n = 10
        n_rng = batch_size // 400
        sigma_x, sigma_y = 0.4, 0.4

        f1 = random_line(n * 5, n_rng, mn=0, mx=5, sigma_x=sigma_x, sigma_y=sigma_y, line="vertical")
        f2 = random_line(n * 3, n_rng, mn=0, mx=3, sigma_x=sigma_x, sigma_y=sigma_y, line="horizontal", shift=5)
        f3 = random_line(n * 2, n_rng, mn=0, mx=2, sigma_x=sigma_x, sigma_y=sigma_y, line="horizontal", shift=3)

        f = np.concatenate((f1, f2, f3))

        m1 = random_line(n * 5, n_rng, mn=0, mx=5, sigma_x=sigma_x, sigma_y=sigma_y, line="vertical", shift=4)
        m2 = random_line(n * 5, n_rng, mn=4, mx=8, sigma_x=sigma_x, sigma_y=sigma_y, line="horizontal", shift=5)
        m3 = random_line(n * 5, n_rng, mn=0, mx=5, sigma_x=sigma_x, sigma_y=sigma_y, line="vertical", shift=6)
        m4 = random_line(n * 5, n_rng, mn=0, mx=5, sigma_x=sigma_x, sigma_y=sigma_y, line="vertical", shift=8)

        fm = np.concatenate((f, m1, m2, m3, m4))

        f21 = random_line(n * 5, n_rng, mn=0, mx=5, sigma_x=sigma_x, sigma_y=sigma_y, line="vertical", shift=10)
        f22 = random_line(n * 3, n_rng, mn=10, mx=13, sigma_x=sigma_x, sigma_y=sigma_y, line="horizontal", shift=5)
        f23 = random_line(n * 2, n_rng, mn=10, mx=12, sigma_x=sigma_x, sigma_y=sigma_y, line="horizontal", shift=3)

        fmf = np.concatenate((fm, f21, f22, f23)) * 0.5
        fmf[:, 0] = fmf[:, 0] - 3
        fmf[:, 1] = fmf[:, 1] - 1.3

        return fmf.astype(np.float32), None

    elif data == "fmf_uniform":
        n = batch_size // 34
        # F
        fmf1 = get_block(0, 5, n)

        fmf2 = get_block(2, 0, n)
        fmf2[:, 0] += 1
        fmf2[:, 1] += 4

        fmf3 = get_block(2, 0, n)
        fmf3[:, 0] += 1
        fmf3[:, 1] += 2

        # M
        fmf4 = get_block(0, 4, n)
        fmf4[:, 0] += 4

        fmf5 = get_block(5, 0, n)
        fmf5[:, 0] += 4
        fmf5[:, 1] += 4

        fmf6 = get_block(0, 4, n)
        fmf6[:, 0] += 8

        fmf7 = get_block(0, 4, n)
        fmf7[:, 0] += 6

        # F
        fmf8 = get_block(0, 5, n)
        fmf8[:, 0] += 10

        fmf9 = get_block(2, 0, n)
        fmf9[:, 0] += 1 + 10
        fmf9[:, 1] += 4

        fmf10 = get_block(2, 0, n)
        fmf10[:, 0] += 1 + 10
        fmf10[:, 1] += 2

        fmf_lst = [fmf1, fmf2, fmf3, fmf4, fmf5, fmf6, fmf7, fmf8, fmf9, fmf10]
        fmf_arr = np.concatenate(fmf_lst)

        fmf_arr[:, 0] -= 6.5
        fmf_arr[:, 1] -= 2.5
        fmf_arr[:, 0] *= 0.65
        fmf_arr[:, 1] *= 1.65

        return fmf_arr.astype(np.float32), None

    elif data == "einstein":
        return create_einstein_data(batch_size), None

    else:
        raise ValueError


def density_plot(z, axs=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots()

    axs.hexbin(z[:, 0], z[:, 1], **kwargs)
    return axs


def _plot_all():
    fig, axs = plt.subplots(4, 4, figsize=(10, 12))
    axs = axs.flatten()

    for i, name in enumerate(TOY_DATASETS):
        data_module = ToyDataModule(name, data_size=20000, batch_size=100)
        data_module.prepare_data()
        data_module.setup()

        z = data_module.train.X

        axs[i].set_title(name)
        density_plot(z, axs[i])

    plt.tight_layout()
    plt.show()


def random_line(n, n_rng, mn=-4, mx=4, sigma_x=1, sigma_y=1, line="vertical", shift=0):
    n = int(n)
    if line == "vertical":
        xs, ys = np.ones(n) * shift, np.linspace(mn, mx, n)
    elif line == "horizontal":
        xs, ys = np.linspace(mn, mx, n), np.ones(n) * shift
    else:
        raise ValueError

    pts_x, pts_y = np.zeros((n, n_rng)), np.zeros((n, n_rng))

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_rng = np.random.normal(x, sigma_x, size=n_rng)
        y_rng = np.random.normal(y, sigma_y, size=n_rng)

        pts_x[i, :] = x_rng
        pts_y[i, :] = y_rng

    pts_x, pts_y = pts_x.flatten(), pts_y.flatten()
    return np.concatenate((pts_x[:, None], pts_y[:, None]), axis=1)


def get_block(nx, ny, n):
    blocks = []
    block_f = lambda n: np.random.uniform(0, 1, size=(n, 2))

    for x in range(nx):
        block = block_f(n)
        block[:, 0] += x
        blocks.append(block)

    for y in range(ny):
        block = block_f(n)
        block[:, 1] += y
        blocks.append(block)

    return np.concatenate(blocks)


def create_einstein_data(n, root="./data", face="einstein"):
    """https://github.com/conormdurkan/autoregressive-energy-machines/blob/master/pytorch/utils/plane.py"""
    path = os.path.join(root, face + ".jpg")
    image = imageio.imread(path)
    image = color.rgb2gray(image)
    image = transform.resize(image, (512, 512))

    grid = np.array([(x, y) for x in range(image.shape[0]) for y in range(image.shape[1])])

    rotation_matrix = np.array([[0, -1], [1, 0]])
    p = image.reshape(-1) / sum(image.reshape(-1))
    ix = np.random.choice(range(len(grid)), size=n, replace=True, p=p)
    points = grid[ix].astype(np.float32)
    points += np.random.rand(n, 2)  # dequantize
    points /= image.shape[0]  # scale to [0, 1]

    data = (points @ rotation_matrix).astype(np.float32)
    data[:, 1] += 1
    return data


if __name__ == "__main__":
    _plot_all()
    data = create_einstein_data(10000)
    density_plot(data, gridsize=100)
    plt.show()
