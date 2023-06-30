import glob
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from ml_hep_sim.plotting.matplotlib_setup import configure_latex

configure_latex()


def get_npy_files(file_path):
    idx, arrs = [], []
    npy_files = glob.glob(f"{file_path}/*.np[yz]")

    for np_name in npy_files:
        head, tail = os.path.split(np_name)
        n = re.search("[0-9]+", tail).group()
        idx.append(int(n))
        arrs.append(np.load(np_name))

    sorted_idx = np.argsort(idx)
    return [arrs[i] for i in sorted_idx]


class FlowAnimate:
    def __init__(self, data, fps=20, mesh_points=200, xmin=-4.0, xmax=4.0, axs_names=None):
        self.data = data
        self.fps = fps
        self.mesh_points = mesh_points
        self.xmin, self.xmax = xmin, xmax

        if any(isinstance(el, list) for el in data):
            self.n = len(data)
        else:
            self.n = 1

        self.fig, self.axs = plt.subplots(1, self.n, figsize=(15, 15 / self.n))

        if self.n == 1:
            self.data = [data]
            self.axs = [self.axs]
        else:
            self.axs = list(self.axs.flatten())

        if axs_names:
            for i in range(self.n):
                self.axs[i].set_title(axs_names[i])

        self.im = []
        for i in range(self.n):
            im = self.axs[i].imshow(
                self.data[i][0].reshape(mesh_points, mesh_points).T, origin="lower", animated=True, vmin=0, vmax=1
            )
            self.im.append(im)

    def init(self):
        for i in range(self.n):
            self.axs[i].set_xticks(
                [
                    1,
                    int(self.mesh_points * 0.25),
                    int(self.mesh_points * 0.5),
                    int(self.mesh_points * 0.75),
                    self.mesh_points - 1,
                ]
            )
            self.axs[i].set_xticklabels([self.xmin, int(self.xmin / 2), 0, int(self.xmax / 2), self.xmax])
            self.axs[i].set_yticks(
                [
                    1,
                    int(self.mesh_points * 0.25),
                    int(self.mesh_points * 0.5),
                    int(self.mesh_points * 0.75),
                    self.mesh_points - 1,
                ]
            )
            self.axs[i].set_yticklabels([self.xmin, int(self.xmin / 2), 0, int(self.xmax / 2), self.xmax])
            self.im[i].set_data(self.data[i][0].reshape(self.mesh_points, self.mesh_points).T)

        return self.im

    def animate(self, i):
        logging.warning(f"drawing frame {i}")

        for n in range(self.n):
            z = self.data[n][i + 1].reshape(self.mesh_points, self.mesh_points).T
            self.im[n].set_array(z / (np.max(z) + 1e-6))

        return self.im


def animate(data, animate_kwargs, dpi=100, name="test.mp4"):
    anim_obj = FlowAnimate(data, **animate_kwargs)

    if any(isinstance(el, list) for el in data):
        l = len(data[0])
    else:
        l = len(data)

    anim = animation.FuncAnimation(
        anim_obj.fig,
        anim_obj.animate,
        init_func=anim_obj.init,
        frames=l - 1,
        interval=1000 / anim_obj.fps,
        blit=True,
    )

    writer = animation.writers["ffmpeg"](fps=anim_obj.fps)
    anim_obj.fig.tight_layout()
    anim.save(name, writer=writer, dpi=dpi)


if __name__ == "__main__":
    file_name = "spline_toys"
    names = ["splines_swissroll", "splines_8gaussians", "splines_checkerboard"]

    data = []
    for name in names:
        data.append(get_npy_files(f"./data/{name}"))

    animate(
        data,
        {
            "fps": 30,
            "mesh_points": 200,
            "xmin": -4.0,
            "xmax": 4.0,
            "axs_names": ["swissroll", "8 Gaussians", "checkerboard"],
        },
        name=f"{file_name}.mp4",
        dpi=200,
    )
