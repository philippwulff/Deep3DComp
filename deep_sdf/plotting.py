import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from typing import Union, List
import os
import deep_sdf.workspace as ws
from deep_sdf import utils, metrics
import trimesh
import pyrender
import math
import pandas as pd


os.environ['PYOPENGL_PLATFORM'] = 'egl'


def plot_train_stats(loss_hists: list, psnr_hist=None, step_hist=None, labels=None, save_path="") -> plt.figure:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    if not step_hist:
        step_hist = list(range(len(loss_hists[0])))

    fig.suptitle(f"Training curves {save_path}")
    for i, loss_hist in enumerate(loss_hists):
        label = f"Loss: {labels[i]}" if labels else "Loss"
        ax.plot(step_hist, loss_hist, c="orange", label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    if psnr_hist:
        ax2 = ax[0].twinx()
        ax2.plot(step_hist, psnr_hist, c="g", label="PSNR")
        ax2.set_ylabel("PSNR")
    fig.legend()

    if save_path:
        fig.savefig(f"{save_path}.jpg", dpi=300, bbox_inches='tight')

    return fig


def plot_dist_violin(data: np.ndarray, percentile_keys: list=[50, 75, 90, 99]) -> Union[plt.figure, dict]:
    start = time.time()
    colors = ["lightblue", "green", "orange", "purple", "lime"]
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel("Distance to NN")
    ax.set_title("Nearest Neighbor Distances (Two-way)")
    ax.set_xticks([])

    vplot = ax.violinplot(data, showmeans=False, showextrema=True)
    vplot["cmaxes"].set_edgecolor("darkblue")
    vplot["cmins"].set_edgecolor("darkblue")
    vplot["cbars"].set_edgecolor("darkblue")

    percentiles = np.percentile(data, percentile_keys)
    percentiles = {k: p for k, p in zip(percentile_keys, percentiles)}

    for k, p in reversed(percentiles.items()):
        k = f"{k}th percentile" if k!=50 else "Median"
        ax.hlines([p], xmin=[0.9], xmax=[1.1], linestyles="--", colors=[colors.pop()], label=k)
        ax.annotate(f"{p:.4f}", xy=[1.1, p], va="center")

    ax.scatter(1, np.mean(data), marker="o", color="red", s=100, zorder=999, label="Mean (CD)")
    ax.legend(loc="upper left")

    for vp in vplot["bodies"]:
        vp.set_facecolor("cornflowerblue")
        vp.set_zorder(2)
        vp.set_alpha(1)
        vp.set_linewidth(1)

    # Reduce the length of horizontal lines 
    # (from: https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py)
    factor_x, factor_y = 0.3, 1 # factor to reduce the lengths
    for vp_part in ("cbars", "cmaxes", "cmins"):
        vp = vplot[vp_part]
        if vp_part in ("cmaxes", "cmins"):
            lines = vp.get_segments()
            new_lines = []
            for line in lines:
                center = line.mean(axis=0)
                line = (line - center) * np.array([factor_x, factor_y]) + center
                new_lines.append(line)
            vp.set_segments(new_lines)
        vp.set_edgecolor("black")
    logging.debug(f"Plotting all chamfer distances took {time.time()-start}sec")

    return fig, percentiles


def pyrender_helper(mesh: trimesh.Trimesh, alpha=0, beta=0, gamma=0):
    """Renders a Trimesh and returns the color and depth image numpy arrays."""
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2      # move in z-dir
    camera_pose = utils.rotate(camera_pose, alpha=alpha, beta=beta, gamma=gamma)
    scene.add(camera, pose=camera_pose)
    # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
    #                             innerConeAngle=np.pi/16.0,
    #                             outerConeAngle=np.pi/6.0)
    # light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=500.)
    light = pyrender.PointLight(color=[1, 1, 1], intensity=1000.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(600, 600)
    color, depth = r.render(scene)
    return color, depth


def plot_reconstruction_comparison(experiment_dirs: List[str], shape_ids: List[str], chckpt: int = 2000, synset_id: str = "02691156", dataset_name: str = "ShapeNetV2", shapenet_dir: str = "/mnt/hdd/ShapeNetCore.v2"):
    """
    Plot reconstructions with CD for the same shape reconstructions from different experiments 
    plus the GT mesh from ShapeNet.
    """
    # Plot each shape from different angles.
    angles = [
        (-math.pi/2, 0, 0),             # birds-eye-view
        (-math.pi/4, 3*math.pi/4, 0),   # view from upper-left
        (0, 3*math.pi/4, 0),    # view from center-left
    ]

    nrows = len(angles)*len(shape_ids)
    ncols = len(experiment_dirs)+1
    fig, ax = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), dpi=200)
    # parameter meanings here: https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.05)

    for r in range(0, len(angles)*len(shape_ids), len(angles)):
        shape_id = shape_ids[r//len(angles)]

        if r == 0:
            ax[r, 0].set_title("GT")

        if r % len(angles) == 0:
            # Add mesh name 
            left = ax[r, 0].get_position().xmin
            bottom = ax[r+len(angles)-1, 0].get_position().ymin
            top = ax[r, 0].get_position().ymax
            # NOTE: This does not work with fig.tight_layout()
            fig.text(left-0.01, (top+bottom)/2, shape_id, va="center", ha="right", rotation="vertical")
        
        # Plot GT.
        for i, (alpha, beta, gamma) in enumerate(angles):
            gt_mesh_path = os.path.join(shapenet_dir, synset_id, shape_id, "models", "model_normalized.obj")
            gt_mesh = utils.scale_to_unit_sphere(trimesh.load(gt_mesh_path))
            color, _ = pyrender_helper(gt_mesh, alpha, beta, gamma)
            ax[r+i, 0].imshow(color)
            ax[r+i, 0].set_xticks([])
            ax[r+i, 0].set_yticks([])
        # Plot other experiments.
        for c, exp_dir in enumerate(experiment_dirs):
            c += 1      # first column is GT.
            if r == 0:
                title = exp_dir.split(os.sep)[-1]
                max_title_len = 25
                title = "\n".join([title[y-max_title_len:y] for y in range(max_title_len, len(title)+max_title_len,max_title_len)])
                ax[r, c].set_title(title)

            mesh_path = os.path.join(exp_dir, ws.reconstructions_subdir, str(chckpt), ws.reconstruction_meshes_subdir, dataset_name, synset_id, shape_id + ".ply")
            try:
                mesh = utils.scale_to_unit_sphere(trimesh.load(mesh_path))
            except ValueError as e:
                logging.error(f"File does not exist as path {mesh_path}")
            cd, cd_all = metrics.compute_metric(gt_mesh, mesh, metric="chamfer")
            ax[r, c].annotate(f"CD={cd:.6f}", (3, color.shape[0]), va="bottom", ha="left")
            for i, (alpha, beta, gamma) in enumerate(angles):
                color, _ = pyrender_helper(mesh, alpha, beta, gamma)
                ax[r+i, c].imshow(color)
                ax[r+i, c].set_xticks([])
                ax[r+i, c].set_yticks([])

    plt.close()
    return fig


def plot_binary_vs_continuous(df: pd.DataFrame, binary: str, continuous: str):
    """Helper function for plotting the loss against categorical variables of a DataFrame."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df.plot(x=binary, y=continuous, kind="bar", ax=ax[0])
    df.groupby(binary).apply(lambda g: g.mean()).plot(y=continuous, kind="bar", ax=ax[1])
