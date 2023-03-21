import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from typing import Union, List, Dict
import os
import deep_sdf.workspace as ws
from deep_sdf import utils, metrics
import sklearn
from sklearn.manifold import TSNE
import trimesh
import math
import pandas as pd
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

import json
if not os.name == "nt":
    # We do not import this on Windows.
    import pyrender
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


def render_sdf(points: np.array, sdf: np.array, cam_angles=(-np.pi/7, np.pi/4, 0)):
    """
    Default angles are from top-right.
    Example use:
        c, d = render_sdf(points, sdf)
        plt.imshow(c)
    """
    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1      # inside -> Blue
    colors[sdf > 0, 0] = 1      # outside -> Red
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

    scene.add(cloud)
    # cam looks in neg z dir: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2,3] = 2.0      # in z dir
    camera_pose = utils.rotate(camera_pose, *cam_angles)
    scene.add(camera, pose=camera_pose)

    light = pyrender.SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/4.0)
    scene.add(light, pose=camera_pose)

    # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    r = pyrender.OffscreenRenderer(viewport_width=480, viewport_height=480, point_size=1.0)
    color, depth = r.render(scene)
    r.delete()

    return color, depth


def render_mesh(mesh: trimesh.Trimesh, cam_angles=(-np.pi/7, np.pi/4, 0)):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # cam looks in neg z dir: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2,3] = 2      # in z dir
    camera_pose = utils.rotate(camera_pose, *cam_angles)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/4.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=480, viewport_height=480, point_size=1.0)
    color, depth = r.render(scene)
    r.delete()
    return color, depth


def render_sdf_vid(points: np.array, sdf: np.array, fps=2, n_seconds=5, save_filepath=""):
    """Renders a SDF from different angles and makes a video."""
    fig = plt.figure( figsize=(8,8) )
    color, depth = render_sdf(points, sdf)
    im = plt.imshow(color, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        # quarter of a full rotation
        rot = (np.pi/2 * i)/(fps * n_seconds)
        color, depth = render_sdf(points, sdf, cam_angles=(-np.pi/7, np.pi/4+rot, 0))
        im.set_array(color)
        return [im]

    # interval in ms
    anim = animation.FuncAnimation(fig, animate_func, frames=n_seconds * fps, interval=1000/fps,)
    if save_filepath:
        anim.save(save_filepath + ".mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
    return anim


def plot_sdf_cross_section(points: np.array, sdf: np.array, margin=0.05, plane_orig=np.array([0,0,0]), plane_normal=np.array([1,0,0]), save_filepath="", ax=None):
    """
    Projecting a point onto a plane: 
    https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    dists_from_plane = (points - plane_orig).dot(plane_normal)
    in_margin = np.abs(dists_from_plane) < margin

    proj_points_in_margin = points[in_margin] - dists_from_plane[in_margin][:,None] * plane_normal

    # Y-axis chosen to always point up
    y_axis = np.array([0,0,1]) - np.array([0,0,1]).dot(plane_normal) * plane_normal
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(plane_normal, y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    p_x = proj_points_in_margin.dot(x_axis)
    p_y = proj_points_in_margin.dot(y_axis)

    colors = ["blue" if _ < 0. else "red" for _ in sdf[in_margin]]

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='SDF > 0', markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='SDF < 0', markerfacecolor='blue', markersize=5),
    ]
    if not ax:
        fig, ax = plt.subplots()
    ax.scatter(p_x, p_y, c=colors, s=0.5)
    ax.legend(handles=legend_elements)
    return ax


def plot_capacity_vs_chamfer_dist(
        exp_dirs_net_capacity: List = None, 
        exp_dirs_lat_capacity: List = None,
        voxelization_logs: List[pd.DataFrame] = None,
        checkpoint: int = 2000, 
    ) -> plt.figure:
    """
    Example usage: 
    ```
    exps = [
        "../../shared/deepsdfcomp/searches/double_nonlinearity/siren_width=64_no_bottleneck",
        "../../shared/deepsdfcomp/searches/double_nonlinearity/siren_width=256_no_bottleneck_v2",
    ]
    vox_logs = [pd.read_csv("data/voxelize_until_cd_meshes_CD=0.001/run_voxelize_until_CD_logs.csv")]
    plotting.plot_capacity_vs_chamfer_dist(exps, type=["latent"], voxelization_logs=vox_logs)
    ```
    """
    exps = {
        "net": [] if not exp_dirs_net_capacity else exp_dirs_net_capacity,
        "lat": [] if not exp_dirs_lat_capacity else exp_dirs_lat_capacity,
        "vox": [] if not voxelization_logs else [pd.read_csv(_) for _ in voxelization_logs],
    }
    assert any([exps["net"], exps["lat"], exps["vox"]]), "NO EXPERIMENT DIRS GIVEN"

    # Combine all results from different experiments.
    results = defaultdict(lambda: defaultdict(list))
    for name, exp_dirs in exps.items():
        for exp_dir in exp_dirs:
            if name == "vox":
                results[name]["voxel_resolutions"].append(exp_dir["voxel_resolution"].mean())
                # +2 because we did not add the padding in the logged results
                results[name]["num_voxels"].append((exp_dir["voxel_resolution"].mean()+2)**3)  
                results[name]["cd_means"].append(exp_dir["cd"].mean())
                print(f"Extracting vox_res={exp_dir['voxel_resolution'].mean():.1f}: CD={exp_dir['cd'].mean():.5f} num_shapes={len(exp_dir['voxel_resolution'])}")
                try:
                    results[name]["num_sparse_voxels"].append(exp_dir["num_sparse_voxels"].mean())
                    results[name]["sparse_cd_means"].append(exp_dir["sparse_cd"].mean())
                except KeyError:
                    pass
            else:
                # Read experiment specs.
                specs = ws.load_experiment_specifications(exp_dir)
                dims = specs["NetworkSpecs"]["dims"]
                arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
                latent_size = specs["CodeLength"]
                decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
                results[name]["latent_sizes"].append(latent_size)
                results[name]["widths"].append(dims[0])
                results[name]["depths"].append(len(dims))
                # Calculate model size.
                param_size = 0
                param_cnt = 0
                for param in decoder.parameters():
                    param_size += param.nelement() * param.element_size()
                    param_cnt += param.nelement()
                buffer_size = 0
                for buffer in decoder.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                model_size_mb = (param_size + buffer_size) / 1024**2
                results[name]["param_cnts"].append(param_cnt)
                # Read evaluation results.
                eval_log_path = os.path.join(ws.get_evaluation_dir(exp_dir, str(checkpoint)), "chamfer.csv")
                ws.get_model_params_dir(exp_dir)
                eval_df = pd.read_csv(eval_log_path, delimiter=";")
                results[name]["cd_means"].append(eval_df["chamfer_dist"].mean())
                results[name]["cd_medians"].append(eval_df["chamfer_dist"].median())
                print(f"Extracting num_params={param_cnt} width={dims[0]} depth={len(dims)}: CD_mean={eval_df['chamfer_dist'].mean():.6f} CD_median={eval_df['chamfer_dist'].median():.6f} num_shapes={len(eval_df['chamfer_dist'])}")
                
                eval_log_train_path = os.path.join(ws.get_evaluation_dir(exp_dir, str(checkpoint)), "chamfer_on_train_set.csv")
                if os.path.exists(eval_log_train_path):
                    eval_train_df = pd.read_csv(eval_log_train_path, delimiter=";")
                    results[name]["param_cnts_train"].append(param_cnt)
                    results[name]["cd_means_train"].append(eval_train_df["chamfer_dist"].mean())
                    results[name]["cd_medians_train"].append(eval_train_df["chamfer_dist"].median())
                    print(f"Eval on train set: CD_mean={eval_train_df['chamfer_dist'].mean():.6f} CD_median={eval_train_df['chamfer_dist'].median():.6f} num_shapes={len(eval_train_df['chamfer_dist'])}")

    # Plot.
    fig, axes = plt.subplots(1, len([_ for _ in exps if exps[_]]))
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        if name in ["net", "lat"]:
            ax.set_title(f"# of {name} params vs. Chamfer distance")
            ax.set_ylabel("Chamfer distance")
            ax.set_xlabel(f"# of {name} params")
            x_values = result["param_cnts"] if name == "net" else result["latent_sizes"]
            idxs = np.array(x_values).argsort()
            x = np.array(x_values)[idxs]
            y1 = np.array(result["cd_means"])[idxs]
            y2 = np.array(result["cd_medians"])[idxs]
            ax.plot(x, y1, ls="-", label="SIREN mean CD")
            ax.plot(x, y2, ls="--", label="SIREN median CD")
            if "cd_means_train" in result:
                x_values = result["param_cnts_train"]# if name == "net" else result["latent_sizes"]
                idxs = np.array(x_values).argsort()
                x = np.array(x_values)[idxs]
                y1 = np.array(result["cd_means_train"])[idxs]
                y2 = np.array(result["cd_medians_train"])[idxs]
                ax.plot(x, y1, ls="-", label="SIREN mean CD (train)")
                ax.plot(x, y2, ls="--", label="SIREN median CD (train)")
            
        elif name == "vox":
            ax.set_title(f"Voxel count vs. Reconstruction Chamfer distance")
            ax.set_ylabel("Chamfer distance")
            ax.set_xlabel(f"Voxels count")
            num_voxels = np.array(result["num_voxels"])
            cd_means = np.array(result["cd_means"])
            idxs = num_voxels.argsort()
            x, y = num_voxels[idxs], cd_means[idxs]
            ax.scatter(x, y, marker="x", label="Dense Voxel Grid", color="red")
            # numpy.polyfit(numpy.log(x), y, 1)
            # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), ls="-.", c="red")

            if "num_sparse_voxels" in result:
                num_voxels = np.array(result["num_sparse_voxels"])
                cd_means = np.array(result["sparse_cd_means"])
                idxs = num_voxels.argsort()
                x, y = num_voxels[idxs], cd_means[idxs]
                ax.scatter(x, y, marker="x", label="Sparse Voxel Grid", color="orange")
                ax.set_xscale('log')


        ax.legend()

    plt.close()
    return fig


def plot_manifold_tsne(exp, checkpoint=2000):
    wordnet_relations_df = pd.read_csv("data/shapenet_wordnet_relations.csv")
    cmap = matplotlib.cm.get_cmap("tab20")
    num_colors = 20
    nrows = 7
    ncols = 4
    default_color = "black"

    def wrap_text(text, max_text_len = 25):
        return "\n".join([text[y-max_text_len:y] for y in range(max_text_len, len(text)+max_text_len,max_text_len)])

    # Create t-SNE.
    lat_vecs = ws.load_latent_vectors(exp, str(checkpoint))
    lat_vecs_embedded = TSNE(n_components=2, perplexity=1700).fit_transform(lat_vecs)

    # Load the shape information files.
    with open(os.path.join(exp, ws.specifications_filename), "r") as f:
        specs = json.load(f)
        with open(specs["TrainSplit"], "r") as split_f:
            split = json.load(split_f)
    dataset_name = list(split.keys())[0]
    synset_id, shape_ids = list(split[dataset_name].items())[0]
    
    # The word class belonging to each latent vector.
    wnlemmas = wordnet_relations_df.set_index("fullId").loc[["3dw."+_ for _ in shape_ids], "wnlemmas"]
    # Count the occurences of each class.
    wnlemmas_cnts = wnlemmas.value_counts()
    top_n_wnlemmas = wnlemmas_cnts[:num_colors].index
    cdict = dict(zip(top_n_wnlemmas, cmap(np.linspace(0, 1, num_colors))))
    colors = [cdict.get(_, default_color) for _ in wnlemmas]

    # Plot.
    gs = GridSpec(nrows, ncols, hspace=0.5)
    fig = plt.figure(figsize=(ncols*3, nrows*3.5))

    # Plot points from all classes.
    ax_main = fig.add_subplot(gs[:2, :2])
    ax_main.scatter(lat_vecs_embedded[:, 0], lat_vecs_embedded[:, 1], s=5, c=colors)
    ax_main.set_xlim(-1, 1)
    ax_main.set_ylim(-1, 1)
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            # if r < 1 or (r < 2 and c < 2):
            if r < 2 or i >= len(top_n_wnlemmas):
                continue
            ax = fig.add_subplot(gs[r, c])
            wnlemma = top_n_wnlemmas[i]
            wnlemma_lat_vecs_indxs = [_ for _ in range(len(lat_vecs)) if wnlemmas.iloc[_] == wnlemma]
            wnlemma_lat_vecs = lat_vecs_embedded[wnlemma_lat_vecs_indxs, :]
            color = cdict[wnlemma]
            ax.scatter(wnlemma_lat_vecs[:, 0], wnlemma_lat_vecs[:, 1], s=5, color=color)
            ax.set_title(wrap_text(wnlemma))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            i += 1

    # Legend and style the plot.
    name_cnts = wnlemmas_cnts[:num_colors]
    other_cnts = len(wnlemmas) - sum(name_cnts)
    legend_elements = [
        *[Line2D([0], [0], marker='o', color='w', label=wrap_text(f"{name} ({name_cnts[i]})", 70), markerfacecolor=color, markersize=10) for i, (name, color) in enumerate(cdict.items())],
        Line2D([0], [0], marker='o', color='w', label=f"Others ({other_cnts})", markerfacecolor=default_color, markersize=10)
    ]
    ax_main.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, prop={'size': 9})
    ax_main.set_title("t-SNE plot of latent space")

    # fig.tight_layout()
    plt.close()
    return fig