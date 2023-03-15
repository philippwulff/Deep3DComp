# Reads a list of voxelization logs, and den computes the sparse voxel grids from the dense
# voxel grids found from those optimizations. 
# Running this script requires running `run_voxelize_until_CD.py` first.

import logging 
import argparse
from deep_sdf import metrics, utils
import trimesh
import pandas as pd
from tqdm import tqdm
import mesh_to_sdf
import skimage
import copy
import math
import numpy as np


def run_sparse_voxelize(input_obj_path: str, voxel_resolution: int):

    gt_mesh = utils.scale_to_unit_sphere(trimesh.load(input_obj_path))
    voxels = mesh_to_sdf.mesh_to_voxels(gt_mesh, voxel_resolution=voxel_resolution, check_result=True, pad=True, sign_method="depth")
    voxel_size = 2.0 / (voxel_resolution - 1)
    voxels[abs(voxels)>2*math.sqrt(2*voxel_size**2)] = 1
    verts, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.0, method="lewiner")
    reconstruction = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    reconstruction = utils.scale_to_unit_sphere(reconstruction)
    cd, _ = metrics.compute_metric(gt_mesh, reconstruction, metric="chamfer")
    num_sparse_voxels = (voxel_resolution+2)**3 - len(voxels[voxels == 1.0])

    return cd, num_sparse_voxels


if __name__ == "__main__":

    voxelize_logs = [
        "data/voxelize_until_cd_meshes_CD=0.0001/run_voxelize_until_CD_logs.csv",
        "data/voxelize_until_cd_meshes_CD=0.0005/run_voxelize_until_CD_logs.csv",
        "data/voxelize_until_cd_meshes_CD=0.001/run_voxelize_until_CD_logs.csv",
        "data/voxelize_until_cd_meshes_CD=0.002/run_voxelize_until_CD_logs.csv",
        "data/voxelize_until_cd_meshes_CD=0.003/run_voxelize_until_CD_logs.csv",
        "data/voxelize_until_cd_meshes_CD=0.005/run_voxelize_until_CD_logs.csv",
    ]

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)

    for vox_log in voxelize_logs:

        logging.info(f"Computing sparse voxel grids for: {vox_log}")

        log_df = pd.read_csv(vox_log)
        log_df["num_sparse_voxels"] = np.nan
        log_df["sparse_cd"] = np.nan

        try:
            for i in tqdm(range(len(log_df))):
                if not np.isnan(log_df.iloc[i]["num_sparse_voxels"]):
                    continue
                cd, num_sparse_voxels = run_sparse_voxelize(
                    log_df.loc[i, "input_obj_path"],
                    log_df.loc[i, "voxel_resolution"],
                )
                log_df.loc[i, "num_sparse_voxels"] = num_sparse_voxels
                log_df.loc[i, "sparse_cd"] = cd
        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt. Cleaning up and exiting.")
        finally:
            log_df.to_csv(vox_log, index=False)
