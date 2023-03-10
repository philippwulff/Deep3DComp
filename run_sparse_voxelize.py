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


def run_sparse_voxelize(input_obj_path: str, voxel_resolution: int, logs: list):

    gt_mesh = utils.scale_to_unit_sphere(trimesh.load(input_obj_path))

    example = pd.read_csv("/home/wulff/deepsdf/data/voxelize_until_cd_meshes_CD=0.001/run_voxelize_until_CD_logs.csv").iloc[3]
    print(example)

    gt_mesh = utils.scale_to_unit_sphere(input_obj_path)
    voxels = mesh_to_sdf.mesh_to_voxels(gt_mesh, voxel_resolution=voxel_resolution, check_result=True, pad=True, sign_method="depth")

    sparse_vox = copy.deepcopy(voxels)
    voxel_size = 2.0 / (example["voxel_resolution"] - 1)
    sparse_vox[abs(sparse_vox)>2*math.sqrt(2*voxel_size**2)] = 1
    verts, faces, normals, values = skimage.measure.marching_cubes(sparse_vox, level=0.0, method="lewiner")
    reconstruction = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    reconstruction = utils.scale_to_unit_sphere(reconstruction)
    cd, _ = metrics.compute_metric(gt_mesh, reconstruction, metric="chamfer")
    num_sparse_voxels = (example["voxel_resolution"]+2)**3 - len(sparse_vox[sparse_vox == 1.0])

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
    logging.info(f"Using {args.num_threads} cores.")

    for vox_log in voxelize_logs:

        logging.info(f"Computing sparse voxel grids for: {vox_log}")

        log_df = pd.read_csv(vox_log)
        log_df["num_sparse_voxels"] = np.nan
        log_df["sparse_cd"] = np.nan

        try:
            for i in tqdm(len(log_df)):
                if log_df.iloc[i]["num_sparse_voxels"] != np.nan:
                    continue
                cd, num_sparse_voxels = run_sparse_voxelize(
                    log_df.iloc[i]["input_obj_path"],
                    log_df.iloc[i]["voxel_resolution"],
                )
                log_df.iloc[i]["num_sparse_voxels"] = num_sparse_voxels
                log_df.iloc[i]["sparse_cd"] = cd
        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt. Cleaning up and exiting.")
        finally:
            log_df.to_csv(vox_log, index=False)
