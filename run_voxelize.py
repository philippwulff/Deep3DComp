# Voxelizes all shapes in some ShapeNet split file with a list of given voxel grid resolutions.
# Saves the Chamfer distance of the reconstruction from Marching Cubes when using a dense and
# a sparse voxel grid.

import os
import json
import logging 
import argparse
import psutil
import time
from deep_sdf import metrics, utils
import trimesh
import pandas as pd
from tqdm import tqdm
import mesh_to_sdf
import skimage
import random
import copy
import math


def get_meshes_targets_and_specific_args(split_path: str):
    # Prepare all input and output mesh files.
    with open(split_path, "r") as f:
        split = json.load(f)
        dataset_name = list(split.keys())[0]
        synset_id = list(split[dataset_name].keys())[0]
        shape_ids = split[dataset_name][synset_id]

    meshes_targets_and_specific_args = []
    file_not_found_cnt = 0
    for shape_id in shape_ids:
        input_obj_paths = [
            # Path that works with ShapeNetCore.v2
            os.path.join(input_dir, synset_id, shape_id, "models/model_normalized.obj"), 
            # Path that works with the DeepSDF dataset structure.
            os.path.join(input_dir, synset_id, shape_id + ".obj")
        ]
        existing_paths = [p for p in input_obj_paths if os.path.exists(p)]  # Should contain only one value.
        if not existing_paths:
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": existing_paths[0],
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".ply"),
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Voxelizing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")
    return meshes_targets_and_specific_args


def run_voxelize(input_obj_path: str, output_obj_path: str, voxel_resolution: int, logs: list):

    gt_mesh = utils.scale_to_unit_sphere(trimesh.load(input_obj_path))
    logging.debug(f"Voxelizing mesh: {input_obj_path}")

    voxel_size = 2.0 / (voxel_resolution - 1)

    # Extract voxel grid.
    try:
        voxels = mesh_to_sdf.mesh_to_voxels(gt_mesh, voxel_resolution=voxel_resolution, check_result=True, pad=True, sign_method="depth")
    except mesh_to_sdf.BadMeshException:
        logging.debug(f"Caught BadMeshException at voxel-res {voxel_resolution} ({input_obj_path})")
        return
    # Reconstruct mesh from voxel grid.
    verts, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.0, spacing=[voxel_size] * 3, method="lewiner")
    reconstruction = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    reconstruction = utils.scale_to_unit_sphere(reconstruction)
    # Compute reconstruction quality.
    cd, _ = metrics.compute_metric(gt_mesh, reconstruction, metric="chamfer")

    vert_cnt = len(reconstruction.vertices)
    num_dense_voxels = (voxel_resolution+2)**3

    # Compute sparse voxel grid.
    sparse_vox = copy.deepcopy(voxels)
    # Drop all voxels further than two voxel diagonals.
    sparse_vox[abs(sparse_vox)>2*math.sqrt(2*voxel_size**2)] = 1
    verts, faces, normals, _ = skimage.measure.marching_cubes(sparse_vox, level=0.0, spacing=[voxel_size] * 3, method="lewiner")
    sparse_reconstruction = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    sparse_reconstruction = utils.scale_to_unit_sphere(sparse_reconstruction)
    sparse_cd, _ = metrics.compute_metric(gt_mesh, sparse_reconstruction, metric="chamfer")
    num_sparse_voxels = num_dense_voxels - len(sparse_vox[sparse_vox == 1.0])
    # Save results to logs.
    logs.append([
        input_obj_path, 
        output_obj_path,
        voxel_resolution, 
        len(gt_mesh.vertices),
        vert_cnt,
        cd,
        sparse_cd,
        num_sparse_voxels,
        num_dense_voxels
    ])

    logging.debug(f"Reduced with voxel-res {voxel_resolution} to chamfer distance of {cd:4f}. Reduced Mesh has {vert_cnt} Vertices.")
    with open(output_obj_path, "wb+") as f:
        f.write(trimesh.exchange.ply.export_ply(reconstruction))
        
if __name__ == "__main__":

    output_dir = "data/voxRes/voxelize_meshes"      # This needs to be changed to where you want your data to be extracted to!
    input_dir = "/mnt/hdd/ShapeNetCore.v2"
    input_dir = "../../shared/deepsdfcomp/data/manifold_meshes"
    split_path = "examples/splits/sv2_planes_test.json"

    voxel_resolutions = [
        128
    ]

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)

    for voxel_resolution in voxel_resolutions:  
        output_dir += f"_res={args.voxel_resolution}"
        os.makedirs(output_dir, exist_ok=True)  
        logging.info(f"Voxelizing with resolution={voxel_resolution} into {output_dir}")
        meshes_targets_and_specific_args = get_meshes_targets_and_specific_args(split_path)
        shared_logs = []
        try:
            for mtsa in tqdm(meshes_targets_and_specific_args):
                if os.path.exists(mtsa["output_obj_path"]):
                    continue
                run_voxelize(
                    mtsa["input_obj_path"],
                    mtsa["output_obj_path"],
                    voxel_resolution,
                    shared_logs,
                )
        except KeyboardInterrupt:
            logging.info("Cleaning up and exiting.")
        finally:
            df_output_path = os.path.join(output_dir, "run_voxelize_until_CD_logs.csv")
            logs_df = pd.DataFrame(
                shared_logs, 
                columns=[
                    "input_obj_path", 
                    "output_obj_path", 
                    "voxel_resolution", 
                    "gt_vertices", 
                    "decimated_vertices", 
                    "cd", 
                    "sparse_cd", 
                    "num_sparse_voxels", 
                    "num_dense_voxels"
                ],
            )
            if os.path.exists(df_output_path):
                logs_df_old = pd.read_csv(df_output_path)
                logs_df_all = pd.concat([logs_df_old, logs_df], ignore_index=True, axis=0)
                logs_df_all.to_csv(df_output_path)
            else:
                logs_df.to_csv(df_output_path, index=False)
