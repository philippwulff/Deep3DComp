#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from typing import Optional
import trimesh
import os
import subprocess
import math
from trimesh import creation, transformations
import tempfile

from deep_sdf import utils


def create_mesh(decoder, latent_vec, filename=None, N=256, max_batch=32 ** 3, offset=None, scale=None, return_trimesh=False) -> Optional[trimesh.Trimesh]:
    """Creates a mesh given the trained decoder and latent code by
    1. Sampling xyz query points
    2. Retrieving the SDF predictions
    3. Running marching cubes to get mesh vertices and faces.
    
    With settings N=256 and max_batch=int(2 ** 18) this takes about 10sec on GPU and 100sec on the CPU.
    """
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    logging.debug("[create_mesh] sampling takes: %f" % (end - start))

    tmpdirname = None
    if not ply_filename: 
        tmpdirname = tempfile.TemporaryDirectory()
        ply_filename = os.path.join(tmpdirname.name, "create_mesh_ply")

    success = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )
    if return_trimesh and success:
        mesh = utils.as_mesh(trimesh.load(ply_filename + ".ply"))
        if tmpdirname: 
            tmpdirname.cleanup()
        return mesh


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
) -> bool:
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3, method="lewiner"
        )
    except ValueError as e:
        logging.error(f"[create_mesh] Caught marching cubes error: {e}.")
        return False
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("[create_mesh] saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "[create_mesh] converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
    return True


def get_SDFGen_voxels(
    shape_id: str,
    voxel_resolution: int,
    padding: float = 3,
    shapenet_path: str = "/mnt/hdd/ShapeNetCore.v2",
    class_id: str = "02691156",
    sdf_gen_path: str = "/home/freissmuth/sdf-gen/build/bin"
):
    voxel_size = 2.0 / (voxel_resolution - 2 * padding)
    in_path = os.path.join(shapenet_path, class_id, shape_id, "models/model_normalized.obj")
    out_path = os.path.join(shapenet_path, class_id, shape_id, "models/normalized")
    unit_path = out_path + "_unit.obj"
    
    in_mesh = utils.as_mesh(trimesh.load(in_path))    
    in_mesh_unit, centroid, scale = utils.scale_to_unit_cube(in_mesh, return_stats=True)
    in_mesh_unit.export(unit_path, file_type='obj')
    
    cmd = f'{sdf_gen_path}/sdf_gen_shapenet {unit_path} {out_path} {str(voxel_size)} ' + str(padding)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)#stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
    
    voxels = np.load(str(out_path)+".npy")
    # Needed because SDFGen rotates the mesh.
    voxels = np.swapaxes(voxels, 0, 2)
    
    os.remove(out_path + "_if.npy")
    os.remove(out_path + ".npy")
    os.remove(unit_path)
    os.remove(os.path.join(shapenet_path, class_id, shape_id, "models/normalized_unit.vti"))
    return {"voxel_size": voxel_size, "padding": padding, "voxels": voxels, "centroid": centroid, "scale": scale}

def get_mesh_from_SDFGen_voxels(voxels, voxel_size, centroid, scale):
    verts, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=voxel_size/2, method="lewiner")
    recon = trimesh.Trimesh(verts, faces, vertex_normals=normals)
    recon = utils.scale_to_unit_cube(recon)
    recon = utils.rescale_unit_mesh(recon, shift=centroid, scale=scale)
    return recon, voxel_size/2
