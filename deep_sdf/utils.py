#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch
import trimesh
import numpy as np
from typing import Union, List
import math
import os
if not os.name == "nt":
    # We do not import this on Windows.
    from pytorch3d.structures import Meshes


R_x = lambda rad: np.array([[1, 0,           0,            0],
                            [0, np.cos(rad), -np.sin(rad), 0],
                            [0, np.sin(rad), np.cos(rad),  0],
                            [0, 0,           0,            1]])

R_y = lambda rad: np.array([[np.cos(rad),  0, np.sin(rad), 0],
                            [0,            1, 0,           0],
                            [-np.sin(rad), 0, np.cos(rad), 0],
                            [0,            0, 0,           1]])

R_z = lambda rad: np.array([[np.cos(rad), -np.sin(rad), 0, 0],
                            [np.sin(rad), np.cos(rad),  0, 0],
                            [0,           0,            1, 0],
                            [0,           0,            0, 1]])


def rotate(x: np.array, alpha=0, beta=0, gamma=0) -> np.array:
    """
    Rotate a vector or matrix about 
        - `alpha` rad around the X axis,
        - `beta` rad around the Y axis and
        - `gamma` rad around the Z axis.
    """
    return R_z(gamma) @ R_y(beta) @ R_x(alpha) @ x


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    logger.handlers.clear()     # Remove existing default handlers.
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdfComp - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf


def psnr(mse: Union[torch.Tensor, np.array]) -> Union[torch.Tensor, np.array]:
    """Peak Signal to Noise Ratio. mse has range [0, 1]"""
    if isinstance(mse, torch.Tensor):
        return 20 * torch.log10(1/torch.sqrt(mse))
    elif isinstance(mse, np.array):
        return 20 * np.log10(1/np.sqrt(mse))
    else:
        raise NotImplementedError

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    From: https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def comp_fc_net_frac_params(num_params, codelength, div):
    """Returns the width of an equal-width FC network with num_params/div parameters.
    lz = 512
    cl = 256
    np = lz*(cl+3) + lz*lz*7 + lz
    """
    return - (codelength+4)/(7*2) + math.sqrt(((codelength+4)/(7*2))**2 + (num_params/(div*7)))


def scale_to_unit_sphere(mesh, return_stats=False):
    """
    From: https://github.com/marian42/mesh_to_sdf/blob/master/mesh_to_sdf/utils.py
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    if return_stats:
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), mesh.bounding_box.centroid, np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def scale_to_unit_cube(mesh, return_stats=False, extent=1.0):
    """
    From: https://github.com/marian42/mesh_to_sdf/blob/master/mesh_to_sdf/utils.py
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents) * extent
    if return_stats:
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), mesh.bounding_box.centroid, np.max(mesh.bounding_box.extents)*extent / 2

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def rescale_unit_mesh(mesh: trimesh.Trimesh, shift: np.ndarray, scale: np.ndarray):
    vertices = mesh.vertices * scale + shift
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def trimesh_to_pytorch3d_meshes(meshes: List[trimesh.Trimesh]):
    """Converts a Trimesh into a Meshes object."""
    verts = [torch.Tensor(_.vertices) for _ in meshes]
    faces = [torch.Tensor(_.faces) for _ in meshes]
    return Meshes(verts, faces)