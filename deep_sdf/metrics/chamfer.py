#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just points, sampled from the surface (see
               compute_metrics.py for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """
    try:
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    except IndexError as e:
        raise IndexError

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    return compute_chamfer(gen_points_sampled, gt_points_np)


def compute_chamfer(gen_points_sampled, gt_points_sampled) -> float:
    """This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gen_points_sampled: np.array of points sampled from the generated mesh surface.
    gt_points_sampled: np.array of points sampled from the GT mesh surface.
    """
    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_sampled)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_sampled)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return float(gt_to_gen_chamfer + gen_to_gt_chamfer), np.concatenate((one_distances, two_distances), axis=0)


def compute_trimesh_iou():
    # TODO
    pass

def compute_trimesh_emd():
    # TODO
    pass