import os
import trimesh
from deep_sdf.metrics.chamfer import compute_chamfer
from deep_sdf.metrics.mesh_normal_consistency import compute_mesh_normal_consistency
from deep_sdf.utils import as_mesh
import point_cloud_utils as pcu


def compute_metric(gt_mesh=None, gen_mesh=None, num_mesh_samples=30000, metric="chamfer"):
    if gt_mesh is not None and isinstance(gt_mesh, str):
        gt_mesh = as_mesh(trimesh.load_mesh(gt_mesh))
    if gen_mesh is not None and isinstance(gen_mesh, str):
        gen_mesh = as_mesh(trimesh.load_mesh(gen_mesh))
        
    if gt_mesh is not None and gen_mesh is not None:
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
        gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
        if metric == "chamfer": 
            return compute_chamfer(gen_points_sampled, gt_points_sampled)
        elif metric == "hausdorff":
            return pcu.hausdorff_distance(gen_points_sampled, gt_points_sampled)
    elif metric == "normal_consistency":
        return compute_mesh_normal_consistency(gen_mesh)
    else:
        return NotImplementedError(f"Chosen metric '{metric}' does not exist.")