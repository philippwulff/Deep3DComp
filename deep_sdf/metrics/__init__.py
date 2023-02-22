import trimesh
from deep_sdf.metrics.chamfer import compute_chamfer
from deep_sdf.utils import as_mesh, trimesh_to_pytorch3d_meshes
from pytorch3d.loss import mesh_normal_consistency


def compute_metric(gt_mesh=None, gen_mesh=None, num_mesh_samples=30000, metric="chamfer"):
    if gt_mesh is not None and isinstance(gt_mesh, str):
        gt_mesh = as_mesh(trimesh.load_mesh(gt_mesh))
    if gen_mesh is not None and isinstance(gen_mesh, str):
        gen_mesh = as_mesh(trimesh.load_mesh(gen_mesh))
        
    if metric == "chamfer" and gt_mesh is not None and gen_mesh is not None:
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
        gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
        return compute_chamfer(gen_points_sampled, gt_points_sampled)
    elif metric == "normal_consistency":
        meshes = trimesh_to_pytorch3d_meshes([gen_mesh])
        return mesh_normal_consistency(meshes)
    else:
        return NotImplementedError(f"Chosen metric '{metric}' does not exist.")