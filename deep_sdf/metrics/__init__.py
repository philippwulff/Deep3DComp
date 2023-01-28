import trimesh
from deep_sdf.metrics.chamfer import compute_chamfer


def compute_metric(gt_mesh, gen_mesh, num_mesh_samples=30000, metric="chamfer"):
    if gt_mesh.isinstance(str):
        pass # TODO
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    if metric == "chamfer":
        return compute_chamfer(gen_points_sampled, gt_points_sampled)
    else:
        return NotImplementedError(f"Chosen metric '{metric}' does not exist.")