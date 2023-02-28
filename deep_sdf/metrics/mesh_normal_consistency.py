import os
import logging
import trimesh
from deep_sdf.utils import as_mesh, trimesh_to_pytorch3d_meshes
import torch

# We do not import this on Windows.
if not os.name == 'nt':
    logging.info("Not loading PyTorch3d because we are on windows.")
    from pytorch3d.loss import mesh_normal_consistency


def compute_mesh_normal_consistency(mesh: trimesh.Trimesh):
    """Measure surface smoothness."""
    if os.name == 'nt':
        # System is windows.
        logging.error("Cannot compute mesh normal consistency on Windows.")
        return torch.nan
    meshes = trimesh_to_pytorch3d_meshes([mesh])
    return mesh_normal_consistency(meshes)
