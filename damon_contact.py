# preprocess/damon_contact.py
import numpy as np
import trimesh      # pip install trimesh
from scipy.spatial import cKDTree

'''
Strategy

A body vertex is in contact if:
distance(body_vertex, object_surface) < threshold (e.g. 2 cm)

Vertex–object distance labeling
. Works for DAMON
. Converts sparse → dense labels
. Research-standard approach
'''
def compute_contact_labels(body_verts, object_mesh, thresh=0.02):
    """
    body_verts: (V,3) SMPL vertices
    object_mesh: trimesh.Trimesh
    """
    obj_pts = object_mesh.sample(20000)
    tree = cKDTree(obj_pts)

    dists, _ = tree.query(body_verts, k=1)
    contact = (dists < thresh).astype(np.int64)
    return contact



'''
Build DAMON training samples:
Save these as .npz or .pkl for fast loading.
'''
def build_damon_sample(image_path, smpl_verts, object_mesh):
    contact = compute_contact_labels(smpl_verts, object_mesh)
    return {
        "image_path": image_path,
        "smpl_vertices": smpl_verts,
        "contact_labels": contact
    }



