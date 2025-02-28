# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.models import Surface
from chimerax.core.errors import UserError
import numpy as np


def save_dms(session, filename, models = None):

    if models is None:
        models = session.models.list()

    models_list = list(models)  # Convert OrderedSet to list
    
    if len(models_list) == 0:
        raise UserError("No model available.")

    molecular_surfaces = models_list[0].surfaces()

    if len(molecular_surfaces) == 0:
        raise UserError(f"No surface available for model {models[0].id}.")
    elif len(molecular_surfaces) > 1:
        raise UserError(f"More than one surface available for model {models[0].id}.")
    else:
        surface = molecular_surfaces[0]

    coords = surface.atom_coords()
    vertices = surface._vertices
    normals = surface._normals
    triangles = surface._triangles
    v_map = surface.vertex_to_atom_map()
    vses = compute_ses_areas_per_vertex(vertices, triangles)
    vtypes = classify_vertex_types(coords, vertices, normals, triangles)

    with open(filename, 'w') as file:

        for i, a in enumerate(surface.atoms):
            atomPart = atomFormat(a)
            c = coords[i]
            file.write( "%s%8.3f %8.3f %8.3f A\n" % (atomPart, c[0], c[1], c[2]) )

            for vi in np.where(v_map == i)[0].tolist():
                v = vertices[vi]
                vtype = vtypes[vi]
                try:
                    dmsType = ('S', 'R', 'C')[vtype-1]
                except IndexError:
                    raise ValueError(f"Vertex type {vtype} not in range 1-3")
                file.write("%s%8.3f %8.3f %8.3f S%s0 %6.3f" % (atomPart, v[0], v[1], v[2], dmsType, vses[vi]))
                file.write(" %6.3f %6.3f %6.3f\n" % tuple(normals[vi]))


def atomFormat(a):
  r = a.residue
  insertion_code = "" if r.insertion_code == ' ' else r.insertion_code
  chain_id = "*" if len(r.chain_id) > 1 else r.chain_id
  resseq = str(r.number) + insertion_code + chain_id
  return "%3s %4s %4.4s" % (r.name, resseq, a.name)


def compute_ses_areas_per_vertex(vertices, triangles):
    """
    Compute per-vertex SES (Solvent-Excluded Surface) areas from a molecular surface mesh.
    
    Parameters:
        vertices (np.ndarray): Nx3 array of surface vertex coordinates.
        triangles (np.ndarray): Mx3 array of vertex indices forming triangles.
    
    Returns:
        np.ndarray: Length-N array of SES areas per vertex.
    """
    # Compute vectors for two triangle edges
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    
    # Compute triangle normal using cross product
    cross_prod = np.cross(v1 - v0, v2 - v0)
    
    # Compute triangle areas (0.5 * magnitude of cross product)
    triangle_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    
    # Initialize per-vertex SES area accumulator
    vertex_ses_areas = np.zeros(len(vertices))  
    
    # Distribute triangle areas to its 3 vertices equally
    for tri_idx, tri in enumerate(triangles):
        for vert_idx in tri:
            vertex_ses_areas[vert_idx] += triangle_areas[tri_idx] / 3.0  # Each vertex gets 1/3

    return vertex_ses_areas


def classify_vertex_types(atom_coords, vertices, normals, triangles, probe_radius=1.4):
    """
    Approximate vtypes for ChimeraX MolecularSurface.
    
    Returns:
        vtypes: N-length array of integer vertex classifications
            - 1 = toric reentrant
            - 2 = inside reentrant
            - 3 = inside contact
    """

    # Step 1: Compute distance to closest atom
    dists = np.linalg.norm(vertices[:, np.newaxis, :] - atom_coords, axis=2)
    min_dists = np.min(dists, axis=1)

    # Step 2: Compute convexity via dot product of normals and atom vectors
    atom_vectors = atom_coords[np.argmin(dists, axis=1)] - vertices
    atom_vectors /= np.linalg.norm(atom_vectors, axis=1, keepdims=True)  # Normalize
    convexity = np.einsum('ij,ij->i', normals, atom_vectors)

    # Step 3: Classify vertex types
    vtypes = np.zeros(len(vertices), dtype=int)

    # Toric reentrant (1): Close to solvent-excluded cavities, slightly concave
    vtypes[(min_dists > probe_radius) & (convexity < 0)] = 1  

    # Inside reentrant (2): Deeply buried, strongly concave
    vtypes[(min_dists > probe_radius * 1.5) & (convexity < -0.5)] = 2  

    # Inside contact (3): Directly on the molecular surface
    vtypes[min_dists <= probe_radius] = 3  

    return vtypes
