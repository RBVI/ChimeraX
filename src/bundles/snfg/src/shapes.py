# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
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

"""
Geometric shape primitives for SNFG visualization.

Provides vertex arrays and triangle arrays for the shapes used in SNFG:
sphere, cube, diamond (octahedron), cone, star, hexagon, pentagon, rectangle.
"""

import numpy as np
from math import pi, sin, cos, sqrt


def sphere_geometry(radius=1.0, divisions=320):
    """
    Generate a sphere via icosahedral subdivision.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    divisions : int
        Approximate number of triangles. Higher = smoother.
        320 gives a nice smooth sphere (3 subdivision levels).

    Returns (vertices, triangles) as numpy arrays.
    """
    from chimerax.geometry.sphere import sphere_triangulation
    # sphere_triangulation returns (vertices, triangles) for a unit sphere
    # with approximately 'divisions' triangles
    varray, tarray = sphere_triangulation(divisions)
    varray = varray * radius
    return varray, tarray


def cube_geometry(size=1.0):
    """
    Generate a cube centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    s = size / 2
    vertices = np.array([
        # Front face
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        # Back face
        [-s, -s, -s], [-s,  s, -s], [ s,  s, -s], [ s, -s, -s],
        # Top face
        [-s,  s, -s], [-s,  s,  s], [ s,  s,  s], [ s,  s, -s],
        # Bottom face
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
        # Right face
        [ s, -s, -s], [ s,  s, -s], [ s,  s,  s], [ s, -s,  s],
        # Left face
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
    ], dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],      # Front
        [4, 5, 6], [4, 6, 7],      # Back
        [8, 9, 10], [8, 10, 11],   # Top
        [12, 13, 14], [12, 14, 15], # Bottom
        [16, 17, 18], [16, 18, 19], # Right
        [20, 21, 22], [20, 22, 23], # Left
    ], dtype=np.int32)

    return vertices, triangles


def diamond_geometry(size=1.0):
    """
    Generate a diamond (octahedron) centered at origin.

    An octahedron has 6 vertices and 8 triangular faces.
    Returns (vertices, triangles) as numpy arrays.
    """
    s = size / 2
    vertices = np.array([
        [ 0,  s,  0],  # Top
        [ 0, -s,  0],  # Bottom
        [ s,  0,  0],  # Right
        [-s,  0,  0],  # Left
        [ 0,  0,  s],  # Front
        [ 0,  0, -s],  # Back
    ], dtype=np.float32)

    # 8 triangular faces
    triangles = np.array([
        [0, 4, 2],  # Top-front-right
        [0, 2, 5],  # Top-right-back
        [0, 5, 3],  # Top-back-left
        [0, 3, 4],  # Top-left-front
        [1, 2, 4],  # Bottom-right-front
        [1, 5, 2],  # Bottom-back-right
        [1, 3, 5],  # Bottom-left-back
        [1, 4, 3],  # Bottom-front-left
    ], dtype=np.int32)

    return vertices, triangles


def cone_geometry(radius=1.0, height=1.0, divisions=24):
    """
    Generate a cone with apex at top, base at bottom, centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    # Apex vertex
    apex = np.array([[0, height/2, 0]], dtype=np.float32)

    # Base center vertex
    base_center = np.array([[0, -height/2, 0]], dtype=np.float32)

    # Base circle vertices
    angles = np.linspace(0, 2*pi, divisions, endpoint=False)
    base_circle = np.zeros((divisions, 3), dtype=np.float32)
    base_circle[:, 0] = radius * np.cos(angles)
    base_circle[:, 2] = radius * np.sin(angles)
    base_circle[:, 1] = -height/2

    # Combine vertices: apex (0), base_center (1), base_circle (2 to divisions+1)
    vertices = np.vstack([apex, base_center, base_circle])

    # Side triangles (apex to base circle)
    side_tris = []
    for i in range(divisions):
        next_i = (i + 1) % divisions
        side_tris.append([0, i + 2, next_i + 2])

    # Base triangles (base_center to base circle)
    base_tris = []
    for i in range(divisions):
        next_i = (i + 1) % divisions
        base_tris.append([1, next_i + 2, i + 2])

    triangles = np.array(side_tris + base_tris, dtype=np.int32)

    return vertices, triangles


def star_geometry(size=1.0, points=5, inner_ratio=0.4):
    """
    Generate a 3D star shape (extruded 2D star).

    Returns (vertices, triangles) as numpy arrays.
    """
    thickness = size * 0.3
    outer_r = size / 2
    inner_r = outer_r * inner_ratio

    # Generate 2D star points
    angles = []
    radii = []
    for i in range(points * 2):
        angle = i * pi / points - pi/2
        angles.append(angle)
        radii.append(outer_r if i % 2 == 0 else inner_r)

    # Front face vertices
    front_verts = []
    for angle, r in zip(angles, radii):
        front_verts.append([r * cos(angle), r * sin(angle), thickness/2])

    # Back face vertices
    back_verts = []
    for angle, r in zip(angles, radii):
        back_verts.append([r * cos(angle), r * sin(angle), -thickness/2])

    # Center vertices for front and back
    front_center = [0, 0, thickness/2]
    back_center = [0, 0, -thickness/2]

    n = points * 2
    vertices = np.array(
        front_verts + back_verts + [front_center, back_center],
        dtype=np.float32
    )

    triangles = []
    fc_idx = 2 * n      # Front center index
    bc_idx = 2 * n + 1  # Back center index

    # Front face triangles
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([fc_idx, i, next_i])

    # Back face triangles
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([bc_idx, n + next_i, n + i])

    # Side triangles
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([i, n + i, next_i])
        triangles.append([next_i, n + i, n + next_i])

    return vertices, np.array(triangles, dtype=np.int32)


def hexagon_geometry(size=1.0, thickness=None):
    """
    Generate a hexagonal prism centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    if thickness is None:
        thickness = size * 0.4

    r = size / 2

    # Hexagon vertices (6 points)
    angles = np.linspace(0, 2*pi, 6, endpoint=False)

    # Front face vertices
    front_verts = np.zeros((6, 3), dtype=np.float32)
    front_verts[:, 0] = r * np.cos(angles)
    front_verts[:, 1] = r * np.sin(angles)
    front_verts[:, 2] = thickness / 2

    # Back face vertices
    back_verts = np.zeros((6, 3), dtype=np.float32)
    back_verts[:, 0] = r * np.cos(angles)
    back_verts[:, 1] = r * np.sin(angles)
    back_verts[:, 2] = -thickness / 2

    # Centers
    front_center = np.array([[0, 0, thickness/2]], dtype=np.float32)
    back_center = np.array([[0, 0, -thickness/2]], dtype=np.float32)

    vertices = np.vstack([front_verts, back_verts, front_center, back_center])

    triangles = []
    n = 6
    fc_idx = 12  # Front center
    bc_idx = 13  # Back center

    # Front face
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([fc_idx, i, next_i])

    # Back face
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([bc_idx, n + next_i, n + i])

    # Side faces
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([i, n + i, next_i])
        triangles.append([next_i, n + i, n + next_i])

    return vertices, np.array(triangles, dtype=np.int32)


def pentagon_geometry(size=1.0, thickness=None):
    """
    Generate a pentagonal prism centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    if thickness is None:
        thickness = size * 0.4

    r = size / 2

    # Pentagon vertices (5 points)
    angles = np.linspace(0, 2*pi, 5, endpoint=False) - pi/2  # Start at top

    # Front face vertices
    front_verts = np.zeros((5, 3), dtype=np.float32)
    front_verts[:, 0] = r * np.cos(angles)
    front_verts[:, 1] = r * np.sin(angles)
    front_verts[:, 2] = thickness / 2

    # Back face vertices
    back_verts = np.zeros((5, 3), dtype=np.float32)
    back_verts[:, 0] = r * np.cos(angles)
    back_verts[:, 1] = r * np.sin(angles)
    back_verts[:, 2] = -thickness / 2

    # Centers
    front_center = np.array([[0, 0, thickness/2]], dtype=np.float32)
    back_center = np.array([[0, 0, -thickness/2]], dtype=np.float32)

    vertices = np.vstack([front_verts, back_verts, front_center, back_center])

    triangles = []
    n = 5
    fc_idx = 10  # Front center
    bc_idx = 11  # Back center

    # Front face
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([fc_idx, i, next_i])

    # Back face
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([bc_idx, n + next_i, n + i])

    # Side faces
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([i, n + i, next_i])
        triangles.append([next_i, n + i, n + next_i])

    return vertices, np.array(triangles, dtype=np.int32)


def rectangle_geometry(width=1.0, height=0.6, thickness=0.3):
    """
    Generate a rectangular prism (flat rectangle) centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    w, h, t = width/2, height/2, thickness/2

    vertices = np.array([
        # Front face
        [-w, -h,  t], [ w, -h,  t], [ w,  h,  t], [-w,  h,  t],
        # Back face
        [-w, -h, -t], [-w,  h, -t], [ w,  h, -t], [ w, -h, -t],
        # Top face
        [-w,  h, -t], [-w,  h,  t], [ w,  h,  t], [ w,  h, -t],
        # Bottom face
        [-w, -h, -t], [ w, -h, -t], [ w, -h,  t], [-w, -h,  t],
        # Right face
        [ w, -h, -t], [ w,  h, -t], [ w,  h,  t], [ w, -h,  t],
        # Left face
        [-w, -h, -t], [-w, -h,  t], [-w,  h,  t], [-w,  h, -t],
    ], dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)

    return vertices, triangles


def flat_diamond_geometry(size=1.0, thickness=0.3):
    """
    Generate a flat diamond (rhombus prism) centered at origin.

    Returns (vertices, triangles) as numpy arrays.
    """
    s = size / 2
    t = thickness / 2

    # Diamond shape in XY plane
    front_verts = np.array([
        [0,  s, t],   # Top
        [s,  0, t],   # Right
        [0, -s, t],   # Bottom
        [-s, 0, t],   # Left
    ], dtype=np.float32)

    back_verts = np.array([
        [0,  s, -t],
        [s,  0, -t],
        [0, -s, -t],
        [-s, 0, -t],
    ], dtype=np.float32)

    vertices = np.vstack([front_verts, back_verts])

    triangles = []
    n = 4

    # Front face
    triangles.append([0, 1, 2])
    triangles.append([0, 2, 3])

    # Back face
    triangles.append([4, 6, 5])
    triangles.append([4, 7, 6])

    # Side faces
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([i, n + i, next_i])
        triangles.append([next_i, n + i, n + next_i])

    return vertices, np.array(triangles, dtype=np.int32)


# Shape factory function
def get_shape_geometry(shape_type, size=1.0):
    """
    Get vertex and triangle arrays for a given shape type.

    Parameters
    ----------
    shape_type : str
        One of: 'sphere', 'cube', 'crossed_cube', 'diamond', 'cone',
        'divided_cone', 'rectangle', 'star', 'hexagon', 'pentagon', 'flat_diamond'
    size : float
        Overall size of the shape in Angstroms.

    Returns
    -------
    vertices : ndarray
        Nx3 array of vertex positions.
    triangles : ndarray
        Mx3 array of triangle vertex indices.
    """
    from .definitions import (SPHERE, CUBE, CROSSED_CUBE, DIAMOND, CONE,
                               DIVIDED_CONE, RECTANGLE, STAR, HEXAGON,
                               PENTAGON, FLAT_DIAMOND)

    if shape_type == SPHERE:
        return sphere_geometry(radius=size/2)
    elif shape_type in (CUBE, CROSSED_CUBE):
        return cube_geometry(size=size * 0.8)  # Slightly smaller to match sphere volume
    elif shape_type == DIAMOND:
        return diamond_geometry(size=size * 1.3)  # Scale to match visual size
    elif shape_type in (CONE, DIVIDED_CONE):
        return cone_geometry(radius=size/2, height=size)
    elif shape_type == RECTANGLE:
        return rectangle_geometry(width=size, height=size*0.6, thickness=size*0.3)
    elif shape_type == STAR:
        return star_geometry(size=size)
    elif shape_type == HEXAGON:
        return hexagon_geometry(size=size * 1.15)
    elif shape_type == PENTAGON:
        return pentagon_geometry(size=size)
    elif shape_type == FLAT_DIAMOND:
        return flat_diamond_geometry(size=size * 1.3, thickness=size*0.3)
    else:
        # Default to sphere for unknown types
        return sphere_geometry(radius=size/2)
