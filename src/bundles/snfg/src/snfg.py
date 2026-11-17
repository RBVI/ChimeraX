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
SNFG (Symbol Nomenclature for Glycans) visualization for ChimeraX.

Displays carbohydrate residues as colored 3D geometric shapes placed at
sugar ring centroids, following the SNFG standard.  The symbols are child
drawings of the structure they depict, like ribbons and nucleotides.
"""

import weakref

import numpy as np
from chimerax.core.state import State, StateManager
from chimerax.atomic import AtomicShapeDrawing, AtomicShapeInfo
from chimerax.graphics import Drawing
from chimerax.surface import calculate_vertex_normals

from .definitions import (
    COLORS, MONOSACCHARIDE_SYMBOLS, PDB_TO_SNFG, DEFAULT_SIZE
)
from .shapes import get_shape_geometry

CONNECTION_COLOR = (180, 180, 180)


class SNFGShapesDrawing(AtomicShapeDrawing):
    """Drawing containing all SNFG glycan symbols with selection support.

    Each glycan symbol is stored as a shape with associated residue atoms,
    enabling bidirectional selection: clicking a symbol selects its atoms,
    and selecting atoms highlights the symbol.
    """

    def __init__(self, name="SNFG shapes"):
        super().__init__(name)
        self._residue_to_shape_index = {}  # residue → shape index
        self._shape_residues = []  # ordered list parallel to self._shapes
        self._shape_centroids = []  # centroids for connection drawing
        self._shape_sizes = []  # current size per shape, parallel to self._shapes

    def add_residues_batch(self, residue_info_list):
        """Add multiple residue shapes efficiently.

        Parameters
        ----------
        residue_info_list : list of (residue, shape_type, color_name, size) tuples

        Returns
        -------
        int
            Number of shapes added.
        """
        shape_infos = []
        new_residues = []
        new_centroids = []
        new_sizes = []

        for residue, shape_type, color_name, size in residue_info_list:
            if residue in self._residue_to_shape_index:
                continue

            # Get geometry
            vertices, triangles = get_shape_geometry(shape_type, size)

            # Get ring centroid
            centroid = _ring_centroid(residue)
            if centroid is None:
                continue

            # Get ring plane normal and linkage direction for orientation
            ring_normal = _ring_plane_normal(residue)
            linkage_dir = _find_linkage_direction(residue, centroid)

            # Compute rotation to align shape with ring plane
            rotation = _compute_shape_rotation(shape_type, ring_normal, linkage_dir)
            if rotation is not None:
                vertices = np.dot(vertices, rotation.T)

            # Translate to centroid
            vertices = vertices + centroid

            # Calculate normals
            normals = calculate_vertex_normals(vertices, triangles)

            # Get color
            rgb = COLORS.get(color_name, (190, 190, 190))
            color = np.array((*rgb, 255), dtype=np.uint8)

            # Description for picking tooltip
            description = f"SNFG {residue.name} ({residue})"

            shape_infos.append(AtomicShapeInfo(
                vertices, normals, triangles, color,
                atoms=residue.atoms, description=description
            ))
            new_residues.append(residue)
            new_centroids.append(centroid)
            new_sizes.append(float(size))

        if not shape_infos:
            return 0

        # Record shape indices before adding
        start_index = len(self._shapes)

        # add_shapes() requires empty geometry, so use it only when empty
        # Otherwise fall back to individual add_shape() calls
        if self.vertices is None:
            # Batch add all shapes (more efficient)
            self.add_shapes(shape_infos)
        else:
            # Add shapes one at a time to existing geometry
            for info in shape_infos:
                self.add_shape(info.vertices, info.normals, info.triangles,
                              info.color, atoms=info.atoms,
                              description=info.description)

        # Update residue tracking
        for i, residue in enumerate(new_residues):
            self._residue_to_shape_index[residue] = start_index + i
        self._shape_residues.extend(new_residues)
        self._shape_centroids.extend(new_centroids)
        self._shape_sizes.extend(new_sizes)

        return len(shape_infos)

    def has_residue(self, residue):
        """Check if a residue has a shape."""
        return residue in self._residue_to_shape_index

    def get_centroid(self, residue):
        """Get the centroid for a residue's shape."""
        if residue not in self._residue_to_shape_index:
            return None
        shape_idx = self._residue_to_shape_index[residue]
        return self._shape_centroids[shape_idx]

    def residues(self):
        """Return iterator over residues with shapes."""
        return iter(self._shape_residues)

    def clear_shapes(self):
        """Remove all shapes."""
        # Clear geometry
        self.set_geometry(None, None, None)
        self.vertex_colors = None
        self._shapes.clear()
        self._selected_shapes.clear()
        self.highlighted_triangles_mask = None

        # Clear residue tracking
        self._residue_to_shape_index.clear()
        self._shape_residues.clear()
        self._shape_centroids.clear()
        self._shape_sizes.clear()

    def _add_selected_shape(self, shape):
        """Override to also select intra-residue bonds."""
        super()._add_selected_shape(shape)
        # Also select bonds within the residue
        if shape.atoms:
            shape.atoms.intra_bonds.selected = True

    def _add_selected_shapes(self, shapes):
        """Override to also select intra-residue bonds."""
        super()._add_selected_shapes(shapes)
        # Also select bonds within each residue
        for s in shapes:
            if s.atoms:
                s.atoms.intra_bonds.selected = True

    def _remove_selected_shape(self, shape):
        """Override to also deselect intra-residue bonds."""
        super()._remove_selected_shape(shape)
        if shape.atoms:
            shape.atoms.intra_bonds.selected = False

    def _remove_selected_shapes(self, shapes):
        """Override to also deselect intra-residue bonds."""
        super()._remove_selected_shapes(shapes)
        for s in shapes:
            if s.atoms:
                s.atoms.intra_bonds.selected = False

    def _shape_vertex_range(self, shape_idx):
        """Return (start, end) vertex indices for a shape, derived from its triangles."""
        if self.triangles is None or shape_idx >= len(self._shapes):
            return None
        s = self._shapes[shape_idx]
        tri = self.triangles[s.triangle_range]
        if tri.size == 0:
            return None
        return int(tri.min()), int(tri.max()) + 1

    def apply_shape_sizes(self, target_sizes):
        """Scale each shape's vertices around its centroid to match target size.

        Parameters
        ----------
        target_sizes : sequence of float
            Target size for each shape; length must equal number of shapes.
        """
        if self.vertices is None or len(target_sizes) != len(self._shapes):
            return
        verts = None
        for i, target in enumerate(target_sizes):
            current = self._shape_sizes[i]
            if current <= 0 or abs(target - current) < 1e-9:
                continue
            r = self._shape_vertex_range(i)
            if r is None:
                continue
            v_start, v_end = r
            if verts is None:
                verts = self.vertices.copy()
            centroid = np.asarray(self._shape_centroids[i], dtype=verts.dtype)
            factor = target / current
            verts[v_start:v_end] = (verts[v_start:v_end] - centroid) * factor + centroid
            self._shape_sizes[i] = target
        if verts is not None:
            # set_geometry() resets vertex_colors and the triangle/edge/highlight
            # masks. Uniform scaling preserves their validity (same vertex/
            # triangle count and order), so save and restore around the call.
            saved_colors = self.vertex_colors
            saved_highlight = self.highlighted_triangles_mask
            saved_triangle_mask = self.triangle_mask
            saved_edge_mask = self.edge_mask
            self.set_geometry(verts, self.normals, self.triangles)
            if saved_colors is not None:
                self.vertex_colors = saved_colors
            if saved_highlight is not None:
                self.highlighted_triangles_mask = saved_highlight
            if saved_triangle_mask is not None:
                self.triangle_mask = saved_triangle_mask
            if saved_edge_mask is not None:
                self.edge_mask = saved_edge_mask

    def set_alpha(self, alpha):
        """Set the opacity (0-1) of every symbol."""
        colors = self.vertex_colors
        if colors is None or len(colors) == 0:
            return
        a = int(round(255 * min(max(alpha, 0.0), 1.0)))
        if (colors[:, 3] == a).all():
            return
        colors = colors.copy()
        colors[:, 3] = a
        self.vertex_colors = colors


class SNFGDepiction:
    """The SNFG symbols and connections drawn inside one structure."""

    def __init__(self, structure):
        self.structure = structure
        self.sizes = {}  # residue → symbol size, for every depicted residue
        self.show_atoms = False
        self.alpha = 1.0
        self.hidden_residues = set()  # residues whose atoms we hid
        self.shapes = SNFGShapesDrawing()
        self.connections = Drawing("SNFG connections")
        structure.add_drawing(self.shapes)
        structure.add_drawing(self.connections)
        self._changes_handler = structure.triggers.add_handler(
            'changes', self._structure_changed)

    def delete(self):
        self.structure.triggers.remove_handler(self._changes_handler)
        self.structure.remove_drawing(self.shapes)
        self.structure.remove_drawing(self.connections)

    def set_residues(self, sizes):
        """Depict the residues in ``sizes`` (residue → size), reusing the
        existing symbols when only their sizes differ.

        Returns the number of residues that have a symbol.
        """
        sizes = {r: s for r, s in sizes.items() if not r.deleted}
        if sizes.keys() == self.sizes.keys():
            self.shapes.apply_shape_sizes([sizes[r] for r in self.shapes.residues()])
            self.sizes = sizes
        else:
            self._build(sizes)
        return len(self.sizes)

    def _build(self, sizes):
        info = []
        for residue, size in sizes.items():
            symbol = identify_sugar(residue)
            if symbol is not None:
                info.append((residue, *symbol, size))
        self.shapes.clear_shapes()
        self.shapes.add_residues_batch(info)
        self.sizes = {r: sizes[r] for r in self.shapes.residues()}
        self._build_connections()
        self.set_appearance(self.show_atoms, self.alpha)

    def set_appearance(self, show_atoms, alpha=1.0):
        """Connections are only drawn when the sugar atoms are hidden."""
        self.show_atoms = show_atoms
        self.alpha = alpha
        self.shapes.set_alpha(alpha)
        self.connections.color = (*CONNECTION_COLOR, int(round(255 * alpha)))
        self.connections.display = not show_atoms

    def _build_connections(self):
        """Draw connections between linked sugar residues."""
        shapes = self.shapes
        seen = set()
        connections = []
        for residue in shapes.residues():
            for atom in residue.atoms:
                for neighbor in atom.neighbors:
                    other = neighbor.residue
                    if other is residue or not shapes.has_residue(other):
                        continue
                    pair = frozenset([residue, other])
                    if pair not in seen:
                        seen.add(pair)
                        connections.append((shapes.get_centroid(residue),
                                            shapes.get_centroid(other)))
        vertices, triangles = _cylinder_connections(connections, radius=0.2)
        if len(vertices) == 0:
            self.connections.set_geometry(None, None, None)
            return
        normals = calculate_vertex_normals(vertices, triangles)
        self.connections.set_geometry(vertices, normals, triangles)

    def hide_atoms(self):
        """Hide the atoms of every residue that has a symbol."""
        for residue in self.sizes:
            atoms = residue.atoms
            if atoms.displays.any():
                self.hidden_residues.add(residue)
                atoms.displays = False

    def restore_atoms(self):
        """Show the atoms we previously hid."""
        for residue in self.hidden_residues:
            if not residue.deleted:
                residue.atoms.displays = True
        self.hidden_residues.clear()

    def _structure_changed(self, trigger_name, data):
        structure, changes = data
        if changes.num_deleted_atoms():
            self.hidden_residues = {r for r in self.hidden_residues if not r.deleted}
        elif ('coord changed' not in changes.atom_reasons()
              and 'active_coordset changed' not in changes.structure_reasons()):
            return
        self._build({r: s for r, s in self.sizes.items() if not r.deleted})
        if not self.sizes:
            _manager(structure.session).remove(structure)


class SNFGManager(StateManager):
    """Saves and restores SNFG depictions in sessions and scenes."""

    VERSION = 1

    def __init__(self, session):
        self._structures = weakref.WeakSet()

    def depiction(self, structure, create=False):
        depiction = getattr(structure, '_snfg_depiction', None)
        if depiction is None and create:
            depiction = structure._snfg_depiction = SNFGDepiction(structure)
            self._structures.add(structure)
        return depiction

    def depictions(self):
        return {s: s._snfg_depiction for s in self._structures if not s.deleted}

    def remove(self, structure):
        depiction = self.depiction(structure)
        if depiction is None:
            return
        self._structures.discard(structure)
        del structure._snfg_depiction
        if not structure.deleted:
            depiction.delete()

    def _restore(self, structure, state, alpha=1.0):
        depiction = self.depiction(structure, create=True)
        depiction.hidden_residues = {r for r in state['hidden'] if not r.deleted}
        depiction.set_appearance(state['show_atoms'], alpha)
        if not depiction.set_residues(state['sizes']):
            self.remove(structure)

    def take_snapshot(self, session, flags):
        depictions = {
            structure: {
                'sizes': dict(d.sizes),
                'show_atoms': d.show_atoms,
                'hidden': list(d.hidden_residues),
            }
            for structure, d in self.depictions().items()
        }
        if flags == State.SCENE:
            return depictions
        return {'version': self.VERSION, 'depictions': depictions}

    @classmethod
    def restore_snapshot(cls, session, data):
        mgr = _manager(session)
        for structure, state in data['depictions'].items():
            if structure is not None and not structure.deleted:
                mgr._restore(structure, state)
        return mgr

    def reset_state(self, session):
        for structure in self.depictions():
            self.remove(structure)

    def restore_scene(self, scene_data):
        """Scene interface; ``scene_data`` is None for a scene saved before
        any structure had SNFG symbols.  Atom visibility is restored by the
        structures themselves."""
        scene_data = scene_data or {}
        for structure in self.depictions():
            if structure not in scene_data:
                self.remove(structure)
        for structure, state in scene_data.items():
            if not structure.deleted:
                self._restore(structure, state)

    def interpolate_scene(self, scene1_data, scene2_data, fraction, *, switchover=False):
        scene1_data = scene1_data or {}
        scene2_data = scene2_data or {}
        for structure in {*scene1_data, *scene2_data, *self.depictions()}:
            if structure.deleted:
                continue
            state1 = scene1_data.get(structure)
            state2 = scene2_data.get(structure)
            if state1 is None and state2 is None:
                self.remove(structure)
            elif state1 is None:
                self._restore(structure, state2, alpha=fraction)
            elif state2 is None:
                self._restore(structure, state1, alpha=1.0 - fraction)
            else:
                state = dict(state2 if switchover else state1)
                sizes1, sizes2 = state1['sizes'], state2['sizes']
                state['sizes'] = {
                    r: ((1.0 - fraction) * sizes1[r] + fraction * sizes2[r]
                        if r in sizes1 and r in sizes2 else size)
                    for r, size in state['sizes'].items()
                }
                self._restore(structure, state)


def _manager(session):
    if not hasattr(session, 'snfg'):
        session.snfg = SNFGManager(session)
    return session.snfg


def _cylinder_connections(connections, radius=0.2, divisions=8):
    """
    Generate cylinder geometry for connection lines.

    Parameters
    ----------
    connections : list of (point1, point2) tuples
        Each tuple contains two numpy arrays representing endpoints.
    radius : float
        Cylinder radius.
    divisions : int
        Number of divisions around the cylinder.

    Returns (vertices, triangles) as numpy arrays.
    """
    if not connections:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    for p1, p2 in connections:
        p1 = np.asarray(p1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)

        # Direction and length
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 0.001:
            continue
        direction = direction / length

        # Find perpendicular vectors
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, np.array([1, 0, 0], dtype=np.float32))
        else:
            perp1 = np.cross(direction, np.array([0, 1, 0], dtype=np.float32))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Generate circle vertices at each end
        angles = np.linspace(0, 2 * np.pi, divisions, endpoint=False)
        circle = radius * (np.outer(np.cos(angles), perp1) + np.outer(np.sin(angles), perp2))

        bottom_verts = p1 + circle
        top_verts = p2 + circle

        # Add center vertices for caps
        vertices = np.vstack([bottom_verts, top_verts, [p1], [p2]])
        all_vertices.append(vertices)

        # Triangles for the cylinder sides
        triangles = []
        n = divisions
        for i in range(n):
            next_i = (i + 1) % n
            # Side quad as two triangles
            triangles.append([vertex_offset + i, vertex_offset + n + i, vertex_offset + next_i])
            triangles.append([vertex_offset + next_i, vertex_offset + n + i, vertex_offset + n + next_i])

        # Bottom cap
        bottom_center = vertex_offset + 2 * n
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([bottom_center, vertex_offset + next_i, vertex_offset + i])

        # Top cap
        top_center = vertex_offset + 2 * n + 1
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([top_center, vertex_offset + n + i, vertex_offset + n + next_i])

        all_triangles.extend(triangles)
        vertex_offset += len(vertices)

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    return np.vstack(all_vertices).astype(np.float32), np.array(all_triangles, dtype=np.int32)


def _find_ring(residue):
    """
    Find the sugar ring in a residue.

    Looks for pyranose (6-membered) or furanose (5-membered) rings.
    Returns the ring object, or None if no ring found.
    """
    atoms = residue.atoms
    if len(atoms) == 0:
        return None

    # Get rings from bonds
    rings = []
    for bond in atoms.bonds.unique():
        try:
            bond_rings = bond.rings(cross_residue=False, all_size_threshold=6)
            for ring in bond_rings:
                if ring.size in (5, 6) and ring not in rings:
                    # Check if all ring atoms are in this residue
                    ring_atoms = ring.atoms
                    if all(a.residue == residue for a in ring_atoms):
                        rings.append(ring)
        except Exception:
            pass

    if not rings:
        return None

    # Use the first suitable ring (prefer 6-membered pyranose)
    pyranose_rings = [r for r in rings if r.size == 6]
    return pyranose_rings[0] if pyranose_rings else rings[0]


def _ring_centroid(residue):
    """
    Calculate the centroid of the sugar ring in a residue.

    Looks for pyranose (6-membered) or furanose (5-membered) rings.
    Returns the centroid as a numpy array, or None if no ring found.
    """
    ring = _find_ring(residue)
    if ring is not None:
        return np.mean(ring.atoms.coords, axis=0)

    # Fall back to residue center
    atoms = residue.atoms
    if len(atoms) > 0:
        return np.mean(atoms.coords, axis=0)
    return None


def _ring_plane_normal(residue):
    """
    Calculate the normal vector of the sugar ring plane.

    Uses SVD to find the best-fit plane through the ring atoms.
    Returns a unit normal vector, or None if no ring found.
    """
    ring = _find_ring(residue)
    if ring is None:
        return None

    coords = ring.atoms.coords
    centroid = np.mean(coords, axis=0)

    # Center the coordinates
    centered = coords - centroid

    # Use SVD to find the plane normal
    # The normal is the eigenvector corresponding to the smallest singular value
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2]  # Last row is the normal to the best-fit plane

    return normal / np.linalg.norm(normal)


def _find_linkage_direction(residue, centroid):
    """
    Find the direction from this sugar to a linked sugar residue.

    Looks for glycosidic bonds (bonds to atoms in other residues that are
    also sugars). Returns a normalized direction vector, or None if no
    linked sugar is found.
    """
    # Look for inter-residue bonds from this sugar
    linked_centroids = []

    for atom in residue.atoms:
        for neighbor in atom.neighbors:
            other_res = neighbor.residue
            if other_res != residue:
                # Check if the other residue is also a sugar
                if identify_sugar(other_res) is not None:
                    other_centroid = _ring_centroid(other_res)
                    if other_centroid is not None:
                        linked_centroids.append(other_centroid)

    if not linked_centroids:
        return None

    # If multiple links, use the average direction
    # (though typically there's one primary linkage)
    avg_target = np.mean(linked_centroids, axis=0)
    direction = avg_target - centroid
    length = np.linalg.norm(direction)

    if length < 0.001:
        return None

    return direction / length


def _rotation_to_align_axis(source_axis, target_direction):
    """
    Compute a rotation matrix that aligns source_axis with target_direction.

    Parameters
    ----------
    source_axis : ndarray
        Unit vector for the source axis (e.g., [0, 0, 1] for Z).
    target_direction : ndarray
        Unit vector for the desired direction.

    Returns
    -------
    rotation : ndarray
        3x3 rotation matrix.
    """
    source = np.asarray(source_axis, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = np.asarray(target_direction, dtype=np.float64)
    target = target / np.linalg.norm(target)

    # Check if already aligned (or anti-aligned)
    dot = np.dot(source, target)

    if dot > 0.9999:
        # Already aligned
        return np.eye(3)
    elif dot < -0.9999:
        # Opposite direction - find any perpendicular axis to rotate around
        if abs(source[0]) < 0.9:
            perp = np.cross(source, np.array([1, 0, 0]))
        else:
            perp = np.cross(source, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        # 180 degree rotation around perp
        K = np.array([
            [0, -perp[2], perp[1]],
            [perp[2], 0, -perp[0]],
            [-perp[1], perp[0], 0]
        ])
        return np.eye(3) + 2 * np.dot(K, K)

    # General case: use Rodrigues' rotation formula
    axis = np.cross(source, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def _rotation_to_align_y(target_direction):
    """
    Compute a rotation matrix that aligns the Y axis with the target direction.
    """
    return _rotation_to_align_axis(np.array([0, 1, 0]), target_direction)


def _rotation_to_align_z(target_direction):
    """
    Compute a rotation matrix that aligns the Z axis with the target direction.
    """
    return _rotation_to_align_axis(np.array([0, 0, 1]), target_direction)


def _compute_shape_rotation(shape_type, ring_normal, linkage_dir):
    """
    Compute the rotation matrix for a shape to align with the ring plane.

    Parameters
    ----------
    shape_type : str
        The SNFG shape type.
    ring_normal : ndarray or None
        Normal vector to the sugar ring plane.
    linkage_dir : ndarray or None
        Direction toward the linked sugar.

    Returns
    -------
    rotation : ndarray or None
        3x3 rotation matrix, or None if no rotation needed.
    """
    from .definitions import (STAR, HEXAGON, PENTAGON, FLAT_DIAMOND, RECTANGLE,
                               CONE, DIVIDED_CONE, DIAMOND)

    # Flat shapes have their face perpendicular to Z axis
    flat_shapes = {STAR, HEXAGON, PENTAGON, FLAT_DIAMOND, RECTANGLE}

    # Vertical shapes have their primary axis along Y
    vertical_shapes = {CONE, DIVIDED_CONE, DIAMOND}

    if shape_type in flat_shapes:
        # Align Z axis with ring plane normal so the flat face lies on the ring
        if ring_normal is not None:
            rotation = _rotation_to_align_z(ring_normal)

            # Optionally rotate around Z to orient toward linkage
            if linkage_dir is not None:
                # Project linkage direction onto the ring plane
                linkage_in_plane = linkage_dir - np.dot(linkage_dir, ring_normal) * ring_normal
                if np.linalg.norm(linkage_in_plane) > 0.01:
                    linkage_in_plane = linkage_in_plane / np.linalg.norm(linkage_in_plane)

                    # After the first rotation, Z is aligned with ring_normal
                    # We want to rotate around Z to point Y toward linkage_in_plane
                    # Transform linkage_in_plane to the rotated coordinate system
                    linkage_local = np.dot(rotation.T, linkage_in_plane)

                    # Calculate angle to rotate around Z to align Y with linkage
                    angle = np.arctan2(linkage_local[0], linkage_local[1])

                    # Rotation around Z axis
                    c, s = np.cos(-angle), np.sin(-angle)
                    rot_z = np.array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]
                    ])
                    rotation = np.dot(rotation, rot_z)

            return rotation

    elif shape_type in vertical_shapes:
        # For vertical shapes, align Y with ring normal (standing up from ring)
        if ring_normal is not None:
            return _rotation_to_align_y(ring_normal)
        elif linkage_dir is not None:
            return _rotation_to_align_y(linkage_dir)

    else:
        # For spheres and other shapes, use linkage direction if available
        if linkage_dir is not None:
            return _rotation_to_align_y(linkage_dir)

    return None


def identify_sugar(residue):
    """
    Identify the SNFG symbol type for a residue.

    Returns (shape_type, color_name) tuple, or None if not a recognized sugar.
    """
    return _symbol_for_name(residue.name)


def _symbol_for_name(res_name):
    res_name = res_name.upper()

    # Try direct PDB code lookup
    if res_name in PDB_TO_SNFG:
        snfg_name = PDB_TO_SNFG[res_name]
        if snfg_name in MONOSACCHARIDE_SYMBOLS:
            return MONOSACCHARIDE_SYMBOLS[snfg_name]

    # Try direct SNFG name lookup (some structures use these)
    if res_name in MONOSACCHARIDE_SYMBOLS:
        return MONOSACCHARIDE_SYMBOLS[res_name]

    # Try case-insensitive match
    for snfg_name, symbol in MONOSACCHARIDE_SYMBOLS.items():
        if snfg_name.upper() == res_name:
            return symbol

    return None


def sugar_residues(structure):
    """Return the carbohydrate residues of a structure as a Residues collection."""
    residues = structure.residues
    names = residues.names
    sugar_names = [name for name in set(names) if _symbol_for_name(name) is not None]
    return residues.filter(np.isin(names, sugar_names))


def show_snfg(session, structures=None, size=DEFAULT_SIZE, replace=True,
               show_atoms=False, log=True):
    """
    Show SNFG symbols for carbohydrate residues.

    Parameters
    ----------
    session : Session
        ChimeraX session.
    structures : list of AtomicStructure, optional
        Structures to process. If None, use all open structures.
    size : float
        Size of SNFG symbols in Angstroms.
    replace : bool
        If True, replace existing symbols. If False, only add missing ones.
    show_atoms : bool
        If True, keep atoms visible with symbols shown inside rings.
        If False (default), hide sugar atoms.
    log : bool
        If True (default), log a summary message to the session logger.

    Returns the number of symbols shown.
    """
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)

    mgr = _manager(session)
    total_shown = 0

    for structure in structures:
        depiction = mgr.depiction(structure)
        sizes = {}
        if depiction is not None:
            if replace:
                depiction.restore_atoms()
                mgr.remove(structure)
            else:
                sizes.update(depiction.sizes)
        existing = set(sizes)
        for residue in sugar_residues(structure):
            sizes.setdefault(residue, size)
        if not sizes:
            continue

        depiction = mgr.depiction(structure, create=True)
        depiction.set_appearance(show_atoms)
        if not depiction.set_residues(sizes):
            mgr.remove(structure)
            continue
        total_shown += len(depiction.sizes.keys() - existing)
        if show_atoms:
            depiction.restore_atoms()
        else:
            depiction.hide_atoms()

    if log:
        session.logger.info(f"Showing SNFG symbols for {total_shown} carbohydrate residues")
    return total_shown


def hide_snfg(session, structures=None, log=True):
    """
    Hide SNFG symbols and restore atom visibility.

    Parameters
    ----------
    session : Session
        ChimeraX session.
    structures : list of AtomicStructure, optional
        Structures to process. If None, use all open structures.
    log : bool
        If True (default), log a summary message to the session logger.

    Returns the number of symbols hidden.
    """
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)

    mgr = _manager(session)
    total_hidden = 0

    for structure in structures:
        depiction = mgr.depiction(structure)
        if depiction is not None:
            total_hidden += len(depiction.sizes)
            depiction.restore_atoms()
            mgr.remove(structure)

    if log:
        session.logger.info(f"Hidden SNFG symbols for {total_hidden} carbohydrate residues")
    return total_hidden


def snfg_command(session, action='show', structures=None, size=DEFAULT_SIZE,
                 atoms=True, log=True):
    """
    Command handler for 'snfg' command.

    Returns the number of symbols shown or hidden.
    """
    if action == 'show':
        return show_snfg(session, structures, size, show_atoms=atoms, log=log)
    elif action == 'hide':
        return hide_snfg(session, structures, log=log)
    else:
        from chimerax.core.errors import UserError
        raise UserError(f"Unknown action '{action}'. Use 'show' or 'hide'.")


def register_command(logger):
    """Register the snfg command."""
    from chimerax.core.commands import CmdDesc, register, EnumOf, FloatArg, BoolArg
    from chimerax.atomic import AtomicStructuresArg

    desc = CmdDesc(
        optional=[('action', EnumOf(['show', 'hide']))],
        keyword=[
            ('structures', AtomicStructuresArg),
            ('size', FloatArg),
            ('atoms', BoolArg),
            ('log', BoolArg),
        ],
        synopsis='Show or hide SNFG glycan symbols'
    )
    register('snfg', desc, snfg_command, logger=logger)


def _glycan_selector(session, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = sugar_residues(m).atoms
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)


def register_selector(logger):
    """Register the glycan selector."""
    from chimerax.core.commands import register_selector
    register_selector("glycan", _glycan_selector, logger, desc="glycan residues")
