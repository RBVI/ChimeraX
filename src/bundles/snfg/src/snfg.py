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
sugar ring centroids, following the SNFG standard.
"""

import numpy as np
from chimerax.core.models import Model, Surface
from chimerax.atomic import AtomicShapeDrawing, AtomicShapeInfo
from chimerax.surface import calculate_vertex_normals

from .definitions import (
    COLORS, MONOSACCHARIDE_SYMBOLS, PDB_TO_SNFG, DEFAULT_SIZE
)
from .shapes import get_shape_geometry


class SNFGShapesDrawing(AtomicShapeDrawing):
    """Drawing containing all SNFG glycan symbols with selection support.

    Each glycan symbol is stored as a shape with associated residue atoms,
    enabling bidirectional selection: clicking a symbol selects its atoms,
    and selecting atoms highlights the symbol.
    """

    SESSION_VERSION = 1

    def __init__(self, name="SNFG shapes"):
        super().__init__(name)
        self._residue_to_shape_index = {}  # residue → shape index
        self._shape_residues = []  # ordered list parallel to self._shapes
        self._shape_centroids = []  # centroids for connection drawing

    def add_residue_shape(self, residue, shape_type, color_name, size=DEFAULT_SIZE):
        """Add a glycan symbol shape for a residue.

        Parameters
        ----------
        residue : Residue
            The carbohydrate residue to visualize.
        shape_type : str
            SNFG shape type (sphere, cube, diamond, etc.).
        color_name : str
            SNFG color name.
        size : float
            Size of the symbol in Angstroms.

        Returns
        -------
        int or None
            Shape index, or None if residue has no ring centroid.
        """
        if residue in self._residue_to_shape_index:
            return self._residue_to_shape_index[residue]

        # Get geometry (shapes are centered at origin)
        vertices, triangles = get_shape_geometry(shape_type, size)

        # Get ring centroid
        centroid = _ring_centroid(residue)
        if centroid is None:
            return None

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

        # Add shape with residue's atoms for selection support
        shape_index = len(self._shapes)
        self.add_shape(vertices, normals, triangles, color,
                      atoms=residue.atoms, description=description)

        # Track residue → shape mapping
        self._residue_to_shape_index[residue] = shape_index
        self._shape_residues.append(residue)
        self._shape_centroids.append(centroid)

        return shape_index

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

    def take_snapshot(self, session, flags):
        """Save session state."""
        data = super().take_snapshot(session, flags)
        data['snfg_version'] = SNFGShapesDrawing.SESSION_VERSION
        data['shape_residues'] = self._shape_residues
        data['shape_centroids'] = self._shape_centroids
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        """Restore from session."""
        d = super().restore_snapshot(session, data)
        d._shape_residues = data.get('shape_residues', [])
        d._shape_centroids = data.get('shape_centroids', [])
        # Rebuild residue → shape index mapping
        d._residue_to_shape_index = {
            res: i for i, res in enumerate(d._shape_residues)
        }
        return d


class SNFGConnectionsDrawing(Surface):
    """Drawing for the connection lines between SNFG symbols."""

    def __init__(self, session, name="SNFG connections"):
        super().__init__(name, session)
        self.color = (180, 180, 180, 255)  # Gray connections


class SNFGModel(Model):
    """Container model for all SNFG drawings associated with a structure."""

    def __init__(self, session, structure):
        super().__init__("SNFG symbols", session)
        self.structure = structure
        self._hidden_residues = set()  # Track which residues we hid
        self._connections_drawing = None

        # Create the shapes drawing for all glycan symbols
        self._shapes_drawing = SNFGShapesDrawing()
        self.add([self._shapes_drawing])

    def add_residue(self, residue, shape_type, color_name, size=DEFAULT_SIZE):
        """Add an SNFG symbol for a residue."""
        return self._shapes_drawing.add_residue_shape(
            residue, shape_type, color_name, size
        )

    def add_residues_batch(self, residue_info_list):
        """Add multiple SNFG symbols efficiently.

        Parameters
        ----------
        residue_info_list : list of (residue, shape_type, color_name, size) tuples
        """
        return self._shapes_drawing.add_residues_batch(residue_info_list)

    def remove_residue(self, residue):
        """Remove the SNFG symbol for a residue.

        Note: This requires rebuilding all shapes. For bulk removal,
        consider clearing and re-adding instead.
        """
        if not self._shapes_drawing.has_residue(residue):
            return

        # Collect info for all residues except the one to remove
        residues_to_keep = []
        for r in self._shapes_drawing.residues():
            if r != residue and not r.deleted:
                symbol = identify_sugar(r)
                if symbol:
                    shape_type, color_name = symbol
                    residues_to_keep.append((r, shape_type, color_name, DEFAULT_SIZE))

        # Clear and rebuild
        self._shapes_drawing.clear_shapes()
        if residues_to_keep:
            self._shapes_drawing.add_residues_batch(residues_to_keep)

    def has_residue(self, residue):
        """Check if a residue has an SNFG symbol."""
        return self._shapes_drawing.has_residue(residue)

    def hide_atoms(self):
        """Hide atoms for all residues with SNFG symbols."""
        for residue in self._shapes_drawing.residues():
            if residue.deleted:
                continue
            atoms = residue.atoms
            # Only hide if currently displayed
            displayed = atoms.displays
            if displayed.any():
                self._hidden_residues.add(residue)
                atoms.displays = False

    def show_atoms(self):
        """Restore visibility for atoms we previously hid."""
        for residue in self._hidden_residues:
            if residue.deleted:
                continue
            residue.atoms.displays = True
        self._hidden_residues.clear()

    def update_connections(self):
        """Draw connections between linked sugar residues."""
        # Remove old connections
        if self._connections_drawing is not None:
            self._connections_drawing.delete()
            self._connections_drawing = None

        # Find connections between sugar residues
        connections = []
        sugar_residues = set(self._shapes_drawing.residues())

        for residue in sugar_residues:
            if residue.deleted:
                continue
            centroid = self._shapes_drawing.get_centroid(residue)
            if centroid is None:
                continue

            # Look for bonds to atoms in other sugar residues
            for atom in residue.atoms:
                for neighbor in atom.neighbors:
                    other_res = neighbor.residue
                    if other_res != residue and other_res in sugar_residues:
                        other_centroid = self._shapes_drawing.get_centroid(other_res)
                        if other_centroid is not None:
                            # Add connection (use frozenset to avoid duplicates)
                            pair = frozenset([residue, other_res])
                            conn = (centroid, other_centroid)
                            connections.append((pair, conn))

        # Remove duplicates
        seen = set()
        unique_connections = []
        for pair, conn in connections:
            if pair not in seen:
                seen.add(pair)
                unique_connections.append(conn)

        if not unique_connections:
            return

        # Create cylinder geometry for connections
        self._connections_drawing = SNFGConnectionsDrawing(self.session)
        vertices, triangles = _cylinder_connections(unique_connections, radius=0.2)
        if len(vertices) > 0:
            normals = calculate_vertex_normals(vertices, triangles)
            self._connections_drawing.set_geometry(vertices, normals, triangles)
            self.add([self._connections_drawing])


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
    res_name = residue.name.upper()

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


def find_sugar_residues(structures):
    """
    Find all carbohydrate residues in the given structures.

    Returns a list of (residue, shape_type, color_name) tuples.
    """
    results = []

    for structure in structures:
        for residue in structure.residues:
            symbol = identify_sugar(residue)
            if symbol is not None:
                shape_type, color_name = symbol
                results.append((residue, shape_type, color_name))

    return results


def get_snfg_model(session, structure, create=True):
    """
    Get or create the SNFG model for a structure.
    """
    # Look for existing SNFG model
    for model in structure.child_models():
        if isinstance(model, SNFGModel) and model.structure == structure:
            return model

    if not create:
        return None

    # Create new SNFG model
    snfg_model = SNFGModel(session, structure)
    structure.add([snfg_model])
    return snfg_model


def show_snfg(session, structures=None, size=DEFAULT_SIZE, replace=True,
               show_atoms=False):
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
    """
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)

    total_shown = 0

    for structure in structures:
        snfg_model = get_snfg_model(session, structure, create=True)

        if replace:
            # Restore any previously hidden atoms before clearing
            snfg_model.show_atoms()
            # Clear existing shapes
            snfg_model._shapes_drawing.clear_shapes()
            # Remove connections drawing
            if snfg_model._connections_drawing is not None:
                snfg_model._connections_drawing.delete()
                snfg_model._connections_drawing = None

        sugars = find_sugar_residues([structure])

        # Collect residues to add (filter out those already present)
        residue_info = []
        for residue, shape_type, color_name in sugars:
            if not snfg_model.has_residue(residue):
                residue_info.append((residue, shape_type, color_name, size))

        # Batch add all residues for efficiency
        if residue_info:
            total_shown += snfg_model.add_residues_batch(residue_info)

        # Hide or show atoms based on parameter
        if show_atoms:
            snfg_model.show_atoms()
        else:
            snfg_model.hide_atoms()
            snfg_model.update_connections()

    session.logger.info(f"Showing SNFG symbols for {total_shown} carbohydrate residues")


def hide_snfg(session, structures=None):
    """
    Hide SNFG symbols and restore atom visibility.

    Parameters
    ----------
    session : Session
        ChimeraX session.
    structures : list of AtomicStructure, optional
        Structures to process. If None, use all open structures.
    """
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)

    total_hidden = 0

    for structure in structures:
        snfg_model = get_snfg_model(session, structure, create=False)
        if snfg_model is not None:
            total_hidden += len(snfg_model._shapes_drawing._shape_residues)
            # Restore atom visibility before deleting
            snfg_model.show_atoms()
            snfg_model.delete()

    session.logger.info(f"Hidden SNFG symbols for {total_hidden} carbohydrate residues")


def snfg_command(session, action='show', structures=None, size=DEFAULT_SIZE,
                 atoms=True):
    """
    Command handler for 'snfg' command.
    """
    if action == 'show':
        show_snfg(session, structures, size, show_atoms=atoms)
    elif action == 'hide':
        hide_snfg(session, structures)
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
        ],
        synopsis='Show or hide SNFG glycan symbols'
    )
    register('snfg', desc, snfg_command, logger=logger)
