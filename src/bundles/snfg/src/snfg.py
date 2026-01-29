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
from chimerax.surface import calculate_vertex_normals

from .definitions import (
    COLORS, MONOSACCHARIDE_SYMBOLS, PDB_TO_SNFG, DEFAULT_SIZE
)
from .shapes import get_shape_geometry


class SNFGDrawing(Surface):
    """A surface model representing an SNFG symbol for a single sugar residue."""

    SESSION_SAVE_DRAWING = True

    def __init__(self, session, residue, shape_type, color_name, size=DEFAULT_SIZE):
        name = f"SNFG {residue.name}"
        super().__init__(name, session)

        self.residue = residue
        self.shape_type = shape_type
        self.color_name = color_name

        # Get geometry (shapes are centered at origin with Y as primary axis)
        vertices, triangles = get_shape_geometry(shape_type, size)

        # Get ring centroid and linkage direction
        centroid = _ring_centroid(residue)
        self.centroid = centroid

        if centroid is not None:
            # Find direction to linked sugar for orientation
            linkage_dir = _find_linkage_direction(residue, centroid)

            if linkage_dir is not None:
                # Rotate shape so Y axis points along linkage direction
                rotation = _rotation_to_align_y(linkage_dir)
                vertices = np.dot(vertices, rotation.T)

            # Translate to centroid
            vertices = vertices + centroid

        normals = calculate_vertex_normals(vertices, triangles)
        self.set_geometry(vertices, normals, triangles)

        # Set color
        rgb = COLORS.get(color_name, (190, 190, 190))
        self.color = (*rgb, 255)

        self.clip_cap = True


class SNFGConnectionsDrawing(Surface):
    """Drawing for the connection lines between SNFG symbols."""

    def __init__(self, session, name="SNFG connections"):
        super().__init__(name, session)
        self.color = (180, 180, 180, 255)  # Gray connections


class SNFGModel(Model):
    """Container model for all SNFG drawings associated with a structure."""

    def __init__(self, session, structure):
        super().__init__(f"SNFG symbols", session)
        self.structure = structure
        self._residue_to_drawing = {}
        self._hidden_residues = set()  # Track which residues we hid
        self._connections_drawing = None

    def add_residue(self, residue, shape_type, color_name, size=DEFAULT_SIZE):
        """Add an SNFG symbol for a residue."""
        if residue in self._residue_to_drawing:
            return self._residue_to_drawing[residue]

        drawing = SNFGDrawing(self.session, residue, shape_type, color_name, size)
        self.add([drawing])
        self._residue_to_drawing[residue] = drawing
        return drawing

    def remove_residue(self, residue):
        """Remove the SNFG symbol for a residue."""
        if residue in self._residue_to_drawing:
            drawing = self._residue_to_drawing.pop(residue)
            drawing.delete()

    def has_residue(self, residue):
        """Check if a residue has an SNFG symbol."""
        return residue in self._residue_to_drawing

    def hide_atoms(self):
        """Hide atoms for all residues with SNFG symbols."""
        for residue in self._residue_to_drawing:
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
        sugar_residues = set(self._residue_to_drawing.keys())

        for residue in sugar_residues:
            if residue.deleted:
                continue
            drawing = self._residue_to_drawing[residue]
            if drawing.centroid is None:
                continue

            # Look for bonds to atoms in other sugar residues
            for atom in residue.atoms:
                for neighbor in atom.neighbors:
                    other_res = neighbor.residue
                    if other_res != residue and other_res in sugar_residues:
                        other_drawing = self._residue_to_drawing[other_res]
                        if other_drawing.centroid is not None:
                            # Add connection (use frozenset to avoid duplicates)
                            pair = frozenset([residue, other_res])
                            conn = (drawing.centroid, other_drawing.centroid)
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


def _ring_centroid(residue):
    """
    Calculate the centroid of the sugar ring in a residue.

    Looks for pyranose (6-membered) or furanose (5-membered) rings.
    Returns the centroid as a numpy array, or None if no ring found.
    """
    # Try to find rings in the residue
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
        # Fall back to residue center
        coords = atoms.coords
        if len(coords) > 0:
            return np.mean(coords, axis=0)
        return None

    # Use the first suitable ring (prefer 6-membered pyranose)
    pyranose_rings = [r for r in rings if r.size == 6]
    ring = pyranose_rings[0] if pyranose_rings else rings[0]

    # Calculate centroid
    ring_coords = ring.atoms.coords
    return np.mean(ring_coords, axis=0)


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


def _rotation_to_align_y(target_direction):
    """
    Compute a rotation matrix that aligns the Y axis with the target direction.

    Parameters
    ----------
    target_direction : ndarray
        Unit vector for the desired Y axis direction.

    Returns
    -------
    rotation : ndarray
        3x3 rotation matrix.
    """
    target = np.asarray(target_direction, dtype=np.float64)
    target = target / np.linalg.norm(target)

    y_axis = np.array([0.0, 1.0, 0.0])

    # Check if already aligned (or anti-aligned)
    dot = np.dot(y_axis, target)

    if dot > 0.9999:
        # Already aligned
        return np.eye(3)
    elif dot < -0.9999:
        # Opposite direction - rotate 180 degrees around X or Z
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)

    # General case: use Rodrigues' rotation formula
    # Rotation axis is cross product of y_axis and target
    axis = np.cross(y_axis, target)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Rodrigues' formula: R = I + sin(a)*K + (1-cos(a))*K^2
    # where K is the skew-symmetric cross-product matrix of the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation


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
            # Remove existing drawings
            for drawing in list(snfg_model.child_drawings()):
                drawing.delete()
            snfg_model._residue_to_drawing.clear()
            snfg_model._connections_drawing = None

        sugars = find_sugar_residues([structure])

        for residue, shape_type, color_name in sugars:
            if not snfg_model.has_residue(residue):
                snfg_model.add_residue(residue, shape_type, color_name, size)
                total_shown += 1

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
            total_hidden += len(snfg_model._residue_to_drawing)
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
