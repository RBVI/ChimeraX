from . import io
from . import models

CATEGORY = io.STRUCTURE


class StructureModel(models.Model):
    """Commom base class for atomic structures"""

    STRUCTURE_STATE_VERSION = 0

    def take_snapshot(self, session, flags):
        data = {}
        return [self.STRUCTURE_STATE_VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.STRUCTURE_STATE_VERSION or len(data) > 0:
            raise RuntimeError("Unexpected version or data")

    def reset_state(self):
        pass

    def make_drawing(self):
        atom_blob = self.mol_blob.atoms
        coords = atom_blob.coords
        radii = atom_blob.radii
        atom_blob.colors = element_colors(atom_blob.element_numbers)
        colors = atom_blob.colors

        # TODO: fill in drawing
        self.create_atom_spheres(coords, radii, colors)

    def create_atom_spheres(self, coords, radii, colors):
        if not hasattr(self, '_atoms_drawing'):
            self._atoms_drawing = self.new_drawing('atoms')
        p = self._atoms_drawing

        # Set instanced sphere triangulation
        from .geometry import icosahedron
        va, ta = icosahedron.icosahedron_geometry()
        from numpy import int32
        p.geometry = va, ta.astype(int32)
        p.normals = va

        # Set instanced sphere center position and radius
        n = len(coords)
        from numpy import empty, float32
        xyzr = empty((n, 4), float32)
        xyzr[:, :3] = coords
        xyzr[:, 3] = radii
        from .geometry import place
        p.positions = place.Places(shift_and_scale=xyzr)

        # Set atom colors
        p.colors = colors

element_rgba_256 = None
def element_colors(element_numbers):
    global element_rgba_256
    if element_rgba_256 is None:
        from numpy import empty, uint8
        element_rgba_256 = ec = empty((256, 4), uint8)
        ec[:, :3] = 180
        ec[:, 3] = 255
        ec[6, :] = (255, 255, 255, 255)     # H
        ec[6, :] = (144, 144, 144, 255)     # C
        ec[7, :] = (48, 80, 248, 255)       # N
        ec[8, :] = (255, 13, 13, 255)       # O
        ec[15, :] = (255, 128, 0, 255)      # P
        ec[16, :] = (255, 255, 48, 255)     # S
    colors = element_rgba_256[element_numbers]
    return colors
