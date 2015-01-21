from . import io
from . import models
from .graphics.drawing import Drawing

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
        atom_blob, bond_list = self.mol_blob.atoms_bonds
        coords = atom_blob.coords
        element_numbers = atom_blob.element_numbers

        # TODO: fill in drawing
