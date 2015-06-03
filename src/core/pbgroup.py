from .molecule import CPseudoBondGroup
from .models import Model
class PseudoBondGroup(CPseudoBondGroup, Model):
    """Pseudobond group model"""

    def __init__(self, name):

        CPseudoBondGroup.__init__(self, name)
        Model.__init__(self, name)
        self.pseudobond_radius = 0.05

        self.update_graphics()

    def update_graphics(self):

        from . import structure
        if self.vertices is None:
            va, na, ta = structure.pseudobond_geometry()
            self.vertices = va
            self.normals = na
            self.triangles = ta

        pbonds = self.pseudobonds
        bond_atoms = pbonds.atoms
        radii = pbonds.radii
        bond_colors = pbonds.colors
        half_bond_coloring = pbonds.halfbonds
        self.positions = structure.bond_cylinder_placements(bond_atoms, radii, half_bond_coloring)
        self.display_positions = self.shown_bond_cylinders(bond_atoms, half_bond_coloring)
        self.set_bond_colors(bond_atoms, bond_colors, half_bond_coloring)

    def set_bond_colors(self, bond_atoms, bond_colors, half_bond_coloring):
        if half_bond_coloring.any():
            bc0,bc1 = bond_atoms[0].colors, bond_atoms[1].colors
            from numpy import concatenate
            c = concatenate((bc0,bc1))
        else:
            c = bond_colors
        self.colors = c

    def shown_bond_cylinders(self, bond_atoms, half_bond_coloring):
        sb = bond_atoms[0].displays & bond_atoms[1].displays  # Show bond if both atoms shown
        if half_bond_coloring.any():
            sb2 = numpy.concatenate((sb,sb))
            return sb2
        return sb

    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        data = {}
        return [self.STRUCTURE_STATE_VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.STRUCTURE_STATE_VERSION or len(data) > 0:
            raise RestoreError("Unexpected version or data")

    def reset_state(self):
        pass
