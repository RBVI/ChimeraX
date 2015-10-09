# vi: set expandtab shiftwidth=4 softtabstop=4:
from .molobject import PseudobondGroupData
from ..models import Model
class PseudobondGroup(PseudobondGroupData, Model):
    """
    A pseudobond group is a named group of :class:`.Pseudobond` objects
    such as distances that depict lines between atoms with distance labels.
    Pseudobond groups are managed in C++ code which is made accessible
    via the :class:`.PseudobondGroupData` base class.
    """

    def __init__(self, pbg_pointer):

        PseudobondGroupData.__init__(self, pbg_pointer)
        Model.__init__(self, self.category)
        self._pbond_drawing = None

        self._update_graphics()

    def delete(self):
        Model.delete(self)
        self._pbond_drawing = None
        PseudobondGroupData.delete(self)

    def added_to_session(self, session):
        # Detect when atoms moved so pseudobonds must be redrawn.
        # TODO: Update only when atoms move or are shown hidden, not when anything shown or hidden.
        self.handlers = [
            session.triggers.add_handler('graphics update',self._update_graphics_if_needed),
            session.triggers.add_handler('shape changed', self.update_graphics)
        ]

    def removed_from_session(self, session):
        while self.handlers:
            session.delete_handler(self.handlers.pop())

    def _update_graphics_if_needed(self, *_):
        c, s, se = self._gc_color, self._gc_shape, self._gc_select
        if c or s or se:
            self._gc_color = self._gc_shape = self._gc_select = False
            self._update_graphics()
            self.redraw_needed(shape_changed = s, selection_changed = se)

    def _update_graphics(self, *_):

        pbonds = self.pseudobonds
        d = self._pbond_drawing
        if len(pbonds) == 0:
            if d:
                d.delete()
                self._pbond_drawing = None
            return

        from . import structure
        if d is None:
            self._pbond_drawing = d = self.new_drawing('pbonds')
            va, na, ta = structure._pseudobond_geometry()
            d.vertices = va
            d.normals = na
            d.triangles = ta

        ba1, ba2 = pbonds.atoms
        to_pbg = self.scene_position.inverse()
        axyz0, axyz1 = to_pbg*ba1.scene_coords, to_pbg*ba2.scene_coords
        d.positions = structure._halfbond_cylinder_placements(axyz0, axyz1, pbonds.radii)
        d.display_positions = structure._shown_bond_cylinders(pbonds)
        d.colors = pbonds.half_colors

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

def all_pseudobond_groups(models):
    return [m for m in models.list() if isinstance(m, PseudobondGroup)]

def interatom_pseudobonds(atoms, session):
    # Inter-model pseudobond groups
    pbgs = session.models.list(PseudobondGroup)
    # Intra-model pseudobond groups
    for m in atoms.unique_structures:
        pbgs.extend(tuple(m.pbg_map.values()))
    # Collect bonds
    from . import Pseudobonds
    ipbonds = Pseudobonds()
    for pbg in pbgs:
        pbonds = pbg.pseudobonds
        a1, a2 = pbonds.atoms
        ipb = pbonds.filter(a1.mask(atoms) & a2.mask(atoms))
        if ipb:
            ipbonds |= ipb
    return ipbonds
