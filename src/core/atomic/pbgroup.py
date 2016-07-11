# vim: set expandtab shiftwidth=4 softtabstop=4:
from .molobject import PseudobondGroupData
from ..models import Model
class PseudobondGroup(PseudobondGroupData, Model):
    """
    A pseudobond group is a named group of :class:`.Pseudobond` objects
    such as distances that depict lines between atoms with distance labels.
    Pseudobond groups are managed in C++ code which is made accessible
    via the :class:`.PseudobondGroupData` base class.
    """

    def __init__(self, pbg_pointer, *, session=None):

        PseudobondGroupData.__init__(self, pbg_pointer)
        if session is None:
            session = self.structure.session
        Model.__init__(self, self.category, session)
        self._pbond_drawing = None
        self._dashes = 9

    def delete(self):
        pbm = self.session.pb_manager
        Model.delete(self)
        self._pbond_drawing = None
        pbm.delete_group(self)

    def added_to_session(self, session):
        # Detect when atoms moved so pseudobonds must be redrawn.
        # TODO: Update only when atoms move or are shown hidden, not when anything shown or hidden.
        t = session.triggers
        self.handlers = [
            t.add_handler('graphics update',self._update_graphics_if_needed),
            t.add_handler('shape changed', self._update_graphics)
        ]

    def removed_from_session(self, session):
        t = session.triggers
        while self.handlers:
            t.delete_handler(self.handlers.pop())
        self.handlers = []

    def _get_dashes(self):
        return self._dashes
    def _set_dashes(self, n):
        self._dashes = n
        pb = self._pbond_drawing
        if pb:
            owner = self.structure if self.structure else self
            owner.remove_drawing(pb)
            self._pbond_drawing = None
            self._graphics_changed |= self._SHAPE_CHANGE
            owner.redraw_needed(shape_changed = True)
    dashes = property(_get_dashes, _set_dashes)

    def _update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc:
            self._graphics_changed = 0
            self._update_graphics()
            self.redraw_needed(shape_changed = (gc & self._SHAPE_CHANGE),
                               selection_changed = (gc & self._SELECT_CHANGE))

    def _update_graphics(self):

        pbonds = self.pseudobonds
        d = self._pbond_drawing
        if len(pbonds) == 0:
            if d:
                owner = self.structure if self.structure else self
                owner.remove_drawing(d)
                self._pbond_drawing = None
            return

        if d is None:
            owner = self.structure
            d = owner.new_drawing(self.category) if owner else self.new_drawing('pbonds')
            self._pbond_drawing = d
            va, na, ta = _pseudobond_geometry(self._dashes//2)
            d.vertices = va
            d.normals = na
            d.triangles = ta

        ba1, ba2 = bond_atoms = pbonds.atoms
        if self.structure:
            axyz0, axyz1 = ba1.coords, ba2.coords
        else:
            to_pbg = self.scene_position.inverse()
            axyz0, axyz1 = to_pbg*ba1.scene_coords, to_pbg*ba2.scene_coords
        from . import structure as s
        d.positions = s._halfbond_cylinder_placements(axyz0, axyz1, pbonds.radii)
        d.display_positions = s._shown_bond_cylinders(pbonds)
        d.colors = pbonds.half_colors
        d.selected_positions = s._selected_bond_cylinders(bond_atoms)

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and hasattr(self, exclude)):
            return None
        from . import structure
        b,f = structure._bond_intercept(self.pseudobonds, mxyz1, mxyz2)
        p = structure.PickedPseudobond(b,f) if b else None
        return p

    def take_snapshot(self, session, flags):
        data = {
            'version': 1,
            'mgr': session.pb_manager, # so that the global manager gets restored before we do
            'category': self.category,
            'dashes': self._dashes
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        if data['version'] != 1:
            raise RestoreError("Unexpected pb group session version")
        if data['structure'] is not None:
            grp = data['structure'].pseudobond_group(data['category'], create_type=None)
        else:
            grp = session.pb_manager.get_group(data['category'], create=False)
        grp._dashes = data['dashes']
        return grp

    def reset_state(self, session):
        pass

def all_pseudobond_groups(models):
    return [m for m in models.list() if isinstance(m, PseudobondGroup)]

def interatom_pseudobonds(atoms, session, group_name = None):
    # Inter-model pseudobond groups
    pbgs = session.models.list(type = PseudobondGroup)
    # Intra-model pseudobond groups
    for m in atoms.unique_structures:
        pbgs.extend(m.pbg_map.values())
    # Collect bonds
    ipbonds = []
    for pbg in pbgs:
        if group_name is not None and pbg.category != group_name:
            continue
        pbonds = pbg.pseudobonds
        ipb = pbonds.filter(pbonds.between_atoms(atoms))
        print ('%s pbonds got %d' % (pbg.category, len(ipb)))
        if ipb:
            ipbonds.append(ipb)
    from . import Pseudobonds, concatenate
    ipb = concatenate(ipbonds, Pseudobonds)
    return ipb

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9):
    from .. import surface
    return surface.dashed_cylinder_geometry(segments)
