# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
        self._structure = s = self.structure	# Keep structure in case PseudobondGroupData deleted.
        if session is None:
            session = s.session
        Model.__init__(self, self.category, session)
        self._pbond_drawing = None
        self._dashes = 9
        self._global_group = (s is None)
        self._handlers = []
        if s:
            s.add([self])            # Add pseudobond group as child model of structure
        
    def delete(self):
        if self._global_group:
            pbm = self.session.pb_manager
            pbm.delete_group(self)
        else:
            s = self._structure
            if s and not s.deleted:
                s.delete_pseudobond_group(self)
            self._structure = None
        Model.delete(self)
        self._pbond_drawing = None

    def added_to_session(self, session):
        if self.structure:
            return	# Structure tells pseudobond group when to update

        # For global pseudobond groups:
        # Detect when atoms moved so pseudobonds must be redrawn.
        # TODO: Update only when atoms move or are shown hidden, not when anything shown or hidden.
        t = session.triggers
        self._handlers = [
            t.add_handler('graphics update',self._update_graphics_if_needed),
            t.add_handler('shape changed', lambda *args, s=self: s._update_graphics())
        ]

    def removed_from_session(self, session):
        t = session.triggers
        h = self._handlers
        while h:
            t.remove_handler(h.pop())
        self._handlers = []

    def _get_dashes(self):
        return self._dashes
    def _set_dashes(self, n):
        self._dashes = n
        pb = self._pbond_drawing
        if pb:
            self.remove_drawing(pb)
            self._pbond_drawing = None
            self._graphics_changed |= self._SHAPE_CHANGE
            self.redraw_needed(shape_changed = True)
    dashes = property(_get_dashes, _set_dashes)

    def _update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc:
            self._graphics_changed = 0
            self._update_graphics(gc)
            self.redraw_needed(shape_changed = (gc & self._SHAPE_CHANGE),
                               selection_changed = (gc & self._SELECT_CHANGE))

    def _update_graphics(self, changes = PseudobondGroupData._ALL_CHANGE):

        pbonds = self.pseudobonds
        d = self._pbond_drawing
        if len(pbonds) == 0:
            if d:
                self.remove_drawing(d)
                self._pbond_drawing = None
            return

        if d is None:
            d = self.new_drawing('pbonds')
            self._pbond_drawing = d
            va, na, ta = _pseudobond_geometry(self._dashes//2)
            d.vertices = va
            d.normals = na
            d.triangles = ta

        if changes & (self._SHAPE_CHANGE | self._SELECT_CHANGE):
            bond_atoms = pbonds.atoms
        if changes & self._SHAPE_CHANGE:
            ba1, ba2 = bond_atoms
            if self._global_group:
                to_pbg = self.scene_position.inverse()
                axyz0, axyz1 = to_pbg*ba1.scene_coords, to_pbg*ba2.scene_coords
            else:
                axyz0, axyz1 = ba1.coords, ba2.coords
            from . import structure as s
            d.positions = s._halfbond_cylinder_placements(axyz0, axyz1, pbonds.radii)
            d.display_positions = s._shown_bond_cylinders(pbonds)
        if changes & (self._COLOR_CHANGE | self._SHAPE_CHANGE):
            d.colors = pbonds.half_colors
        if changes & (self._SELECT_CHANGE | self._SHAPE_CHANGE):
            from . import structure as s
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
            'category': self.category,
            'dashes': self._dashes,
            'structure': self.structure,
        }
        if self._global_group:
            # Make the global manager restore before we do
            data['mgr'] = session.pb_manager

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
    pbgs = set(session.models.list(type = PseudobondGroup))
    # Intra-model pseudobond groups
    for m in atoms.unique_structures:
        for pbg in m.pbg_map.values():
            pbgs.add(pbg)
    # Collect bonds
    pbonds = [pbg.pseudobonds for pbg in pbgs
              if group_name is None or pbg.category == group_name]
    from . import Pseudobonds, concatenate
    pb = concatenate(pbonds, Pseudobonds)
    ipb = pb.filter(pb.between_atoms(atoms))
    return ipb

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9):
    from .. import surface
    return surface.dashed_cylinder_geometry(segments, height = 0.5)
