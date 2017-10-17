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
        self._visible_pbonds = None		# Drawing only contains visible pseudobonds
        self._visible_pbond_atoms = None
        self._dashes = 9
        self._global_group = (s is None)
        self._handlers = []
        if s:
            s.add([self])            # Add pseudobond group as child model of structure
        
    def delete(self):
        if self._global_group:
            pbm = self.session.pb_manager
            pbm._delete_group(self)
        else:
            s = self._structure
            if s and not s.deleted:
                s._delete_pseudobond_group(self)
            self._structure = None
        Model.delete(self)
        self._pbond_drawing = None

    def added_to_session(self, session):
        if self.structure:
            return	# Structure tells pseudobond group when to update

        # For global pseudobond groups:
        # Detect when atoms moved so pseudobonds must be redrawn.
        # TODO: Update only when atoms move or are shown hidden, not when anything shown or hidden.
        # TODO: Only update on selection change if pseudobond atoms selection changed.
        from ..selection import SELECTION_CHANGED
        t = session.triggers
        self._handlers = [
            t.add_handler('graphics update',self._update_graphics_if_needed),
            t.add_handler('shape changed', lambda *args, s=self: s._update_graphics()),
            t.add_handler(SELECTION_CHANGED, lambda *args, s=self: s._update_graphics())
        ]

    def removed_from_session(self, session):
        t = session.triggers
        h = self._handlers
        while h:
            t.remove_handler(h.pop())
        self._handlers = []

    # Atom specifier API
    def atomspec_has_pseudobonds(self):
        return True

    def atomspec_pseudobonds(self):
        return self.pseudobonds

    def _set_selected(self, sel):
        self.pseudobonds.selected = sel
        Model.set_selected(self, sel)
    selected = property(Model.get_selected, _set_selected)

    def selected_items(self, itype):
        if itype == 'pseudobonds':
            pbonds = self.pseudobonds
            if pbonds.num_selected > 0:
                return [pbonds.filter(pbonds.selected)]
        return []

    def any_part_selected(self):
        if self.pseudobonds.num_selected > 0:
            return True
        for c in self.child_models():
            if c.any_part_selected():
                return True
        return False

    def selection_promotion(self):
        pbonds = self.pseudobonds
        n = pbonds.num_selected
        if n == 0 or n == len(pbonds):
            return None
        return PromotePseudobondSelection(self, pbonds.selected)

    def clear_selection(self):
        self.selected = False
        self.pseudobonds.selected = False

    def _get_dashes(self):
        return self._dashes
    def _set_dashes(self, n):
        if n == self._dashes:
            return
        self._dashes = n
        pb = self._pbond_drawing
        if pb:
            self.remove_drawing(pb)
            self._pbond_drawing = None
            self._graphics_changed |= self._SHAPE_CHANGE
            self.redraw_needed(shape_changed = True)
        self.session.change_tracker.add_modified(self, "dashes changed")

    dashes = property(_get_dashes, _set_dashes)

    def _get_name(self):
        return self.category

    def _set_name(self, name):
        if name != self.category:
            self.change_category(name)
        # allow Model to fire 'name changed' trigger
        Model.name.fset(self, name)
    name = property(_get_name, _set_name)

    def _update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc:
            self._graphics_changed = 0
            self._update_graphics(gc)
            self.redraw_needed(shape_changed = (gc & self._SHAPE_CHANGE),
                               selection_changed = (gc & self._SELECT_CHANGE))

    def _update_graphics(self, changes = PseudobondGroupData._ALL_CHANGE):

        d = self._pbond_drawing
        if d is None:
            d = self.new_drawing('pbonds')
            self._pbond_drawing = d
            va, na, ta = _pseudobond_geometry(self._dashes//2)
            d.vertices = va
            d.normals = na
            d.triangles = ta
            changes = self._ALL_CHANGE
        elif self.num_pseudobonds == 0:
            self.remove_drawing(d)
            self._pbond_drawing = None
            return

        if changes & (self._ADDDEL_CHANGE | self._DISPLAY_CHANGE):
            changes = self._ALL_CHANGE
            
        if changes & self._DISPLAY_CHANGE or self._visible_pbonds is None:
            vpb = self._shown_pbonds(self.pseudobonds)
            self._visible_pbonds = vpb
            self._visible_pbond_atoms = vpb.atoms
        
        pbonds = self._visible_pbonds
        bond_atoms = self._visible_pbond_atoms

        if changes & self._SHAPE_CHANGE:
            d.positions = self._update_positions(pbonds, bond_atoms)
            
        if changes & self._COLOR_CHANGE:
            d.colors = pbonds.half_colors
            
        if changes & self._SELECT_CHANGE:
            from . import structure as s
            d.selected_positions = s._selected_bond_cylinders(pbonds)

    def _update_positions(self, pbonds, bond_atoms):
        ba1, ba2 = bond_atoms
        if self._global_group:
            to_pbg = self.scene_position.inverse()
            axyz0, axyz1 = to_pbg*ba1.scene_coords, to_pbg*ba2.scene_coords
        else:
            axyz0, axyz1 = ba1.coords, ba2.coords
        from . import structure as s
        return s._halfbond_cylinder_placements(axyz0, axyz1, pbonds.radii)

    def _shown_pbonds(self, pbonds):
        # Check if models containing end-point have displayed structures.
        dpb = pbonds.showns
        if self.structure is None:
            ba1, ba2 = pbonds.atoms
            for hs in (hidden_structures(ba1.structures), hidden_structures(ba2.structures)):
                if hs is not None:
                    from numpy import logical_and
                    logical_and(dpb, ~hs, dpb)
        return pbonds[dpb]

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None
        from . import structure
        b,f = structure._bond_intercept(self.pseudobonds, mxyz1, mxyz2)
        p = structure.PickedPseudobond(b,f) if b else None
        return p

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []

        picks = []
        from ..geometry import transform_planes
        for p in self.positions:
            pplanes = transform_planes(p, planes)
            picks.extend(self._pseudobonds_planes_pick(pplanes))

        return picks

    def _pseudobonds_planes_pick(self, planes):
        from .structure import _bonds_planes_pick, PickedPseudobonds
        pmask = _bonds_planes_pick(self._pbond_drawing, planes)
        if pmask.sum() == 0:
            return []
        bonds = self._visible_pbonds.filter(pmask)
        p = PickedPseudobonds(bonds)
        return [p]

    def take_snapshot(self, session, flags):
        data = {
            'version': 1,
            'category': self.category,
            'dashes': self._dashes,
            'model state': Model.take_snapshot(self, session, flags),
            'structure': self.structure,
        }
        if self._global_group:
            # Make the global manager restore before we do
            data['mgr'] = session.pb_manager

        return data

    @staticmethod
    def restore_snapshot(session, data):
        if data['structure'] is not None:
            grp = data['structure'].pseudobond_group(data['category'], create_type=None)
        else:
            grp = session.pb_manager.get_group(data['category'], create=False)
        if 'model state' in data:
            Model.set_state_from_snapshot(grp, session, data['model state'])
        grp._dashes = data['dashes']
        return grp

    def reset_state(self, session):
        pass

# -----------------------------------------------------------------------------
#
def selected_pseudobonds(session):
    '''All selected bonds in all structures as an :class:`.Bonds` collection.'''
    blist = []
    for m in session.models.list(type = PseudobondGroup):
        pbonds = m.pseudobonds
        pbsel = pbonds.selected
        if len(pbsel) > 0:
            blist.append(pbonds[pbsel])
    from .molarray import concatenate, Pseudobonds
    pbonds = concatenate(blist, Pseudobonds)
    return pbonds

# -----------------------------------------------------------------------------
#
from ..selection import SelectionPromotion
class PromotePseudobondSelection(SelectionPromotion):
    def __init__(self, pbgroup, prev_pbond_sel_mask):
        level = 1001
        SelectionPromotion.__init__(self, level)
        self._pbgroup = pbgroup
        self._prev_pbond_sel_mask = prev_pbond_sel_mask
    def promote(self):
        pbonds = self._pbgroup.pseudobonds
        pbonds.selected = True
    def demote(self):
        pbonds = self._pbgroup.pseudobonds
        pbonds.selected = self._prev_pbond_sel_mask

# -----------------------------------------------------------------------------
#
def all_pseudobond_groups(models):
    return [m for m in models.list() if isinstance(m, PseudobondGroup)]

# -----------------------------------------------------------------------------
#
def interatom_pseudobonds(atoms, group_name = None):
    structures = atoms.unique_structures
    if len(structures) == 0:
        from . import Pseudobonds
        return Pseudobonds()
    # Inter-model pseudobond groups
    session = structures[0].session
    pbgs = set(session.models.list(type = PseudobondGroup))
    # Intra-model pseudobond groups
    for m in structures:
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
def hidden_structures(structures):
    us = structures.unique()
    vis = set(s for s in us if s.visible)
    if len(vis) == len(us):
        return None
    from numpy import array, bool
    hs = array([(s not in vis) for s in structures], bool)
    return hs

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9):
    from .. import surface
    return surface.dashed_cylinder_geometry(segments, height = 0.5)
