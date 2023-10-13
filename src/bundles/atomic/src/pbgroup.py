# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .molobject import PseudobondGroupData
from chimerax.core.models import Model
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
        Model.__init__(self, self.name, session)
        self._pbond_drawing = None
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
        # TODO: Update only when atoms move or are shown hidden,
        #       not when non-atomic models shown or hidden.
        # TODO: Need to update if parent of structure moves.
        t = session.triggers
        from . import get_triggers
        ta = get_triggers()
        def pbg_update(*args, self=self):
            self._update_graphics()
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        self._handlers = [
            (t, t.add_handler('graphics update', self._update_graphics_if_needed)),
            (ta, ta.add_handler('changes', pbg_update)),
            (t, t.add_handler(MODEL_DISPLAY_CHANGED, pbg_update)),
        ]

    def removed_from_session(self, session):
        th = self._handlers
        for t, h in th:
            t.remove_handler(h)
        th.clear()

    # Don't allow changing a pseudobond group position.
    # Pseudobonds are always drawn to their end-point atoms so the model position
    # should never be used.  But child models (e.g. labels) and things like picking
    # may unintentially use it.  So prevent it from being changed.
    def _dont_set_position(self, position):
        pass
    position = property(Model.position.fget, _dont_set_position)
    def _dont_set_positions(self, positions):
        pass
    positions = property(Model.positions.fget, _dont_set_positions)

    # Atom specifier API
    def atomspec_has_pseudobonds(self):
        return True

    def atomspec_pseudobonds(self):
        return self.pseudobonds

    def get_selected(self, include_children=False, fully=False):
        if fully:
            if self.pseudobonds.num_selected < self.num_pseudobonds:
                return False
            if include_children:
                for c in self.child_models():
                    if not c.get_selected(include_children=True, fully=True):
                        return False
            return True

        if self.pseudobonds.num_selected > 0:
            return True

        if include_children:
            for c in self.child_models():
                if c.get_selected(include_children=True):
                    return True
                
        return False
            
    def set_selected(self, sel, *, fire_trigger=True):
        self.pseudobonds.selected = sel
        Model.set_selected(self, sel, fire_trigger=fire_trigger)

    selected = property(get_selected, set_selected)

    def selected_items(self, itype):
        if itype == 'pseudobonds':
            pbonds = self.pseudobonds
            if pbonds.num_selected > 0:
                return [pbonds.filter(pbonds.selected)]
        elif itype == 'pseudobond groups':
            if self.pseudobonds.num_selected > 0:
                return [[self]]
        return []

    def selection_promotion(self):
        pbonds = self.pseudobonds
        n = pbonds.num_selected
        if n == 0 or n == len(pbonds):
            return None
        return PromotePseudobondSelection(self, pbonds.selected)

    def clear_selection(self):
        self.selected = False
        self.pseudobonds.selected = False
        super().clear_selection()

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
        self.change_tracker.add_modified(self, "dashes changed")

    dashes = property(_get_dashes, _set_dashes,
        doc="How many dashes pseudobonds will be drawn with")

    def _get_name(self):
        return self._category

    def _set_name(self, name):
        if name != self.name:
            self.change_name(name)
        # allow Model to fire 'name changed' trigger
        Model.name.fset(self, name)
    name = property(_get_name, _set_name,
        doc="Supported API. The name of the group.")

    # since we're a Model, we already have a 'session' attr, so don't need property

    def _get_model_color(self):
        pbonds = self.pseudobonds
        from chimerax.core.colors import most_common_color
        shown = pbonds.filter(pbonds.displays)
        if shown:
            return most_common_color(shown.colors)
        return self.color
    def _set_model_color(self, color):
        self.pseudobonds.colors = color
    model_color = property(_get_model_color, _set_model_color)

    def _update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc:
            self._graphics_changed = 0
            self._update_graphics(gc)
            self.redraw_needed(shape_changed = (gc & self._SHAPE_CHANGE),
                               highlight_changed = (gc & self._SELECT_CHANGE))

    def _update_graphics(self, changes = PseudobondGroupData._ALL_CHANGE):

        d = self._pbond_drawing
        if d is None:
            from .structure import BondsDrawing, PickedPseudobond, PickedPseudobonds
            d = self._pbond_drawing = BondsDrawing('pbonds', PickedPseudobond, PickedPseudobonds)
            self.update_cylinder_sides()
            self.add_drawing(d)
            d._visible_atoms = None
            changes = self._ALL_CHANGE
        elif self.num_pseudobonds == 0:
            self.remove_drawing(d)
            self._pbond_drawing = None
            return

        if changes & (self._ADDDEL_CHANGE | self._DISPLAY_CHANGE):
            changes = self._ALL_CHANGE
            
        if changes & self._DISPLAY_CHANGE or d.visible_bonds is None:
            vpb = self._shown_pbonds(self.pseudobonds)
            d.visible_bonds = vpb
            d._visible_atoms = vpb.atoms
        
        pbonds = d.visible_bonds
        bond_atoms = d._visible_atoms

        if changes & self._SHAPE_CHANGE:
            d.positions = self._update_positions(pbonds, bond_atoms)
            
        if changes & self._COLOR_CHANGE:
            d.colors = pbonds.half_colors
            
        if changes & self._SELECT_CHANGE:
            from . import structure as s
            d.highlighted_positions = s._selected_bond_cylinders(pbonds)

    def update_cylinder_sides(self):
        d = self._pbond_drawing
        if d is None:
            return False
        sides = self._cylinder_sides
        if sides == getattr(d, '_current_cylinder_sides', None):
            return False
        va, na, ta = _pseudobond_geometry(self._dashes//2, sides)
        d.set_geometry(va, na, ta)
        d._current_cylinder_sides = sides
        self._graphics_changed |= self._SHAPE_CHANGE
        return True

    def _update_positions(self, pbonds, bond_atoms):
        ba1, ba2 = bond_atoms
        if self._global_group:
            to_pbg = self.scene_position.inverse()
            axyz0, axyz1 = to_pbg*ba1.pb_scene_coords, to_pbg*ba2.pb_scene_coords
        else:
            axyz0, axyz1 = ba1.pb_coords, ba2.pb_coords
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

    @property
    def _cylinder_sides(self):
        from .structure import level_of_detail
        return level_of_detail(self.session).pseudobond_sides
    
    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None
        from . import structure
        if self._global_group:
            # Use scene coordinates since atoms may belong to different models.
            p = self.parent
            if p and not p.scene_position.is_identity():
                ps = p.scene_position
                sxyz1, sxyz2 = ps*mxyz1, ps*mxyz2               
            else:
                sxyz1, sxyz2 = mxyz1, mxyz2
            b,f = structure._bond_intercept(self.pseudobonds, sxyz1, sxyz2, scene_coordinates = True)
        else:
            b,f = structure._bond_intercept(self.pseudobonds, mxyz1, mxyz2)
        p = structure.PickedPseudobond(b,f) if b else None
        return p

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []

        picks = []
        from chimerax.geometry import transform_planes
        for p in self.positions:
            pplanes = transform_planes(p, planes)
            picks.extend(self._pseudobonds_planes_pick(pplanes))

        return picks

    def _pseudobonds_planes_pick(self, planes):
        from .structure import _bonds_planes_pick, PickedPseudobonds
        d = self._pbond_drawing
        if d is None or not d.display or d.visible_bonds is None:
            return []
        pmask = _bonds_planes_pick(d, planes)
        if pmask is None or pmask.sum() == 0:
            return []
        bonds = d.visible_bonds.filter(pmask)
        p = PickedPseudobonds(bonds)
        return [p]

    def take_snapshot(self, session, flags):
        data = {
            'version': 1,
            'category': self.name,
            'dashes': self._dashes,
            'model state': Model.take_snapshot(self, session, flags),
            'structure': self.structure,
            'custom attrs': self.custom_attrs,
        }
        if self._global_group:
            # Make the global manager restore before we do
            data['mgr'] = session.pb_manager

        return data

    @staticmethod
    def restore_snapshot(session, data):
        if data['structure'] is not None:
            grp = data['structure'].pseudobond_group(data['category'], create_type=None)
            # Handle case where the pseudobond group is not a child of the structure. Ticket #8853
            data['structure'].remove_drawing(grp, delete=False)
        else:
            grp = session.pb_manager.get_group(data['category'], create=False)
        if 'model state' in data:
            Model.set_state_from_snapshot(grp, session, data['model state'])
        grp._dashes = data['dashes']
        grp.set_custom_attrs(data)
        return grp

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
from chimerax.core.selection import SelectionPromotion
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
def all_pseudobond_groups(session):
    from . import PseudobondGroups
    return PseudobondGroups([m for m in session.models.list() if isinstance(m, PseudobondGroup)])

# -----------------------------------------------------------------------------
#
def all_pseudobonds(session):
    '''All pseudobonds in a :class:`.Pseudobonds` collection.'''
    pbgroups = all_pseudobond_groups(session)
    from .molarray import concatenate, Pseudobonds
    pbonds = concatenate([pbg.pseudobonds for pbg in pbgroups], Pseudobonds)
    return pbonds

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
              if group_name is None or pbg.name == group_name]
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
    from numpy import array
    hs = array([(s not in vis) for s in structures], bool)
    return hs

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9, sides = 10):
    from chimerax import surface
    return surface.dashed_cylinder_geometry(segments, height = 0.5, nc = sides)
