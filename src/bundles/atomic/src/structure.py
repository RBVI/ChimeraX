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

from chimerax.core import toolshed
from chimerax.core.models import Model
from chimerax.core.state import State
from .molobject import StructureData
from chimerax.graphics import Drawing, Pick

# If STRUCTURE_STATE_VERSION changes, then bump the bundle's
# (maximum) session version number.
STRUCTURE_STATE_VERSION = 1

# Auto-styling tunables
MULTI_SHADOW_THRESHOLD = 300_000  # reduce amount of shadow rays if more than threshold atoms
MULTI_SHADOW = 16               # lighting defaults to 64, so a 4x reduction
SMALL_THRESHOLD = 200_000       # not a small polymer if more than threshold atoms
MEDIUM_THRESHOLD = 1_000_000    # not a medium polymer if more than threshold atoms
MIN_RIBBON_THRESHOLD = 10       # skip ribbons if less than threshold ribbonable residues
MAX_RIBBON_THRESHOLD = 5000     # skip ribbons if more than threshold ribbonable residues
SLAB_THRESHOLD = 100            # skip slab nucleotide styling if more than threshold residues
LADDER_THRESHOLD = 2000         # skip ladder nucleotide styling if more than threshold residues

CATEGORY = toolshed.STRUCTURE

class Structure(Model, StructureData):
    """
    Structure model including atomic coordinates.
    The data is managed by the :class:`.StructureData` base class
    which provides access to the C++ structures.
    """

    def __init__(self, session, *, name = "structure", c_pointer = None, restore_data = None,
                 auto_style = True, log_info = True):
        from .molarray import Residues
        from numpy import array
        from .ribbon import XSectionManager

        # attrs that should be saved in sessions, along with their initial values...
        self._session_attrs = {
            '_auto_chain_trace': isinstance(self, AtomicStructure),
            '_bond_radius': 0.2,
            '_pseudobond_radius': 0.05,
            '_use_spline_normals': False,
            'ribbon_xs_mgr': XSectionManager(),
            'filename': None,
        }

        StructureData.__init__(self, c_pointer)
        for attr_name, val in self._session_attrs.items():
            setattr(self, attr_name, val)
        self.ribbon_xs_mgr.set_structure(self)
        Model.__init__(self, name, session)
        self._auto_style = auto_style
        self._log_info = log_info

        # for now, restore attrs to default initial values even for sessions...
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._chain_trace_pbgroup = None
        self._ribbons_drawing = None
        self._ring_drawing = None

        self._ses_handlers = []
        t = self.session.triggers
        for ses_func, trig_name in [("save_setup", "begin save session"),
                ("save_teardown", "end save session")]:
            self._ses_handlers.append(t.add_handler(trig_name,
                    lambda *args, qual=ses_func: self._ses_call(qual)))
        from chimerax.core.models import MODEL_POSITION_CHANGED, MODEL_DISPLAY_CHANGED
        self._ses_handlers.append(t.add_handler(MODEL_POSITION_CHANGED, self._update_position))
        self.triggers.add_trigger("changes")
        _register_hover_trigger(session)
        
        self._make_drawing()

        self.model_panel_show_expanded = False	# Don't show submodels initially in model panel

    def __str__(self):
        return self.string()

    def string(self, style=None):
        '''Return a human-readable string for this structure.'''
        if style is None:
            from .settings import settings
            style = settings.atomspec_contents

        # may need '#!' if there are Structure submodels
        for cm in self.all_models(): # child_models() is only direct children
            if cm is self:
                continue
            if isinstance(cm, Structure):
                prefix = "#!"
                break
        else:
            prefix = "#"
        id = prefix + self.id_string
        if style.startswith("command") or not self.name:
            return id
        return '%s %s' % (self.name, id)

    def delete(self):
        '''Delete this structure.'''
        t = self.session.triggers
        for handler in self._ses_handlers:
            t.remove_handler(handler)
        self._ses_handlers.clear()
        ses = self.session
        Model.delete(self)	# Delete children (pseudobond groups) before deleting structure
        # ensure we are checking StructureData.deleted, not Model.deleted
        if not StructureData.deleted.fget(self):
            self.session = ses
            StructureData.delete(self)
            delattr(self, 'session')

    @property
    def deleted(self):
        return StructureData.deleted.fget(self) or Model.deleted.fget(self)

    def combine(self, s, chain_id_mapping, ref_xform):
        '''
        Combine structure 's' into this structure.  'chain_id_mapping' is a chain ID -> chain ID
        dictionary describing how to change chain IDs of 's' when in conflict with this structure.
        'ref_xform' is the scene_position of the reference model.
        '''
        totals = self._get_instance_totals()
        StructureData._combine(self, s, chain_id_mapping, ref_xform)
        self._copy_custom_attrs(s, totals)

    def copy(self, name = None):
        '''
        Return a copy of this structure with a new name.
        No atoms or other components of the structure
        are shared between the original and the copy.
        '''
        if name is None:
            name = self.name
        m = self.__class__(self.session, name = name,
            c_pointer = StructureData._copy(self), auto_style = False, log_info = False)
        m.positions = self.positions
        m._copy_custom_attrs(self)
        return m

    def _get_instance_totals(self):
        return {
            'atoms': self.num_atoms,
            'bonds': self.num_bonds,
            'residues': self.num_residues,
            'chains': self.num_chains
        }

    def _copy_custom_attrs(self, source, totals=None):
        from .molobject import Chain
        for class_obj in [Atom, Bond, Chain, Residue]:
            py_objs = [py_obj for py_obj in python_instances_of_class(class_obj)
                if (not py_obj.deleted) and py_obj.structure == source and py_obj.has_custom_attrs]
            if not py_objs:
                continue
            class_attr = class_obj.__name__.lower() + 's'
            index_lookup = { obj:i for i, obj in enumerate(getattr(source, class_attr)) }
            base_index = 0 if totals is None else totals[class_attr]
            collection = getattr(self, class_attr)
            for py_obj in py_objs:
                collection[base_index + index_lookup[py_obj]].set_custom_attrs(
                    {'custom attrs': py_obj.custom_attrs})

    def added_to_session(self, session):
        if not self.scene_position.is_identity():
            self._cpp_notify_position(self.scene_position)
        if self._auto_style:
            self.apply_auto_styling(set_lighting = self._is_only_model())
        self._start_change_tracking(session.change_tracker)

        # Setup handler to manage C++ data changes that require graphics updates.
        self._graphics_updater.add_structure(self)
        Model.added_to_session(self, session)

    def removed_from_session(self, session):
        self._graphics_updater.remove_structure(self)

    def _is_only_model(self):
        id = self.id
        d = len(id)
        for m in self.session.models.list():
            if m.id[:d] != id:
                return False
        return True

    def apply_auto_styling(self, set_lighting = False, style=None):
        # most auto-styling only makes sense for atomic structures
        if set_lighting:
            kw = {} if self.num_atoms >= MULTI_SHADOW_THRESHOLD else {'multi_shadow': MULTI_SHADOW}
            from chimerax.std_commands.lighting import lighting
            lighting(self.session, preset = 'full', **kw)

    def take_snapshot(self, session, flags):
        data = {'model state': Model.take_snapshot(self, session, flags),
                'structure state': StructureData.save_state(self, session, flags),
                'custom attrs': self.custom_attrs }
        for attr_name in self._session_attrs.keys():
            data[attr_name] = getattr(self, attr_name)
        data['version'] = STRUCTURE_STATE_VERSION
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = Structure(session, auto_style = False, log_info = False)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        StructureData.set_state_from_snapshot(self, session, data['structure state'])
        Model.set_state_from_snapshot(self, session, data['model state'])

        for attr_name, default_val in self._session_attrs.items():
            setattr(self, attr_name, data.get(attr_name, default_val))
        self.ribbon_xs_mgr.set_structure(self)

        # Create Python pseudobond group models so they are added as children.
        list(self.pbg_map.values())

        # TODO: For some reason ribbon drawing does not update automatically.
        # TODO: Also marker atoms do not draw without this.
        self._graphics_changed |= (self._SHAPE_CHANGE | self._RIBBON_CHANGE | self._RING_CHANGE)

        self.set_custom_attrs(data)

    def _get_bond_radius(self):
        return self._bond_radius
    def _set_bond_radius(self, radius):
        self._bond_radius = radius
        self._graphics_changed |= self._SHAPE_CHANGE
    bond_radius = property(_get_bond_radius, _set_bond_radius)

    def _get_pseudobond_radius(self):
        return self._pseudobond_radius
    def _set_pseudobond_radius(self, radius):
        self._pseudobond_radius = radius
        self._graphics_changed |= self._SHAPE_CHANGE
    pseudobond_radius = property(_get_pseudobond_radius, _set_pseudobond_radius)

    def _structure_set_position(self, pos):
        if pos != self.position:
            Model.position.fset(self, pos)
            self.change_tracker.add_modified(self, "position changed")
    position = property(Model.position.fget, _structure_set_position)

    def _structure_set_positions(self, positions):
        if positions != self.positions:
            Model.positions.fset(self, positions)
            self.change_tracker.add_modified(self, "position changed")
    positions = property(Model.positions.fget, _structure_set_positions)

    def initial_color(self, bg_color):
        from .colors import structure_color
        id = self.id
        if id is None:
            max_id = max((m.id[0] for m in self.session.models.list(type = Structure)), default = 0)
            id = (max_id + 1,)
        return structure_color(id, bg_color)

    def set_initial_color(self):
        c = self.initial_color(self.session.main_view.background_color)
        self.set_color(c)
        from .colors import element_colors
        atoms = self.atoms
        het_atoms = atoms.filter(atoms.element_numbers != 6)
        het_atoms.colors = element_colors(het_atoms.element_numbers)

    def set_color(self, color):
        from chimerax.core.colors import Color
        if isinstance(color, Color):
            rgba = color.uint8x4()
        else:
            rgba = color
        StructureData.set_color(self, rgba)
        Model.set_color(self, rgba)

    def _get_model_color(self):
        residues = self.residues
        ribbon_displays = residues.ribbon_displays
        from chimerax.core.colors import most_common_color
        if ribbon_displays.any():
            return most_common_color(residues.filter(ribbon_displays).ribbon_colors)
        atoms = self.atoms
        shown = atoms.filter(atoms.displays)
        if shown:
            return most_common_color(shown.colors)
        if atoms:
            most_common_color(atoms.colors)
        return self.color

    def _set_model_color(self, color):
        Model.model_color.fset(self, color)
        self.atoms.colors = color
        residues = self.residues
        residues.ribbon_colors = color
        residues.ring_colors = color

    model_color = property(_get_model_color, _set_model_color)

    def _get_spline_normals(self):
        return self._use_spline_normals
    def _set_spline_normals(self, sn):
        if sn != self._use_spline_normals:
            self._use_spline_normals = sn
            self._graphics_changed |= self._RIBBON_CHANGE
    spline_normals = property(_get_spline_normals, _set_spline_normals)

    def _make_drawing(self):
        # Create graphics
        self._update_atom_graphics()
        self._update_bond_graphics()
        for pbg in self.pbg_map.values():
            pbg._update_graphics()
        self._create_ribbon_graphics()

    @property
    def _level_of_detail(self):
        return self._graphics_updater.level_of_detail

    @property
    def _graphics_updater(self):
        return structure_graphics_updater(self.session)

    def new_atoms(self):
        # TODO: Handle instead with a C++ notification that atoms added or deleted
        pass

    def update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc == 0:
            return

        if gc & self._RIBBON_CHANGE:
            self._create_ribbon_graphics()
            # Displaying ribbon can set backbone atom hide bits producing shape change.
            gc |= self._graphics_changed

        if gc & self._RING_CHANGE:
            self._create_ring_graphics()

        # Update graphics
        self._graphics_changed = 0
        s = (gc & self._SHAPE_CHANGE)
        if gc & (self._COLOR_CHANGE | self._RIBBON_CHANGE) or s:
            self._update_ribbon_tethers()
        self._update_graphics(gc)
        self.redraw_needed(shape_changed = s,
                           highlight_changed = (gc & self._SELECT_CHANGE))

        if gc & self._SELECT_CHANGE:
            # Update selection in child drawings (e.g., surfaces)
            # TODO: Won't work for surfaces spanning multiple molecules
            for d in self.child_drawings():
                update = getattr(d, 'update_selection', None)
                if update is not None:
                    update()

    def _update_graphics(self, changes = StructureData._ALL_CHANGE):
        self._update_atom_graphics(changes)
        self._update_bond_graphics(changes)
        if self._auto_chain_trace:
            self._update_chain_trace_graphics(changes)
        for pbg in self.pbg_map.values():
            pbg._update_graphics(changes)
        self._update_ribbon_graphics(changes)

    def _update_atom_graphics(self, changes = StructureData._ALL_CHANGE):

        p = self._atoms_drawing
        if p is None:
            changes = self._ALL_CHANGE
            self._atoms_drawing = p = AtomsDrawing('atoms')
            self.add_drawing(p)
            # Update level of detail of spheres
            self._level_of_detail.set_atom_sphere_geometry(p)

        if changes & (self._ADDDEL_CHANGE | self._DISPLAY_CHANGE):
            changes |= self._ALL_CHANGE

        if changes & self._DISPLAY_CHANGE:
            all_atoms = self.atoms
            p.visible_atoms = all_atoms[all_atoms.visibles]

        atoms = p.visible_atoms

        if changes & self._SHAPE_CHANGE:
            # Set instanced sphere center position and radius
            n = len(atoms)
            from numpy import empty, float32, multiply
            xyzr = empty((n, 4), float32)
            xyzr[:, :3] = atoms.coords
            xyzr[:, 3] = self._atom_display_radii(atoms)

            from chimerax.geometry import Places
            p.positions = Places(shift_and_scale=xyzr)

        if changes & self._COLOR_CHANGE:
            # Set atom colors
            p.colors = atoms.colors

        if changes & self._SELECT_CHANGE:
            # Set selected
            p.highlighted_positions = atoms.selected if atoms.num_selected > 0 else None

    def _atom_display_radii(self, atoms):
        return atoms.display_radii(self.ball_scale, self.bond_radius)

    def _update_bond_graphics(self, changes = StructureData._ALL_CHANGE):

        p = self._bonds_drawing
        if p is None:
            if self.num_bonds_visible == 0:
                return
            changes = self._ALL_CHANGE
            self._bonds_drawing = p = BondsDrawing('bonds', PickedBond, PickedBonds)
            self.add_drawing(p)
            # Update level of detail of cylinders
            self._level_of_detail.set_bond_cylinder_geometry(p)

        if changes & (self._ADDDEL_CHANGE | self._DISPLAY_CHANGE):
            changes |= self._ALL_CHANGE

        if changes & self._DISPLAY_CHANGE:
            all_bonds = self.bonds
            p.visible_bonds = all_bonds[all_bonds.showns]

        bonds = p.visible_bonds

        if changes & self._SHAPE_CHANGE:
            p.positions = bonds.halfbond_cylinder_placements(p.positions.opengl_matrices())

        if changes & self._COLOR_CHANGE:
            p.colors = bonds.half_colors

        if changes & self._SELECT_CHANGE:
            p.highlighted_positions = _selected_bond_cylinders(bonds)

    def _get_autochain(self):
        return self._auto_chain_trace
    def _set_autochain(self, autochain):
        if autochain != self._auto_chain_trace:
            self._auto_chain_trace = autochain
            if autochain:
                self._update_chain_trace_graphics()
            else:
                self._close_chain_trace()
    autochain = property(_get_autochain, _set_autochain)
    '''Whether chain trace between principal residue atoms is shown when only those atoms are displayed.'''
    
    def _update_chain_trace_graphics(self, changes = StructureData._ALL_CHANGE):

        if changes & (self._ADDDEL_CHANGE | self._DISPLAY_CHANGE):
            changes |= self._ALL_CHANGE

        if changes & self._DISPLAY_CHANGE:
            cta = self.chain_trace_atoms()

            pbg = self._chain_trace_pbgroup
            if pbg is None or pbg.deleted:
                if cta is None:
                    return
                changes = self._ALL_CHANGE
                self._chain_trace_pbgroup = pbg = self.pseudobond_group('chain trace')
                pbg._chain_atoms = None
                pbg.dashes = 0
            elif cta is None:
                self._close_chain_trace()
                return

            if cta != pbg._chain_atoms:
                pbg.pseudobonds.delete()
                pbonds = pbg.new_pseudobonds(cta[0], cta[1])
                pbonds.halfbonds = True
                pbg._chain_atoms = cta

    def _close_chain_trace(self):
        pbg = self._chain_trace_pbgroup
        if pbg:
            self.session.models.close([pbg])
            self._chain_trace_pbgroup = None

    def _get_display(self):
        return Model.display.fget(self)

    def _set_display(self, display):
        if display == self.display:
            return
        Model.display.fset(self, display)
        # ensure that "display changed" trigger fires
        StructureData.display.fset(self, display)

    display = property(_get_display, _set_display)

    def _update_level_of_detail(self, total_atoms):
        lod = self._level_of_detail
        bd = self._bonds_drawing
        if bd:
            lod.set_bond_cylinder_geometry(bd, total_atoms)
        ad = self._atoms_drawing
        if ad:
            lod.set_atom_sphere_geometry(ad, total_atoms)

    def _update_position(self, trig_name, updated_model):
        need_update = False
        check_model = self
        while isinstance(check_model, Model):
            if updated_model == check_model:
                need_update = True
                break
            check_model = check_model.parent

        if need_update:
            self._cpp_notify_position(self.scene_position)

    def _create_ring_graphics(self):
        p = self._ring_drawing
        if p is not None:
            self.remove_drawing(p)
            self._ring_drawing = None
        if self.ring_display_count == 0:
            return

        from .shapedrawing import AtomicShapeDrawing
        self._ring_drawing = p = self.new_drawing('rings', subclass=AtomicShapeDrawing)

        # TODO:
        #   find all residue rings
        #   limit to 3, 4, 5 and 6 member rings
        #   check if all atoms are shown (displayed and not hidden)
        #   if thin, will only use one two-sided fill
        #   if thick, use stick radius to separate fills
        ring_count = 0
        all_rings = self.rings(all_size_threshold=6)
        # Ring info will change spontaneously when we ask for radii, so remember what we need now
        ring_atoms = [ring.ordered_atoms for ring in all_rings]
        rings = []
        for atoms in ring_atoms:
            residue = atoms[0].residue
            if not residue.ring_display or not all(atoms.visibles):
                continue
            ring_count += 1
            if residue.thin_rings:
                offset = 0
            else:
                offset = min(self._atom_display_radii(atoms))
            if len(atoms) < 6:
                rings.append(self.fill_small_ring(atoms, offset, residue.ring_color))
            else:
                rings.append(self.fill_6ring(atoms, offset, residue.ring_color))

        if ring_count:
            self._ring_drawing.add_shapes(rings)
            self._graphics_changed |= self._SHAPE_CHANGE

    def _res_numbering(self, rn):
        rn_lookup = { 'author': Residue.RN_AUTHOR, 'canonical': Residue.RN_CANONICAL,
            'uniprot': Residue.RN_UNIPROT }
        if isinstance(rn, int):
            if not (0 <= rn < len(rn_lookup)):
                raise ValueError("Residue numbering value must be between 0 and %d inclusive"
                    % len(rn_lookup))
        else:
            try:
                rn = rn_lookup[rn.lower()]
            except KeyError:
                from chimerax.core.commands import commas
                raise ValueError("Residue numbering value must be %s"
                    % commas([repr(k) for k in rn_lookup.values()]))
        if rn == self.res_numbering:
            return
        if not self.res_numbering_valid(rn) and rn == Residue.RN_UNIPROT:
            # see if we can set it
            u_info = uniprot_ids(self)
            if u_info:
                self.res_numbering = Residue.RN_AUTHOR
                by_chain = { u.chain_id:(u.chain_sequence_range,u.database_sequence_range) for u in u_info }
                for chain in self.chains:
                    try:
                        struct_range, db_range = by_chain[chain.chain_id]
                    except KeyError:
                        continue
                    offset = db_range[0] - struct_range[0]
                    # can't use self.renumber_residues() because of possible missing structure
                    for r in chain.existing_residues:
                        r.set_number(rn, r.number + offset)
                self.set_res_numbering_valid(rn, True)
        if not self.res_numbering_valid(rn):
            reverse_lookup = { Residue.RN_AUTHOR: "author", Residue.RN_CANONICAL: "canonical",
                Residue.RN_UNIPROT: "UniProt" }
            raise ValueError("%s residue numbering has not been assigned; maintaining %s numbering"
                % (reverse_lookup[rn].capitalize(), reverse_lookup[self.res_numbering]))
        StructureData.res_numbering.fset(self, rn)
    res_numbering = property(StructureData.res_numbering.fget, _res_numbering)


    def fill_small_ring(self, atoms, offset, color):
        # 3-, 4-, and 5- membered rings
        from chimerax.geometry import fill_small_ring
        from .shapedrawing import AtomicShapeInfo
        vertices, normals, triangles = fill_small_ring(atoms.coords, offset)
        return AtomicShapeInfo(vertices, normals, triangles, color, atoms)

    def fill_6ring(self, atoms, offset, color):
        # 6-membered rings
        from chimerax.geometry import fill_6ring
        from .shapedrawing import AtomicShapeInfo
        # Picking the "best" orientation to show chair/boat configuration is hard
        # so choose anchor the ring using atom nomenclature.
        # Find index of atom with lowest element with lowest number (C1 < C6).
        # Cheat and do lexicographical comparison of name.
        # TODO: compare speed of algorithms
        # Algorithm1:
        # choices = [(a.element.number, a.name, i) for i, a in enumerate(atoms)]
        # choices.sort()
        # anchor = choices[0][2]
        # Algorithm2:
        # choices = zip(atoms.elements.numbers, atoms.names, range(len(atoms)))
        # choices.sort()
        # anchor = choices[0][2]
        # Algorithm3:
        anchor = 0
        anchor_element = atoms[0].element.number
        anchor_name = atoms[0].name
        for i, a in enumerate(atoms[1:], 1):
            e = a.element.number
            if e > anchor_element:
                continue
            if e == anchor_element and a.name >= anchor_name:
                continue
            anchor = i
            anchor_element = e
            anchor_name = a.name
        vertices, normals, triangles = fill_6ring(atoms.coords, offset, anchor)
        return AtomicShapeInfo(vertices, normals, triangles, color, atoms)

    def _create_ribbon_graphics(self):
        ribbons_drawing = self._ribbons_drawing
        if ribbons_drawing is None:
            from .ribbon import RibbonsDrawing
            ribbons_drawing = rd = RibbonsDrawing('ribbons', self.string(style='simple'))
            self._ribbons_drawing = rd
            self.add_drawing(rd)

        ribbons_drawing.compute_ribbons(self)
        
        self._graphics_changed |= self._SHAPE_CHANGE

    def _update_ribbon_graphics(self, changes = StructureData._ALL_CHANGE):
        # Ribbon is recomputed when needed by _create_ribbon_graphics()
        # Only selection and color is updated here.

        rd = self._ribbons_drawing
        if rd is None:
            return
        
        if changes & (self._SELECT_CHANGE | self._RIBBON_CHANGE):
            rd.update_ribbon_highlight()

        if changes & self._COLOR_CHANGE and not (changes & self._RIBBON_CHANGE):
            rd.update_ribbon_colors()

    def _update_ribbon_tethers(self):
        rd = self._ribbons_drawing
        if rd:
            rd.update_tethers(self)

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None

        picks = []
        np = len(self.positions)
        if np > 1:
            b = self._pick_bounds()
            pos_nums = self.bounds_intercept_copies(b, mxyz1, mxyz2)
        else:
            # Don't do bounds check for single copy because bounds are not cached.
            pos_nums = range(np)
        for pn in pos_nums:
            ppicks = self._position_intercepts(self.positions[pn], mxyz1, mxyz2, exclude)
            picks.extend(ppicks)
            for p in ppicks:
                p.copy_number = pn

        pclosest = None
        for p in picks:
            if pclosest is None or p.distance < pclosest.distance:
                pclosest = p
        return pclosest

    def _pick_bounds(self):
        '''
        Bounds for this model not including positions.  Used for optimizing picking
        when there are multiple positions.  Includes atoms and ribbons.
        '''
        ad = self._atoms_drawing
        rd = self._ribbons_drawing
        drawings = [ad, rd]
        from chimerax.geometry import union_bounds
        b = union_bounds([d.bounds() for d in drawings if d is not None])
        return b

    def _position_intercepts(self, place, mxyz1, mxyz2, exclude=None):
        # TODO: check intercept of bounding box as optimization
        xyz1, xyz2 = place.inverse() * (mxyz1, mxyz2)
        pa = None
        pb = None
        ppb = None
        picks = []
        for d in self.child_drawings():
            if not d.display or (exclude is not None and exclude(d)):
                continue
            p = d.first_intercept(xyz1, xyz2, exclude=exclude)
            if p is None:
                continue
            if isinstance(p, PickedAtom):
                pa = p
            elif isinstance(p, PickedBond):
                pb = p
                continue
            elif isinstance(p, PickedPseudobond):
                ppb = p
                continue
            picks.append(p)
        if pb:
            if pa:
                a = pa.atom
                if a.draw_mode != a.STICK_STYLE or a not in pb.bond.atoms:
                    picks.append(pb)
            else:
                picks.append(pb)
        if ppb:
            if pa:
                a = pa.atom
                if a.draw_mode != a.STICK_STYLE or a not in ppb.pbond.atoms:
                    picks.append(ppb)
            else:
                picks.append(ppb)
        return picks

    def x3d_needs(self, x3d_scene):
        self.update_graphics_if_needed()       # Ribbon drawing lazily computed
        super().x3d_needs(x3d_scene)

    def write_x3d(self, *args, **kw):
        self.update_graphics_if_needed()       # Ribbon drawing lazily computed
        super().write_x3d(*args, **kw)

    def get_selected(self, include_children=False, fully=False):
        if fully:
            if self.atoms.num_selected < self.num_atoms or self.bonds.num_selected < self.num_bonds:
                return False
            if include_children:
                for c in self.child_models():
                    if not c.get_selected(include_children=True, fully=True):
                        return False
            return True

        if self.atoms.num_selected > 0 or self.bonds.num_selected > 0:
            return True

        if include_children:
            for c in self.child_models():
                if c.get_selected(include_children=True):
                    return True

        return False

    def set_selected(self, sel, *, fire_trigger=True):
        self.atoms.selected = sel
        self.bonds.selected = sel
        Model.set_selected(self, sel, fire_trigger=fire_trigger)
    selected = property(get_selected, set_selected)

    def set_selected_positions(self, spos):
        sel = (spos is not None and spos.sum() > 0)
        self.atoms.selected = sel
        self.bonds.selected = sel
        Model.set_highlighted_positions(self, spos)
    selected_positions = property(Model.selected_positions.fget, set_selected_positions)

    def selected_items(self, itype):
        if itype == 'atoms':
            atoms = self.atoms
            if atoms.num_selected > 0:
                return [atoms.filter(atoms.selected)]
        elif itype == 'bonds':
            bonds = self.bonds
            if bonds.num_selected > 0:
                return [bonds.filter(bonds.selected)]
        elif itype == 'residues':
            from . import concatenate, Atoms
            atoms, bonds = self.atoms, self.bonds
            sel_residues = []
            if atoms.num_selected > 0:
                sel_residues.append(atoms.filter(atoms.selected).residues)
            if bonds.num_selected > 0:
                sel_bonds = bonds.filter(bonds.selected)
                atoms1, atoms2 = sel_bonds.atoms
                is_intra = atoms1.residues.pointers == atoms2.residues.pointers
                same_res = atoms1.residues.filter(is_intra)
                if same_res:
                    sel_residues.append(same_res)
            if sel_residues:
                from . import concatenate, Residues
                return [concatenate(sel_residues, Residues, remove_duplicates=True).unique()]
        elif itype == 'structures':
            return [[self]] if self.selected else []
        return []

    def clear_selection(self):
        self.selected = False
        self.atoms.selected = False
        self.bonds.selected = False
        super().clear_selection()

    def selection_promotion(self):
        atoms = self.atoms
        bonds = self.bonds
        na = atoms.num_selected
        nb = bonds.num_selected
        if (na == 0 and nb == 0) or (na == len(atoms) and nb == len(bonds)):
            return None
        asel = atoms.selected
        bsel = bonds.selected

        if nb > 0 and not bonds[bsel].ends_selected.all():
            # Promote to include selected bond atoms
            level = 1006
            psel = asel | atoms.has_selected_bonds
        else:
            r = atoms.residues
            rids = r.unique_ids
            from numpy import unique, in1d
            sel_rids = unique(rids[asel])
            ares = in1d(rids, sel_rids)
            if ares.sum() > na:
                # Promote to entire residues
                level = 1005
                psel = ares
            else:
                ssids = r.secondary_structure_ids
                sel_ssids = unique(ssids[asel])
                ass = in1d(ssids, sel_ssids)
                if ass.sum() > na:
                    # Promote to secondary structure
                    level = 1004
                    psel = ass
                else:
                    frag_sel = self.frag_sel
                    if frag_sel.sum() > na:
                        level = 1003
                        psel = frag_sel
                    else:
                        from numpy import array
                        cids = array(r.chain_ids)
                        sel_cids = unique(cids[asel])
                        ac = in1d(cids, sel_cids)
                        if ac.sum() > na:
                            # Promote to entire chains
                            level = 1002
                            psel = ac
                        else:
                            # Promote to entire molecule
                            level = 1001
                            ac[:] = True
                            psel = ac

        return PromoteAtomSelection(self, level, psel, asel, bsel)

    def surfaces(self):
        '''List of :class:`.MolecularSurface` objects for this structure.'''
        from .molsurf import MolecularSurface
        surfs = [s for s in self.child_models() if isinstance(s, MolecularSurface)]
        return surfs

    # Atom specifier API
    def atomspec_has_atoms(self):
        return True

    def atomspec_atoms(self, ordered=False):
        if ordered:
            from .molarray import Atoms
            return Atoms(sorted(self.atoms))
        return self.atoms

    def atomspec_filter(self, level, atoms, num_atoms, parts, attrs):
        # print("Structure.atomspec_filter", level, num_atoms, parts, attrs)
        if parts is None:
            parts = []
        if attrs is None:
            attrs = []
        if level == '/':
            return self._atomspec_filter_chain(atoms, num_atoms, parts, attrs)
        elif level == ':':
            return self._atomspec_filter_residue(atoms, num_atoms, parts, attrs)
        elif level == '@':
            return self._atomspec_filter_atom(atoms, num_atoms, parts, attrs)

    def _atomspec_filter_chain(self, atoms, num_atoms, parts, attrs):
        # print("Structure._atomspec_filter_chain", num_atoms, parts, attrs)
        import numpy
        chain_ids = atoms.residues.chain_ids
        if not parts:
            selected = numpy.ones(num_atoms, dtype=numpy.bool_)
        else:
            selected = numpy.zeros(num_atoms, dtype=numpy.bool_)
            for part in parts:
                choose = part.string_matcher(self.lower_case_chains)
                s = numpy.vectorize(choose)(chain_ids)
                selected = numpy.logical_or(selected, s)
        if attrs:
            chains = self.chains
            chain_selected = numpy.ones(len(chains), dtype=numpy.bool_)
            chain_selected = self._atomspec_attr_filter(chains, chain_selected, attrs)
            chain_map = dict(zip(chains, chain_selected))
            for i, a in enumerate(atoms):
                if not selected[i]:
                    continue
                try:
                    if not chain_map[a.residue.chain]:
                        selected[i] = False
                except KeyError:
                    # XXX: Atom is not in a chain, "must not be selected?"
                    selected[i] = False
        # print("AtomicStructure._atomspec_filter_chain", selected)
        return selected

    def _atomspec_attr_filter(self, objects, selected, attrs):
        import numpy
        for attr in attrs:
            choose = attr.attr_matcher()
            selected = [(selected[i] and choose(obj)) for i, obj in enumerate(objects)]
        return numpy.array(selected)


    def _atomspec_filter_residue(self, atoms, num_atoms, parts, attrs):
        # print("Structure._atomspec_filter_residue", num_atoms, parts, attrs)
        import numpy
        if not parts:
            # No residue specifier, choose everything
            selected = numpy.ones(num_atoms, dtype=numpy.bool_)
        else:
            res_names = numpy.array(atoms.residues.names)
            res_numbers = atoms.residues.numbers
            res_ics = atoms.residues.insertion_codes
            selected = numpy.zeros(num_atoms, dtype=numpy.bool_)
            for part in parts:
                choose_id = part.res_id_matcher()
                if choose_id:
                    s = numpy.vectorize(choose_id)(res_numbers, res_ics)
                    if s.any():
                        selected = numpy.logical_or(selected, s)
                    else:
                        choose_id = None
                        # Try using input as name instead of number
                if not choose_id:
                    choose_type = part.string_matcher(False)
                    s = numpy.vectorize(choose_type)(res_names)
                    selected = numpy.logical_or(selected, s)
        if attrs:
            selected = self._atomspec_attr_filter(atoms.residues, selected, attrs)
        # print("AtomicStructure._atomspec_filter_residue", selected)
        return selected

    def _atomspec_filter_atom(self, atoms, num_atoms, parts, attrs):
        # print("Structure._atomspec_filter_atom", num_atoms, parts, attrs)
        import numpy
        if not parts:
            # No name specifier, use everything
            selected = numpy.ones(num_atoms, dtype=numpy.bool_)
        else:
            names = numpy.array(atoms.names)
            selected = numpy.zeros(num_atoms, dtype=numpy.bool_)
            for part in parts:
                choose = part.string_matcher(False)
                s = numpy.vectorize(choose)(names)
                selected = numpy.logical_or(selected, s)
        if attrs:
            selected = self._atomspec_attr_filter(atoms, selected, attrs)
        # print("AtomicStructure._atomspec_filter_atom", selected)
        return selected

    def atomspec_zone(self, session, coords, distance, target_type, operator, results):
        from chimerax.geometry import find_close_points
        atoms = self.atoms
        a, _ = find_close_points(atoms.scene_coords, coords, distance)
        def not_a():
            from numpy import ones, bool_
            mask = ones(len(atoms), dtype=bool_)
            mask[a] = False
            return mask
        expand_by = None
        if target_type == '@':
            if '<' in operator:
                expand_by = atoms.filter(a)
            else:
                expand_by = atoms.filter(not_a())
        elif target_type == ':':
            if '<' in operator:
                expand_by = atoms.filter(a).unique_residues.atoms
            else:
                expand_by = atoms.filter(not_a()).full_residues.atoms
        elif target_type == '/':
            # There is no "full_chain" property for atoms so we have
            # to do it the hard way
            from numpy import in1d, invert
            matched_chain_ids = atoms.filter(a).unique_chain_ids
            mask = in1d(atoms.chain_ids, matched_chain_ids)
            if '<' in operator:
                expand_by = atoms.filter(mask)
            else:
                expand_by = atoms.filter(invert(mask))
        elif target_type == '#':
            if '<' in operator:
                expand_by = atoms.filter(a).unique_structures.atoms
            else:
                expand_by = atoms.filter(not_a()).full_structures.atoms
        if expand_by:
            results.add_atoms(expand_by)
            results.add_model(self)

class AtomsDrawing(Drawing):
    # can't have any child drawings
    # requires self.parent._atom_display_radii()

    def __init__(self, name):
        self.visible_atoms = None
        super().__init__(name)

    def bounds(self):
        cpb = self._cached_position_bounds	# Attribute of Drawing.
        if cpb is not None:
            return cpb
        a = self.visible_atoms
        if len(a) == 0:
            return None
        adisp = a[a.displays]
        coords = adisp.coords
        radii = self.parent._atom_display_radii(adisp)
        # TODO: Currently 40% of time is taken in getting atom radii because
        #       they are recomputed from element and bonds every time. Ticket #789.
        #       If that was fixed by using a precomputed radius, then it would make
        #       sense to optimize this bounds calculation in C++ so arrays
        #       of display state, radii and coordinates are not needed.
        from chimerax.geometry import sphere_bounds, copies_bounding_box
        sb = sphere_bounds(coords, radii)
        spos = self.parent.get_scene_positions(displayed_only=True)
        b = sb if spos.is_identity() else copies_bounding_box(sb, spos)
        self._cached_position_bounds = b

        return b

    def add_drawing(self, d):
        raise NotImplemented("AtomsDrawing may not have children")

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or self.visible_atoms is None or (exclude and exclude(self)):
            return None

        if len(self.visible_atoms) < len(self.positions):
            # Some atoms were deleted since the last time the graphics was drawn.
            return None

        xyzr = self.positions.shift_and_scale_array()
        coords, radii = xyzr[:,:3], xyzr[:,3]

        # Check for atom sphere intercept
        from chimerax import geometry
        fmin, anum = geometry.closest_sphere_intercept(coords, radii, mxyz1, mxyz2)
        if fmin is None:
            return None

        atom = self.visible_atoms[anum]

        # Create pick object
        s = PickedAtom(atom, fmin)
        return s

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []
        if self.visible_atoms is None:
            return []

        xyz = self.positions.shift_and_scale_array()[:,:3]
        from chimerax import geometry
        pmask = geometry.points_within_planes(xyz, planes)
        if pmask.sum() == 0:
            return []
        atoms = self.visible_atoms.filter(pmask)
        p = PickedAtoms(atoms)
        return [p]

    def x3d_needs(self, x3d_scene):
        from chimerax.core import x3d
        x3d_scene.need(x3d.Components.Grouping, 1)  # Group, Transform
        x3d_scene.need(x3d.Components.Shape, 1)  # Appearance, Material, Shape
        x3d_scene.need(x3d.Components.Geometry3D, 1)  # Sphere

    def custom_x3d(self, stream, x3d_scene, indent, place):
        from numpy import empty, float32
        if self.empty_drawing():
            return
        xyzr = self.positions.shift_and_scale_array()
        coords, radii = xyzr[:, :3], xyzr[:, 3]
        tab = ' ' * indent
        for xyz, r, c in zip(coords, radii, self.colors):
            print('%s<Transform translation="%g %g %g">' % (tab, xyz[0], xyz[1], xyz[2]), file=stream)
            print('%s <Shape>' % tab, file=stream)
            self.reuse_appearance(stream, x3d_scene, indent + 2, c)
            print('%s  <Sphere radius="%g"/>' % (tab, r), file=stream)
            print('%s </Shape>' % tab, file=stream)
            print('%s</Transform>' % tab, file=stream)

class BondsDrawing(Drawing):
    # Used for both bonds and pseudoonds.
    # Should not have any child drawings, as bounds and picking will ignore any children.
    #
    # If zero length bonds are included then there will be singular position matrices
    # that will cause errors in any code that relies on inverting position matrices.
    # Ideally zero length bonds would be removed from drawn geometry and all positions
    # would be invertible.  But the code is simpler and faster if all displayed bonds are
    # included, so we will tolerate the singular position matrices unless the cause
    # problems.
    #
    skip_bounds = True

    def __init__(self, name, pick_class, picks_class):
        self.visible_bonds = None
        self._pick_class = pick_class
        self._picks_class = picks_class
        super().__init__(name)

    def bounds(self):
        cpb = self._cached_position_bounds	# Attribute of Drawing.
        if cpb is not None:
            return cpb
        bonds = self.visible_bonds
        if bonds is None or len(bonds) == 0:
            return None
        ba1, ba2 = bonds.atoms
        c1, c2, r = ba1.coords, ba2.coords, bonds.radii
        r.shape = (r.shape[0], 1)
        from numpy import amin, amax
        xyz_min = amin([amin(c1 - r, axis=0), amin(c2 - r, axis=0)], axis=0)
        xyz_max = amax([amax(c1 + r, axis=0), amax(c2 + r, axis=0)], axis=0)
        from chimerax.geometry import Bounds, copies_bounding_box
        cb = Bounds(xyz_min, xyz_max)
        spos = self.parent.get_scene_positions(displayed_only=True)
        b = cb if spos.is_identity() else copies_bounding_box(cb, spos)
        self._cached_position_bounds = b

        return b

    def add_drawing(self, d):
        raise NotImplemented("BondsDrawing may not have children")

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None
        bonds = self.visible_bonds
        b, f = _bond_intercept(bonds, mxyz1, mxyz2)
        if b:
            return self._pick_class(b, f)
        return None

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []
        if self.visible_bonds is None:
            return []

        pmask = _bonds_planes_pick(self, planes)
        if pmask is None or pmask.sum() == 0:
            return []
        bonds = self.visible_bonds.filter(pmask)
        p = PickedBonds(bonds)
        return [p]

    def x3d_needs(self, x3d_scene):
        from chimerax.core import x3d
        x3d_scene.need(x3d.Components.Grouping, 1)  # Group, Transform
        x3d_scene.need(x3d.Components.Shape, 1)  # Appearance, Material, Shape
        x3d_scene.need(x3d.Components.Geometry3D, 1)  # Cylinder

    def custom_x3d(self, stream, x3d_scene, indent, place):
        # TODO: handle dashed bonds
        from numpy import empty, float32
        bonds = self.visible_bonds
        if bonds is None:
            return
        ba1, ba2 = bonds.atoms
        cyl_info = _halfbond_cylinder_x3d(ba1.effective_coords, ba2.effective_coords, bonds.radii)
        tab = ' ' * indent
        for ci, c in zip(cyl_info, self.colors):
            h = ci[0]
            r = ci[1]
            rot = ci[2:6]
            xyz = ci[6:9]
            print('%s<Transform translation="%g %g %g" rotation="%g %g %g %g">' % (tab, xyz[0], xyz[1], xyz[2], rot[0], rot[1], rot[2], rot[3]), file=stream)
            print('%s <Shape>' % tab, file=stream)
            self.reuse_appearance(stream, x3d_scene, indent + 2, c)
            print('%s  <Cylinder height="%g" radius="%g" bottom="false" top="false"/>' % (tab, h, r), file=stream)
            print('%s </Shape>' % tab, file=stream)
            print('%s</Transform>' % tab, file=stream)



class AtomicStructure(Structure):
    """
    Molecular model including support for chains, hetero-residues,
    and assemblies.
    """

    # changes to the below have to be mirrored in C++ AS_PBManager::get_group
    from chimerax.core.colors import BuiltinColors
    default_hbond_color = BuiltinColors["deep sky blue"]
    default_hbond_radius = 0.075
    default_hbond_dashes = 6

    default_metal_coordination_color = BuiltinColors["medium purple"]
    default_metal_coordination_radius = 0.075
    default_metal_coordination_dashes = 6

    default_missing_structure_color = BuiltinColors["yellow"]
    default_missing_structure_radius = 0.075
    default_missing_structure_dashes = 6

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._set_chain_descriptions(self.session)
        self._determine_het_res_descriptions(self.session)

    def added_to_session(self, session):
        super().added_to_session(session)

        if self._log_info:
            # don't report models in an NMR ensemble individually...
            if len(self.id) > 1:
                sibs = [m for m in session.models
                        if isinstance(m, AtomicStructure) and m.id[:-1] == self.id[:-1]]
                if len(set([s.name for s in sibs])) > 1:
                    # not an NMR ensemble
                    self._report_chain_descriptions(session)
                    self._report_res_info(session)
                else:
                    sibs.sort(key=lambda m: m.id)
                    if sibs[-1] == self:
                        self._report_ensemble_chain_descriptions(session, sibs)
                        self._report_res_info(session)
            else:
                self._report_chain_descriptions(session)
                self._report_res_info(session)
            self._report_assemblies(session)
            self._report_model_info(session)

    def apply_auto_styling(self, set_lighting = False, style=None):
        explicit_style = style is not None
        if style is None:
            if self.num_chains == 0:
                style = "non-polymer"
            elif self.num_chains < 5 and len(self.atoms) < SMALL_THRESHOLD \
            and len(self.chains.existing_residues) < MAX_RIBBON_THRESHOLD:
                style = "small polymer"
            elif self.num_chains < 250 and len(self.atoms) < MEDIUM_THRESHOLD:
                style = "medium polymer"
            else:
                style = "large polymer"

        color = self.initial_color(self.session.main_view.background_color)
        self.set_color(color)

        atoms = self.atoms
        if style == "non-polymer":
            lighting = {'preset': 'default'}
            from .molobject import Atom, Bond
            atoms.draw_modes = Atom.STICK_STYLE
            from .colors import element_colors
            het_atoms = atoms.filter(atoms.element_numbers != 6)
            het_atoms.colors = element_colors(het_atoms.element_numbers)
        elif style == "small polymer":
            lighting = {'preset': 'default'}
            from .molobject import Atom, Bond, Residue
            atoms.draw_modes = Atom.STICK_STYLE
            from .colors import element_colors
            het_atoms = atoms.filter(atoms.element_numbers != 6)
            het_atoms.colors = element_colors(het_atoms.element_numbers)
            ribbonable = self.chains.existing_residues
            # 10 residues or less is basically a trivial depiction if ribboned
            if explicit_style or MIN_RIBBON_THRESHOLD < len(ribbonable):
                atoms.displays = False
                ligand = atoms.filter(atoms.structure_categories == "ligand").residues
                ribbonable -= ligand
                metal_atoms = atoms.filter(atoms.elements.is_metal)
                metal_atoms.draw_modes = Atom.SPHERE_STYLE
                ions = atoms.filter(atoms.structure_categories == "ions")
                lone_ions = ions.filter(ions.residues.num_atoms == 1)
                lone_ions.draw_modes = Atom.SPHERE_STYLE
                ligand |= metal_atoms.residues
                display = ligand
                mask = ribbonable.polymer_types == Residue.PT_NUCLEIC
                nucleic = ribbonable.filter(mask)
                display |= nucleic
                if nucleic:
                    from chimerax.nucleotides.cmd import nucleotides
                    if len(nucleic) <= SLAB_THRESHOLD:
                        nucleotides(self.session, 'tube/slab', objects=nucleic, create_undo=False)
                    elif len(nucleic) <= LADDER_THRESHOLD:
                        nucleotides(self.session, 'ladder', objects=nucleic, create_undo=False)
                    from .colors import nucleotide_colors
                    nucleic.ring_colors = nucleotide_colors(nucleic)[0]
                if ligand:
                    # show residues interacting with ligand
                    lig_points = ligand.atoms.coords
                    mol_points = atoms.coords
                    from chimerax.geometry import find_close_points
                    close_indices = find_close_points(lig_points, mol_points, 3.6)[1]
                    display |= atoms.filter(close_indices).residues
                display_atoms = display.atoms
                if self.num_residues > 1:
                    display_atoms = display_atoms.filter(display_atoms.idatm_types != "HC")
                display_atoms.displays = True
                ribbonable.ribbon_displays = True
        elif style == "medium polymer":
            lighting = {'preset': 'full'}
            if self.num_atoms >= MULTI_SHADOW_THRESHOLD:
                lighting['multi_shadow'] = MULTI_SHADOW
            from .colors import chain_colors, element_colors, polymer_colors
            residues = self.residues
            nseq = len(residues.unique_sequences[0])
            if nseq <= 2:
                # Only one polymer sequence.  Sequence 0 is for non-polymers.
                rcolors = chain_colors(residues.chain_ids)
                acolors = chain_colors(atoms.residues.chain_ids)
            else:
                rcolors = polymer_colors(residues)[0]
                acolors = polymer_colors(atoms.residues)[0]
            residues.ribbon_colors = residues.ring_colors = rcolors
            atoms.colors = acolors
            from .molobject import Atom
            ligand_atoms = atoms.filter(atoms.structure_categories == "ligand")
            ligand_atoms.draw_modes = Atom.STICK_STYLE
            ligand_atoms.colors = element_colors(ligand_atoms.element_numbers)
            solvent_atoms = atoms.filter(atoms.structure_categories == "solvent")
            solvent_atoms.draw_modes = Atom.BALL_STYLE
            solvent_atoms.colors = element_colors(solvent_atoms.element_numbers)
        else:
            residues = self.residues
            nseq = len(residues.unique_sequences[0])
            if nseq > 2:
                # More than one sequence (sequence 0 is for non-polymers)
                from .colors import polymer_colors
                rcolors = polymer_colors(residues)[0]
                acolors = polymer_colors(atoms.residues)[0]
                residues.ribbon_colors = residues.ring_colors = rcolors
                atoms.colors = acolors
            # since this is now available as a preset, allow for possibly a smaller number of atoms
            lighting = {'preset': 'soft'}
            if self.num_atoms >= MULTI_SHADOW_THRESHOLD:
                lighting['multi_shadow'] = MULTI_SHADOW

        # correct the styling of per-structure pseudobond bond groups
        for cat, pbg in self.pbg_map.items():
            if cat == self.PBG_METAL_COORDINATION:
                color = self.default_metal_coordination_color
                radius = self.default_metal_coordination_radius
                dashes = self.default_metal_coordination_dashes
            elif cat == self.PBG_MISSING_STRUCTURE:
                color = self.default_missing_structure_color
                radius = self.default_missing_structure_radius
                dashes = self.default_missing_structure_dashes
            elif cat == self.PBG_HYDROGEN_BONDS:
                color = self.default_hbond_color
                radius = self.default_hbond_radius
                dashes = self.default_hbond_dashes
            else:
                continue
            pbg.color = color.uint8x4()
            pbg.radius = radius
            pbg.dashes = dashes

        if set_lighting:
            from chimerax.std_commands.lighting import lighting as light_cmd
            light_cmd(self.session, **lighting)

    def take_snapshot(self, session, flags):
        data = {
            'AtomicStructure version': 3,
            'structure state': Structure.take_snapshot(self, session, flags),
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = AtomicStructure(session, auto_style = False, log_info = False)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        version = data.get('AtomicStructure version', 1)
        if version == 1:
            Structure.set_state_from_snapshot(self, session, data)
        else:
            Structure.set_state_from_snapshot(self, session, data['structure state'])
            if version < 3:
                self.set_custom_attrs(data)

    def _determine_het_res_descriptions(self, session):
        # Don't actually set the description in the residue in order to avoid having
        # to create all the residue objects; just determine the descriptions to
        # be looked up later on demand
        hnd = self._hetnam_descriptions = {}
        recs = self.metadata.get('HETNAM', []) + self.metadata.get('HETSYN', [])
        alternatives = {}
        for rec in recs:
            if rec[8:10].strip():
                # continuation
                hnd[het] = hnd[het] + rec[15:].strip()
            else:
                het = rec[11:14].strip()
                if het in hnd:
                    alternatives[het] = hnd[het]
                hnd[het] = rec[15:].strip()
        # use "punchier" description :-)
        for het, alt in alternatives.items():
            if len(alt) < len(hnd[het]):
                hnd[het] = alt
        from chimerax.pdb import process_chem_name
        for k, v in hnd.items():
            hnd[k] = process_chem_name(v)

    def _set_chain_descriptions(self, session):
        from chimerax import mmcif
        chain_to_desc = {}
        struct_asym, entity = mmcif.get_mmcif_tables_from_metadata(self, ['struct_asym', 'entity'])
        if struct_asym:
            if not entity:
                # bad mmCIF file
                return
            try:
                mmcif_chain_to_entity = struct_asym.mapping('id', 'entity_id')
                entity_to_description = entity.mapping('id', 'pdbx_description')
            except ValueError:
                pass
            else:
                for ch in self.chains:
                    mmcif_cid = ch.existing_residues.mmcif_chain_ids[0]
                    try:
                        chain_to_desc[ch.chain_id] = (
                            entity_to_description[mmcif_chain_to_entity[mmcif_cid]], False)
                    except KeyError:
                        pass  # ignore bad metadata
        elif 'COMPND' in self.metadata and self.pdb_version > 1:
            compnd_recs = self.metadata['COMPND']
            compnd_chain_ids = None
            description = ""
            continued = False
            for rec in compnd_recs:
                if continued:
                    v += " " + rec[10:].strip()
                else:
                    try:
                        k, v = rec[10:].strip().split(": ", 1)
                    except ValueError:
                        # bad PDB file
                        break
                if v.endswith(';'):
                    v = v[:-1]
                    continued = False
                elif rec == compnd_recs[-1]:
                    continued = False
                else:
                    continued = True
                    continue
                if k == "MOL_ID":
                    if compnd_chain_ids and description:
                        for chain_id in compnd_chain_ids:
                            chain_to_desc[chain_id] = (description, synonym)
                    compnd_chain_ids = None
                    description = ""
                elif k == "MOLECULE":
                    if v.startswith("PROTEIN (") and v.endswith(")"):
                        description = v[9:-1]
                    else:
                        description = v
                    synonym = False
                elif k == "SYNONYM":
                    if ',' not in v:
                        # not a long list of synonyms
                        description = v
                        synonym = True
                elif k == "CHAIN":
                    compnd_chain_ids = v.split(", ")
            if compnd_chain_ids and description:
                for chain_id in compnd_chain_ids:
                    chain_to_desc[chain_id] = (description, synonym)
        if chain_to_desc:
            from chimerax.pdb import process_chem_name
            for k, v in chain_to_desc.items():
                description, synonym = v
                chain_to_desc[k] = process_chem_name(description, probable_abbrs=synonym)
            chains = sorted(self.chains, key=lambda c: c.chain_id)
            for chain in chains:
                chain.description = chain_to_desc.get(chain.chain_id, None)

    def _report_chain_descriptions(self, session):
        chains = sorted(self.chains, key=lambda c: c.chain_id)
        if not chains:
            return
        from collections import OrderedDict
        descripts = OrderedDict()
        for chain in chains:
            description = chain.description if chain.description else "No description available"
            descripts.setdefault((description, chain.characters), []).append(chain)
        def chain_text(chain):
            return '<a title="Select chain" href="cxcmd:select %s">%s</a>' % (
               chain_res_range(chain), (chain.chain_id if not chain.chain_id.isspace() else '?'))
        self._report_chain_summary(session, descripts, chain_text, False)

    def _report_ensemble_chain_descriptions(self, session, ensemble):
        from .molarray import AtomicStructures
        structs = AtomicStructures(ensemble)
        chains = sorted(structs.chains, key=lambda c: c.chain_id)
        if not chains:
            return
        from collections import OrderedDict
        descripts = OrderedDict()
        for chain in chains:
            description = chain.description if chain.description else "No description available"
            descripts.setdefault((description, chain.characters), []).append(chain)
        def chain_text(chain):
            return '<a title="Select chain" href="cxcmd:select %s">%s/%s</a>' % (chain_res_range(chain),
                chain.structure.id_string, (chain.chain_id if not chain.chain_id.isspace() else '?'))
        self._report_chain_summary(session, descripts, chain_text, True)

    def _report_model_info(self, session):
        # report Model Archive info [#5601]
        from chimerax.mmcif import get_mmcif_tables_from_metadata
        align_data, template_deets, template_segment, scoring_metrics, local_scores = \
            get_mmcif_tables_from_metadata(self, ['ma_alignment', 'ma_template_ref_db_details',
            'ma_template_poly_segment', 'ma_qa_metric', 'ma_qa_metric_local'])
        if local_scores and scoring_metrics:
            from chimerax.core.attributes import string_to_attr
            scoring_metric_cache = {}
            chain_cache = {}
            res_scoring = []
            metric_names = scoring_metrics.mapping('id', 'name')
            for chain_id, res_name, seq_id, metric_id, value in local_scores.fields(
                    ['label_asym_id', 'label_comp_id', 'label_seq_id', 'metric_id', 'metric_value']):
                try:
                    chain = chain_cache[chain_id]
                except KeyError:
                    for chain in self.chains:
                        if chain.chain_id == chain_id:
                            chain_cache[chain_id] = chain
                            break
                    else:
                        session.logger.warning("No chain in structure corresponds to chain ID given"
                            " in local score info (chain '%s')" % chain_id)
                        break
                res = chain.residues[int(seq_id)-1]
                if not res:
                    continue
                if res.name != res_name:
                    session.logger.warning("Residue name for residue %s in chain %s (%s) does not correspond"
                        " to name in local score info (%s)" % (seq_id, chain_id, res.name, res_name))
                    break
                try:
                    metric_name, metric_attr = scoring_metric_cache[metric_id]
                except KeyError:
                    try:
                        metric_name = metric_names[metric_id]
                    except KeyError:
                        session.logger.warning("No scoring metric with ID '%s'" % metric_id)
                        break
                    metric_attr = string_to_attr(metric_name) + '_score'
                    scoring_metric_cache[metric_id] = (metric_name, metric_attr)
                try:
                    value = float(value)
                except ValueError:
                    session.logger.warning("Value for metric '%s' is non-numeric ('%s')"
                        % (metric_name, value))
                    break
                res_scoring.append((res, metric_attr, value))
            else:
                # everything worked
                for res, attr_name, value in res_scoring:
                    setattr(res, attr_name, value)
                from chimerax.atomic import Residue
                for metric_name, metric_attr in scoring_metric_cache.values():
                    Residue.register_attr(session, metric_attr, "Local model scoring", attr_type=float)
                    session.logger.info('<a href="cxcmd:color byattribute r:%s %s palette red:yellow:green">'
                        'Color</a> %s by residue' ' <a href="help:user/attributes.html">attribute</a> %s'
                        % (metric_attr, self.atomspec, self.name, metric_attr), is_html=True)

        if not align_data:
            return
        template_names = {}
        if template_deets:
            for template_id, db_name, db_accession_code in template_deets.fields(
                    ['template_id', 'db_name', 'db_accession_code']):
                template_names[template_id] = "%s %s" % (db_name, db_accession_code)
        # since the chain IDs provided are not the author IDs, don't add them into the template sequence
        # name since it will just be confusing to the user unless we use some kind of web lookup to
        # resolve them to author IDs
        """
        try:
            template_details_headers = self.metadata['ma_template_details']
            template_details = self.metadata['ma_template_details data']
        except KeyError:
            pass
        else:
            if len(template_details_headers) != 11:
                session.warning("Don't know how to parse model template detail information")
            else:
                for i in range(0, len(template_details), 10):
                    template_id, template_cid = template_details[i+1], template_details[i+7]
                    try:
                        template_names[template_id] += " /%s" % template_cid
                    except KeyError:
                        session.warning("Unknown template ID in detail information: %s" % template_id)
        """
        if template_segment:
            for template_id, begin, end in template_segment.fields(
                    ['template_id', 'residue_number_begin', 'residue_number_end']):
                try:
                    template_names[template_id] += ":%s-%s" % (begin, end)
                except KeyError:
                    session.warning("Unknown template ID in residue-range information: %s" % template_id)
        cur_align = None
        seqs =[]
        from . import Sequence
        for alignment_id, target_template, seq in align_data.fields(
                ['alignment_id', 'target_template_flag', 'sequence']):
            if cur_align != alignment_id:
                if cur_align is not None:
                    session.alignments.new_alignment(seqs, None, name="target-template alignment")
                    seqs = []
                cur_align = alignment_id
            # Since the alignment data does not include a template_id, if only one template is given
            # then base the name on that, otherwise just use "template".  See issue:
            # https://github.com/ihmwg/MA-dictionary/issues/4
            if target_template == '1':
                seq_name = "target"
            elif len(template_names) == 1:
                seq_name = list(template_names.values())[0]
            else:
                seq_name = "template"
            seqs.append(Sequence(name=seq_name, characters=seq))
        if cur_align is not None:
            session.alignments.new_alignment(seqs, None, name="target-template alignment")
        # have to hold a reference to the timer
        self._timer = session.ui.timer(500, session.logger.status,
            'Use "more info..." link in log to see overall model scores [if any]', color="forest green")

    def _report_res_info(self, session):
        if hasattr(self, 'get_formatted_res_info'):
            res_info = self.get_formatted_res_info(standalone=True)
            if res_info:
                session.logger.info(res_info, is_html=True)

    def _report_chain_summary(self, session, descripts, chain_text, is_ensemble):
        def descript_text(description, chains):
            from html import escape
            return '<a title="Show sequence" href="cxcmd:sequence chain %s">%s</a>' % (
                ''.join([chain.string(style="command", include_structure=True)
                    for chain in chains]), escape(description))
        uids = uniprot_ids(self)
        uchains = set(uid.chain_id for uid in uids)
        have_uniprot_ids = len([chain for chains in descripts.values()
                                for chain in chains if chain.chain_id in uchains]) > 0
        from chimerax.core.logger import html_table_params
        struct_name = self.name if is_ensemble else str(self)
        lines = ['<table %s>' % html_table_params,
                 '  <thead>',
                 '    <tr>',
                 '      <th colspan="%d">Chain information for %s</th>'
                   % ((3 if have_uniprot_ids else 2), struct_name),
                 '    </tr>',
                 '    <tr>',
                 '      <th>Chain</th>',
                 '      <th>Description</th>',
                 '      <th>UniProt</th>' if have_uniprot_ids else '',
                 '    </tr>',
                 '  </thead>',
                 '  <tbody>',
        ]
        for key, chains in descripts.items():
            description, characters = key
            cids = ' '.join([chain_text(chain) for chain in chains])
            cdescrip = descript_text(description, chains)
            if have_uniprot_ids:
                cuids = uniprot_chain_descriptions(uids, chains)
            lines.extend([
                '    <tr>',
                '      <td style="text-align:center">' + cids + '</td>',
                '      <td>' + cdescrip + '</td>',
                (('      <td style="text-align:center">' + cuids + '</td>')
                 if have_uniprot_ids else ''),
                '    </tr>',
            ])
        lines.extend(['  </tbody>',
                      '</table>'])
        summary = '\n'.join(lines)
        session.logger.info(summary, is_html=True)

    def _report_assemblies(self, session):
        if getattr(self, 'ignore_assemblies', False):
            return
        html = assembly_html_table(self)
        if html:
            session.logger.info(html, is_html=True)

    def show_info(self):
        from chimerax.core.commands import run, concise_model_spec
        spec = concise_model_spec(self.session, [self], allow_empty_spec=False, relevant_types=AtomicStructure)
        if assembly_html_table(self):
            base_cmd = "sym %s; " % spec
        else:
            base_cmd = ""
        run(self.session, base_cmd + "log metadata %s; log chains %s" % (spec, spec))


# also used by model panel to determine if its "Info" button should issue a "sym" command...
def assembly_html_table(mol):
    '''HTML table listing assemblies using info from metadata instead of reparsing mmCIF file.'''
    from chimerax import mmcif
    sat = mmcif.get_mmcif_tables_from_metadata(mol, ['pdbx_struct_assembly'])[0]
    sagt = mmcif.get_mmcif_tables_from_metadata(mol, ['pdbx_struct_assembly_gen'])[0]
    if not sat or not sagt:
        return None

    try:
        sa = sat.fields(('id', 'details'))
        sag = sagt.mapping('assembly_id', 'oper_expression')
    except ValueError:
        return	None # Tables do not have required fields

    if len(sa) == 1 and sag.get(sa[0][0]) == '1':
        # Probably just have the identity assembly, so don't show table.
        # Should check that it is the identity operator and all
        # chains are transformed. Requires reading more tables.
        return

    lines = ['<table border=1 cellpadding=4 cellspacing=0 bgcolor="#f0f0f0">',
             '<tr><th colspan=2>%s mmCIF Assemblies' % mol.name]
    for id, details in sa:
        lines.append('<tr><td><a title="Generate assembly" href="cxcmd:sym #%s assembly %s ; view">%s</a><td>%s'
                     % (mol.id_string, id, id, details))
    lines.append('</table>')
    html = '\n'.join(lines)
    return html

def chain_res_range(chain):
    existing = chain.existing_residues
    if len(existing) == 1:
        return existing[0].string(style="command")
    def range_string(first, last, first_res_only=False):
        if first == last:
            return first.string(residue_only=first_res_only, style="command")
        return "%s-%s" % (first.string(residue_only=first_res_only, style="command"),
            last.string(residue_only=True, style="command")[1:])
    first, last = existing[0], existing[-1]
    if first.number < last.number:
        return range_string(first, last)
    # circular permutation, do something more elaborate
    ranges = []
    cur_num = None
    for r in existing:
        if cur_num is None:
            start_res = end_res = r
        elif r.number < cur_num:
            ranges.append((start_res, end_res))
            start_res = end_res = r
        else:
            end_res = r
        cur_num = r.number
    ranges.append((start_res, end_res))
    return range_string(*ranges[0], first_res_only=False) + ',' + ','.join(
        [range_string(first, last, first_res_only=True)[1:] for first, last in ranges[1:]])

def uniprot_chain_descriptions(uids, chains):

    if len(chains) == 0:
        return ''
    
    # Group uniport ids with different sequence ranges.
    uranges = {}
    chain_ids = set(chain.chain_id for chain in chains)
    for uid in uids:
        if uid.chain_id in chain_ids:
            if uid.uniprot_id in uranges:
                uranges[uid.uniprot_id].append(uid)
            else:
                uranges[uid.uniprot_id] = [uid]

    # Make a link for each Uniprot id and list sequence ranges
    descrips = []
    ucmd = '<a title="Show annotations" href="cxcmd:open %s from uniprot associate %s">%s</a>'
    cspec = f'#{chains[0].structure.id_string}/{",".join(sorted(chain_ids))}'
    scmd = f'<a title="Select sequence" href="cxcmd:select {cspec}:%d-%d">%d-%d</a>'
    for ruids in uranges.values():
        uid = ruids[0]
        utext = uid.uniprot_name if uid.uniprot_name else uid.uniprot_id
        # ensure chain specifier alway includes model ID
        descrip = ucmd % (uid.uniprot_id, chains[0].structure.atomspec + '/' + ','.join(
            [c.chain_id for c in chains]), utext)
        seq_ranges = set(tuple(uid.chain_sequence_range)
                         for uid in ruids if uid.chain_sequence_range)
        if seq_ranges:
            descrip += ' ' + ' '.join(scmd % (s,e,s,e) for s,e in sorted(seq_ranges))
        descrips.append(descrip)
        
    return ', '.join(descrips)


# -----------------------------------------------------------------------------
# Before each redraw this singleton object gets a graphics update trigger and
# checks the C++ graphics changed flags for all structures, updating the graphics
# drawings for those structures if needed.  Also it updates the level of detail
# for atom spheres and bond cylinders.
#
class StructureGraphicsChangeManager:
    def __init__(self, session):
        self.session = session
        t = session.triggers
        self._handler = t.add_handler('graphics update', self._update_graphics_if_needed)
        self._structures = set()
        self._structures_array = None		# StructureDatas object
        self.num_atoms_shown = 0
        self.level_of_detail = LevelOfDetail()
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        self._display_handler = t.add_handler(MODEL_DISPLAY_CHANGED, self._model_display_changed)
        self._model_display_change = False

    def __del__(self):
        self.session.triggers.remove_handler(self._handler)
        self.session.triggers.remove_handler(self._display_handler)

    @property
    def structures(self):
        return self._structures
    
    def add_structure(self, s):
        self._structures.add(s)
        self._structures_array = None
        self.num_atoms_shown = 0	# Make sure new structure gets a level of detail update

    def remove_structure(self, s):
        self._structures.remove(s)
        self._structures_array = None

    def _model_display_changed(self, tname, model):
        if isinstance(model, Structure) or _has_structure_descendant(model):
            self._model_display_change = True

    def _update_graphics_if_needed(self, *_):
        s = self._array()
        gc = s._graphics_changeds	# Includes pseudobond group changes.
        if self._model_display_change or gc.any():
            # Update graphics for each changed structure
            for i in gc.nonzero()[0]:
                s[i].update_graphics_if_needed()

            # Update level of detail if number of atoms shown changed.
            if self._model_display_change or (gc & StructureData._SHAPE_CHANGE).any():
                n = sum(m.num_atoms_visible * m.num_displayed_positions
                        for m in s if m.visible)
                if n > 0 and n != self.num_atoms_shown:
                    self.num_atoms_shown = n
                    self.update_level_of_detail()

            self._model_display_change = False

        # set by changes.py when "selected changed" is in the global Atom reasons,
        # which is the easiest way to detect that there was a selection in a
        # deleted structure; also set by non-atomic models
        if self.session.selection.trigger_fire_needed:
            self.session.selection.trigger_fire_needed = False
            from chimerax.core.selection import SELECTION_CHANGED
            self.session.triggers.activate_trigger(SELECTION_CHANGED, None)

    def update_level_of_detail(self):
        n = self.num_atoms_shown
        for m in self._structures:
            if m.display:
                m._update_level_of_detail(n)

    def set_ribbon_divisions(self, divisions):
        self.level_of_detail.ribbon_fixed_divisions = divisions
        self._update_ribbons()

    def _update_ribbons(self):
        for m in self._structures:
            m._graphics_changed |= m._RIBBON_CHANGE

    def _array(self):
        sa = self._structures_array
        if sa is None:
            from .molarray import StructureDatas, object_pointers
            self._structures_array = sa = StructureDatas(object_pointers(self._structures))
        return sa

    def set_quality(self, quality):
        lod = self.level_of_detail
        lod.quality = quality
        lod.atom_fixed_triangles = None
        lod.bond_fixed_triangles = None
        lod.ribbon_fixed_divisions = None
        self.update_level_of_detail()
        self._update_ribbons()

# -----------------------------------------------------------------------------
#
def structure_graphics_updater(session):
    gu = getattr(session, '_structure_graphics_updater', None)
    if gu is None:
        session._structure_graphics_updater = gu = StructureGraphicsChangeManager(session)
    return gu

# -----------------------------------------------------------------------------
#
def level_of_detail(session):
    gu = structure_graphics_updater(session)
    return gu.level_of_detail

# -----------------------------------------------------------------------------
#
class LevelOfDetail(State):

    def __init__(self, restore_data=None):
        # if restore_data is not None:
        #     self.quality = restore_data[0]
        # else:
        self.quality = 1

        # Number of triangles used for an atom sphere.
        self._atom_min_triangles = 10
        self._atom_max_triangles = 2000
        self._atom_default_triangles = 200
        self._atom_max_total_triangles = 5000000
        self._step_factor = 1.2
        self.atom_fixed_triangles = None	# If not None use fixed number of triangles
        self._sphere_geometries = {}	# Map ntri to (va,na,ta)

        # Number of triangles used for a bond cylinder.
        self._bond_min_triangles = 24
        self._bond_max_triangles = 160
        self._bond_default_triangles = 60
        self._bond_max_total_triangles = 5000000
        self.bond_fixed_triangles = None	# If not None use fixed number of triangles
        self._cylinder_geometries = {}	# Map ntri to (va,na,ta)

        # Number of cylinder sides for pseudobonds
        self._pseudobond_sides = 10
        
        # Number of bands between two residues along the length of a ribbon.
        self._ribbon_min_divisions = 2
        self._ribbon_max_divisions = 20
        self._ribbon_residue_count_best = 20000	# Use max divisions for fewer residues
        self.ribbon_fixed_divisions = None
        
    def take_snapshot(self, session, flags):
        return {'quality': self.quality,
                'version': 1}

    def _get_total_atom_triangles(self):
        return self._atom_max_total_triangles
    def _set_total_atom_triangles(self, ntri):
        self._atom_max_total_triangles = ntri
    total_atom_triangles = property(_get_total_atom_triangles, _set_total_atom_triangles)

    def _get_total_bond_triangles(self):
        return self._bond_max_total_triangles
    def _set_total_bond_triangles(self, ntri):
        self._bond_max_total_triangles = ntri
    total_bond_triangles = property(_get_total_bond_triangles, _set_total_bond_triangles)

    def _get_pseudobond_sides(self):
        return self._pseudobond_sides
    def _set_pseudobond_sides(self, sides):
        self._pseudobond_sides = sides
    pseudobond_sides = property(_get_pseudobond_sides, _set_pseudobond_sides)
    
    @staticmethod
    def restore_snapshot(session, data):
        lod = LevelOfDetail()
        lod.quality = data['quality']
        return lod

    def set_atom_sphere_geometry(self, drawing, natoms = None):
        if natoms == 0:
            return
        ntri = self.atom_sphere_triangles(natoms)
        ta = drawing.triangles
        if ta is None or len(ta) != ntri:
            # Update instanced sphere triangulation
            w = len(ta) if ta is not None else 0
            va, na, ta = self.sphere_geometry(ntri)
            drawing.set_geometry(va, na, ta)

    def sphere_geometry(self, ntri):
        # Cache sphere triangulations of different sizes.
        sg = self._sphere_geometries
        if not ntri in sg:
            from chimerax.geometry.sphere import sphere_triangulation
            va, ta = sphere_triangulation(ntri)
            sg[ntri] = (va,va,ta)
        return sg[ntri]

    def atom_sphere_triangles(self, natoms):
        aft = self.atom_fixed_triangles
        if aft is not None:
            ntri = aft
        elif natoms is None or natoms == 0:
            ntri = self._atom_default_triangles
        else:
            ntri = self.quality * self._atom_max_total_triangles // natoms
            nmin, nmax = self._atom_min_triangles, self._atom_max_triangles
            ntri = self.clamp_geometric(ntri, nmin, nmax)
        ntri = 2*(ntri//2)	# Require multiple of 2.
        return ntri

    def clamp_geometric(self, n, nmin, nmax):
        f = self._step_factor
        from math import log, pow
        n1 = int(nmin*pow(f,int(log(max(n,nmin)/nmin,f))))
        n2 = min(n1, nmax)
        n3 = max(n2, nmin)
        return n3

    def set_bond_cylinder_geometry(self, drawing, nbonds = None):
        if nbonds == 0:
            return
        ntri = self.bond_cylinder_triangles(nbonds)
        ta = drawing.triangles
        if ta is None or len(ta) != ntri//2:
            # Update instanced cylinder triangulation.
            # Since halfbond mode makes two cylinders per bond we use ntri/2
            # per cylinder, or ntri/4 cylinder sides.
            w = len(ta) if ta is not None else 0
            va, na, ta = self.cylinder_geometry(div = ntri//4)
            drawing.set_geometry(va, na, ta)

    def cylinder_geometry(self, div):
        # Cache cylinder triangulations of different sizes.
        cg = self._cylinder_geometries
        if not div in cg:
            from chimerax import surface
            cg[div] = surface.cylinder_geometry(nc = div, caps = False, height = 0.5)
        return cg[div]

    def bond_cylinder_triangles(self, nbonds):
        bft = self.bond_fixed_triangles
        if bft is not None:
            ntri = bft
        elif nbonds is None or nbonds == 0:
            ntri = self._bond_default_triangles
        else:
            ntri = self.quality * self._bond_max_total_triangles // nbonds
            nmin, nmax = self._bond_min_triangles, self._bond_max_triangles
            ntri = self.clamp_geometric(ntri, nmin, nmax)
        ntri = 4*(ntri//4)	# Require multiple of 4
        return ntri

    def ribbon_divisions(self, num_residues):
        div = self.ribbon_fixed_divisions
        if div is not None:
            return div
        f = num_residues / self._ribbon_residue_count_best
        dmin, dmax = self._ribbon_min_divisions, self._ribbon_max_divisions
        div = int(self.quality * (dmax if f <= 1 else dmax / f))
        if div < dmin:
            div = dmin
        elif div > dmax:
            div = dmax
        return div
    
# -----------------------------------------------------------------------------
#
from chimerax.core.selection import SelectionPromotion
class PromoteAtomSelection(SelectionPromotion):
    def __init__(self, structure, level, atom_sel_mask, prev_atom_sel_mask, prev_bond_sel_mask):
        SelectionPromotion.__init__(self, level)
        self._structure = structure
        self._atom_sel_mask = atom_sel_mask
        self._prev_atom_sel_mask = prev_atom_sel_mask
        self._prev_bond_sel_mask = prev_bond_sel_mask
    def promote(self):
        s = self._structure
        if s.deleted:
            return
        atoms = s.atoms
        asel = self._atom_sel_mask
        if len(atoms) != len(asel):
            return	# Atoms added or deleted
        atoms.selected = asel
        atoms[asel].intra_bonds.selected = True
    def demote(self):
        s = self._structure
        if s.deleted:
            return
        if (s.num_atoms != len(self._prev_atom_sel_mask) or
            s.num_bonds != len(self._prev_bond_sel_mask)):
            return   # Atoms or bonds deleted or added.
        s.atoms.selected = self._prev_atom_sel_mask
        s.bonds.selected = self._prev_bond_sel_mask

# -----------------------------------------------------------------------------
#
class PickedAtom(Pick):
    def __init__(self, atom, distance):
        Pick.__init__(self, distance)
        self.atom = atom
    def description(self):
        return str(self.atom)
    def specifier(self):
        # have to do something fancy for the rare case of duplicate atom specs [#4617]
        residues = self.atom.structure.residues
        same_numbers = residues.numbers == self.atom.residue.number
        import numpy
        if numpy.count_nonzero(same_numbers) > 1:
            same_chains = residues.chain_ids == self.atom.residue.chain_id
            same = numpy.logical_and(same_numbers, same_chains)
            if numpy.count_nonzero(same) > 1:
                same_inserts = residues.insertion_codes == self.atom.residue.insertion_code
                same = numpy.logical_and(same, same_inserts)
                if numpy.count_nonzero(same) > 1:
                    # Well, I could go off and see if the other residue(s) contain an atom
                    # with the same name, but at this point I'm going to be lazy...
                    base = '@@serial_number=%d' % self.atom.serial_number
                    if len([s for s in self.atom.structure.session.models if isinstance(s, Structure)]) == 1:
                        return base
                    return self.atom.structure.string(style="command") + base
        return self.atom.string(style='command')
    @property
    def residue(self):
        return self.atom.residue
    def select(self, mode = 'add'):
        select_atom(self.atom, mode)
    def selected(self):
        return self.atom.selected
    def drawing(self):
        return self.atom.structure
    
# -----------------------------------------------------------------------------
#
def select_atom(a, mode = 'add'):
    if mode == 'add':
        s = True
    elif mode == 'subtract':
        s = False
    elif mode == 'toggle':
        s = not a.selected
    a.selected = s

# -----------------------------------------------------------------------------
#
class PickedAtoms(Pick):
    def __init__(self, atoms):
        Pick.__init__(self)
        self.atoms = atoms
    def description(self):
        return '%d atoms' % len(self.atoms)
    @property
    def residue(self):
        rs = self.atoms.unique_residues
        if len(rs) == 1:
            for res in residues:
                return res
        return None
    def select(self, mode = 'add'):
        select_atoms(self.atoms, mode)

# -----------------------------------------------------------------------------
#
def select_atoms(a, mode = 'add'):
    if mode == 'add':
        s = True
    elif mode == 'subtract':
        s = False
    elif mode == 'toggle':
        from numpy import logical_not
        s = logical_not(a.selected)
    a.selected = s

# -----------------------------------------------------------------------------
# Handles bonds and pseudobonds.
#
def _bond_intercept(bonds, mxyz1, mxyz2, scene_coordinates = False):

    bshown = bonds.showns
    bs = bonds.filter(bshown)
    a1, a2 = bs.atoms
    from .molarray import Pseudobonds
    if isinstance(bonds, Pseudobonds):
        # End positions may be projected onto ribbons.
        if scene_coordinates:
            cxyz1, cxyz2 = (a1.pb_scene_coords, a2.pb_scene_coords)
        else:
            cxyz1, cxyz2 = (a1.pb_coords, a2.pb_coords)
    else:
        if scene_coordinates:
            cxyz1, cxyz2 = (a1.scene_coords, a2.scene_coords)
        else:
            cxyz1, cxyz2 = (a1.coords, a2.coords)
    r = bs.radii

    # Check for atom sphere intercept
    from chimerax import geometry
    f, bnum = geometry.closest_cylinder_intercept(cxyz1, cxyz2, r, mxyz1, mxyz2)

    if f is None:
        return None, None

    # Remap index to include undisplayed positions
    bnum = bshown.nonzero()[0][bnum]

    return bonds[bnum], f

# -----------------------------------------------------------------------------
#
def _bonds_planes_pick(drawing, planes):
    if drawing is None or not drawing.display:
        return None

    hb_xyz = drawing.positions.array()[:,:,3]	# Half-bond centers
    n = len(hb_xyz)//2
    xyz = 0.5*(hb_xyz[:n] + hb_xyz[n:])	# Bond centers
    from chimerax import geometry
    pmask = geometry.points_within_planes(xyz, planes)
    return pmask

# -----------------------------------------------------------------------------
#
class PickedBond(Pick):
    def __init__(self, bond, distance):
        Pick.__init__(self, distance)
        self.bond = bond
    def description(self):
        dist_fmt = self.bond.session.pb_dist_monitor.distance_format
        return str(self.bond) + " " + dist_fmt % self.bond.length
    @property
    def residue(self):
        a1, a2 = self.bond.atoms
        if a1.residue == a2.residue:
            return a1.residue
        return None
    def select(self, mode = 'add'):
        select_bond(self.bond, mode)
    def drawing(self):
        return self.bond.structure

# -----------------------------------------------------------------------------
#
def select_bond(b, mode = 'add'):
    if mode == 'add':
        s = True
    elif mode == 'subtract':
        s = False
    elif mode == 'toggle':
        s = not b.selected
    b.selected = s

# -----------------------------------------------------------------------------
#
class PickedBonds(Pick):
    def __init__(self, bonds):
        Pick.__init__(self)
        self.bonds = bonds
    def description(self):
        return '%d bonds' % len(self.bonds)
    def select(self, mode = 'add'):
        select_bonds(self.bonds, mode)

# -----------------------------------------------------------------------------
#
def select_bonds(b, mode = 'add'):
    if mode == 'add':
        s = True
    elif mode == 'subtract':
        s = False
    elif mode == 'toggle':
        from numpy import logical_not
        s = logical_not(b.selected)
    b.selected = s

# -----------------------------------------------------------------------------
#
class PickedPseudobond(Pick):
    def __init__(self, pbond, distance):
        Pick.__init__(self, distance)
        self.pbond = pbond
    def description(self):
        dist_fmt = self.pbond.session.pb_dist_monitor.distance_format
        return str(self.pbond) + " " + dist_fmt % self.pbond.length
    def drawing(self):
        return self.pbond.group
    @property
    def residue(self):
        a1, a2 = self.pbond.atoms
        if a1.residue == a2.residue:
            return a1.residue
        return None
    def select(self, mode = 'add'):
        select_bond(self.pbond, mode)
        pbg = self.pbond.group
        pbg._graphics_changed |= pbg._SELECT_CHANGE

# -----------------------------------------------------------------------------
#
class PickedPseudobonds(Pick):
    def __init__(self, pbonds):
        Pick.__init__(self)
        self.pseudobonds = pbonds
    def description(self):
        return '%d pseudobonds' % len(self.pseudobonds)
    def select(self, mode = 'add'):
        select_bonds(self.pseudobonds, mode)

# -----------------------------------------------------------------------------
#
class PickedResidue(Pick):
    def __init__(self, residue, distance):
        Pick.__init__(self, distance)
        self.residue = residue
    def description(self):
        return str(self.residue)
    def specifier(self):
        return self.residue.string(style='command')
    def select(self, mode = 'add'):
        a = self.residue.atoms
        if mode == 'add':
            a.selected = True
        elif mode == 'subtract':
            a.selected = False
        elif mode == 'toggle':
            a.selected = not a.selected.any()
    def selected(self):
        return self.residue.atoms.selected.any()
    def drawing(self):
        return self.residue.structure

# -----------------------------------------------------------------------------
#
class PickedResidues(Pick):
    def __init__(self, residues):
        Pick.__init__(self)
        self.residues = residues
    def description(self):
        return '%d residues' % len(self.residues)
    @property
    def residue(self):
        if len(self.residues) == 1:
            for res in self.residues:
                return res
        return None
    def select(self, mode = 'add'):
        select_atoms(self.residues.atoms, mode)

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking one prototype cylinder to each bond location.
#
def _bond_cylinder_placements(axyz0, axyz1, radii):

  n = len(axyz0)
  from numpy import empty, float32
  p = empty((n,4,4), float32)

  from chimerax.geometry import cylinder_rotations
  cylinder_rotations(axyz0, axyz1, radii, p)

  p[:,3,:3] = 0.5*(axyz0 + axyz1)

  from chimerax.geometry import Places
  pl = Places(opengl_array = p)
  return pl

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking two prototype cylinders to each bond location.
#
def _halfbond_cylinder_placements(axyz0, axyz1, radii, parray = None):

  n = len(axyz0)
  if parray is None or len(parray) != 2*n:
      from numpy import empty, float32
      p = empty((2*n,4,4), float32)
  else:
      p = parray

  from chimerax.geometry import half_cylinder_rotations
  half_cylinder_rotations(axyz0, axyz1, radii, p)

  from chimerax.geometry import Places
  pl = Places(opengl_array = p)

  return pl

# -----------------------------------------------------------------------------
# Return height, radius, rotation, and translation for each halfbond cylinder.
# Each row is [height, radius, *rotationAxis, rotationAngle, *translation]
#
def _halfbond_cylinder_x3d(axyz0, axyz1, radii):

  n = len(axyz0)
  from numpy import empty, float32
  ci = empty((2 * n, 9), float32)

  from chimerax.geometry import cylinder_rotations_x3d
  cylinder_rotations_x3d(axyz0, axyz1, radii, ci[:n])
  ci[n:, :] = ci[:n, :]

  # Translations
  ci[:n, 6:9] = 0.75 * axyz0 + 0.25 * axyz1
  ci[n:, 6:9] = 0.25 * axyz0 + 0.75 * axyz1

  return ci

# -----------------------------------------------------------------------------
# Display mask for 2 cylinders representing each bond.
#
def _shown_bond_cylinders(bonds):
    sb = bonds.showns
    import numpy
    sb2 = numpy.concatenate((sb,sb))
    return sb2

# -----------------------------------------------------------------------------
# Bond is selected if both atoms are selected.
#
def _selected_bond_cylinders(bonds):
    if bonds.num_selected > 0:
        bsel = bonds.selected
        from numpy import concatenate
        sel = concatenate((bsel,bsel))
    else:
        sel = None
    return sel

# -----------------------------------------------------------------------------
#
def _has_structure_descendant(model):
    for c in model.child_models():
        if c.display and (isinstance(c, Structure) or _has_structure_descendant(c)):
            return True
    return False

# -----------------------------------------------------------------------------
#
def all_atomic_structures(session):
    '''List of all :class:`.AtomicStructure` objects.'''
    from .molarray import AtomicStructures
    return AtomicStructures([m for m in session.models.list() if isinstance(m,AtomicStructure)])

# -----------------------------------------------------------------------------
#
def all_structures(session, atomic_only=False):
    '''List of all :class:`.Structure` objects.'''
    from .molarray import Structures
    class_obj = AtomicStructure if atomic_only else Structure
    return Structures([m for m in session.models.list() if isinstance(m,class_obj)])

# -----------------------------------------------------------------------------
#
def all_atoms(session, atomic_only=False):
    '''All atoms in all structures as an :class:`.Atoms` collection.'''
    structures = all_structures(session, atomic_only=atomic_only)
    from .molarray import concatenate, Atoms
    atoms = concatenate([m.atoms for m in structures], Atoms)
    return atoms

# -----------------------------------------------------------------------------
#
def all_bonds(session, atomic_only=False):
    '''All bonds in all structures as an :class:`.Bonds` collection.'''
    structures = all_structures(session, atomic_only=atomic_only)
    from .molarray import concatenate, Bonds
    bonds = concatenate([m.bonds for m in structures], Bonds)
    return bonds

# -----------------------------------------------------------------------------
#
def all_residues(session, atomic_only=False):
    '''All residues in all structures as a :class:`.Residues` collection.'''
    structures = all_structures(session, atomic_only=atomic_only)
    from .molarray import concatenate, Residues
    residues = concatenate([m.residues for m in structures], Residues)
    return residues

# -----------------------------------------------------------------------------
#
def is_informative_name(name):
    '''Does the string 'name' seem like it would actually be an informative name for the structure'''
    nm = name.strip().lower()
    if "unknown" in nm:
        return False

    for c in nm:
        if c.isalnum():
            return True
    return False

# -----------------------------------------------------------------------------
#
def structure_atoms(structures):
    '''Return all atoms in specified atomic structures as an :class:`.Atoms` collection.'''
    from .molarray import concatenate, Atoms
    atoms = concatenate([m.atoms for m in structures], Atoms)
    return atoms

# -----------------------------------------------------------------------------
#
def selected_atoms(session):
    '''All selected atoms in all structures as an :class:`.Atoms` collection.'''
    from . import changes
    return changes.selected_atoms(session)

# -----------------------------------------------------------------------------
#
def selected_bonds(session, *, intra_residue=True, inter_residue=True):
    '''All selected bonds in all structures as a :class:`.Bonds` collection.'''
    blist = []
    for m in session.models.list(type = Structure):
        for b in m.selected_items('bonds'):
            if inter_residue and intra_residue:
                blist.append(b)
                continue
            import numpy
            atoms1, atoms2 = b.atoms
            # "atoms1.residues == atoms2.residues" returns a scalar boolean, so...
            is_intra = atoms1.residues.pointers == atoms2.residues.pointers
            if intra_residue:
                blist.append(b.filter(is_intra))
            if inter_residue:
                blist.append(b.filter(numpy.logical_not(is_intra)))
    from .molarray import concatenate, Bonds
    bonds = concatenate(blist, Bonds)
    return bonds

# -----------------------------------------------------------------------------
#
def selected_residues(session):
    '''All selected residues in all structures as an :class:`.Residues` collection.'''
    from .molarray import concatenate, Atoms
    sel_atoms = concatenate((selected_atoms(session),)
        + selected_bonds(session, inter_residue=False).atoms, Atoms)
    return sel_atoms.residues.unique()

# -----------------------------------------------------------------------------
#
def structure_residues(structures):
    '''Return all residues in specified atomic structures as an :class:`.Residues` collection.'''
    from .molarray import Residues
    res = Residues()
    for m in structures:
        res = res | m.residues
    return res

# -----------------------------------------------------------------------------
#
def uniprot_ids(structure):
    from chimerax.mmcif.uniprot_id import uniprot_ids
    uids = uniprot_ids(structure)
    if len(uids) == 0:
        from chimerax.pdb.uniprot_id import uniprot_ids
        uids = uniprot_ids(structure)
    return uids

def _residue_mouse_hover(pick, log):
    res = getattr(pick, 'residue', None)
    if res is None:
        return
    from .molobject import Residue
    if not isinstance(res, Residue):
        return
    if not getattr(log, '_next_hover_chain_info', True):
        # Supress status message if another mouse mode such
        # as surface color value reporting is issuing status messages.
        log._next_hover_chain_info = True
        return
    chain = res.chain
    if chain and chain.description:
        log.status("chain %s: %s" % (chain.chain_id, chain.description))
    elif res.name in getattr(res.structure, "_hetnam_descriptions", {}):
        log.status(res.structure._hetnam_descriptions[res.name])
            
def _register_hover_trigger(session):
    if not hasattr(session, '_residue_hover_handler') and session.ui.is_gui:
        def res_hover(tname, pick, session=session):
            _residue_mouse_hover(pick, session.logger)
        session._residue_hover_handler = session.triggers.add_handler('mouse hover', res_hover)

# custom Chain attrs should be registered in the StructureSeq base class
from chimerax.core.attributes import register_class
from .molobject import python_instances_of_class, Atom, Bond, CoordSet, Pseudobond, PseudobondManager, \
    Residue, Sequence, StructureSeq
from .pbgroup import PseudobondGroup
for reg_class in [ Atom, Structure, Bond, CoordSet, Pseudobond, PseudobondGroup, PseudobondManager,
        Residue, Sequence, StructureSeq ]:
    register_class(reg_class, lambda *args, cls=reg_class: python_instances_of_class(cls),
        {attr_name: types for attr_name, types in getattr(reg_class, '_attr_reg_info', [])})
