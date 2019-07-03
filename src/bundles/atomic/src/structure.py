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

from chimerax.core import toolshed
from chimerax.core.models import Model
from chimerax.core.state import State
from .molobject import StructureData
from chimerax.core.graphics import Drawing, Pick, PickedTriangles

CATEGORY = toolshed.STRUCTURE

class Structure(Model, StructureData):
    """
    Structure model including atomic coordinates.
    The data is managed by the :class:`.StructureData` base class
    which provides access to the C++ structures.
    """

    def __init__(self, session, *, name = "structure", c_pointer = None, restore_data = None,
                 auto_style = True, log_info = True):
        # Cross section coordinates are 2D and counterclockwise
        # Use C++ version of XSection instead of Python version
        from .molobject import RibbonXSection as XSection
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
        self._ribbon_drawing = None
        self._ribbon_t2r = {}         # ribbon triangles-to-residue map
        self._ribbon_r2t = {}         # ribbon residue-to-triangles map
        self._ribbon_tether = []      # ribbon tethers from ribbon to floating atoms
        self._ribbon_spline_backbone = {}     # backbone atom positions on ribbon
        self._ring_drawing = None

        self._ses_handlers = []
        t = self.session.triggers
        for ses_func, trig_name in [("save_setup", "begin save session"),
                ("save_teardown", "end save session")]:
            self._ses_handlers.append(t.add_handler(trig_name,
                    lambda *args, qual=ses_func: self._ses_call(qual)))
        from chimerax.core.models import MODEL_POSITION_CHANGED
        self._ses_handlers.append(t.add_handler(MODEL_POSITION_CHANGED, self._update_position))
        self.triggers.add_trigger("changes")

        self._make_drawing()

        self.model_panel_show_expanded = False	# Don't show submodels initially in model panel

    def __str__(self):
        return self.string()

    def string(self, style=None):
        '''Return a human-readable string for this structure.'''
        if style is None:
            from .settings import settings
            style = settings.atomspec_contents

        id = '#' + self.id_string
        if style.startswith("command") or not self.name:
            return id
        return '%s %s' % (self.name, id)

    @property
    def atomspec(self):
        '''Return the atom specifier string for this structure.'''
        return '#' + self.id_string

    def delete(self):
        '''Delete this structure.'''
        t = self.session.triggers
        for handler in self._ses_handlers:
            t.remove_handler(handler)
        self._ses_handlers.clear()
        ses = self.session
        Model.delete(self)	# Delete children (pseudobond groups) before deleting structure
        if not self.deleted:
            self.session = ses
            StructureData.delete(self)
            delattr(self, 'session')

    deleted = StructureData.deleted

    def copy(self, name = None):
        '''
        Return a copy of this structure with a new name.
        No atoms or other components of the structure
        are shared between the original and the copy.
        '''
        if name is None:
            name = self.name
        m = self.__class__(self.session, name = name, c_pointer = StructureData._copy(self),
                           auto_style = False, log_info = False)
        m.positions = self.positions
        return m

    def added_to_session(self, session):
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
            from chimerax.core.commands import Command
            cmd = Command(self.session)
            lighting = "full" if self.num_atoms < 300000 else "full multiShadow 16"
            cmd.run("lighting " + lighting, log=False)

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        from .molobject import has_custom_attrs
        return has_custom_attrs(Structure, self)

    def take_snapshot(self, session, flags):
        from .molobject import get_custom_attrs
        data = {'model state': Model.take_snapshot(self, session, flags),
                'structure state': StructureData.save_state(self, session, flags),
                'custom attrs': get_custom_attrs(Structure, self),
                # put in dependency on custom-attribute manager so that
                # such attributes are registered with the classes before the
                # instances become available to other part of the session
                # restore, which may access attributes with default values
                'attr-reg manager': session.attr_registration}
        for attr_name in self._session_attrs.keys():
            data[attr_name] = getattr(self, attr_name)
        from chimerax.core.state import CORE_STATE_VERSION
        data['version'] = CORE_STATE_VERSION
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

        from .molobject import set_custom_attrs
        set_custom_attrs(self, data)

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

    def _get_single_color(self):
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

    def _set_single_color(self, color):
        self.atoms.colors = color
        residues = self.residues
        residues.ribbon_colors = color
        residues.ring_colors = color

    single_color = property(_get_single_color, _set_single_color)

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

    def _update_graphics_if_needed(self, *_):
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
        self._graphics_updater.need_update()
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
        self._update_ribbon_graphics()

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

            from chimerax.core.geometry import Places
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
            p.skip_bounds = True
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
        for ring in all_rings:
            atoms = ring.ordered_atoms
            residue = atoms[0].residue
            if not residue.ring_display or not all(atoms.visibles):
                continue
            ring_count += 1
            if residue.thin_rings:
                offset = 0
            else:
                offset = min(self._atom_display_radii(atoms))
            if len(atoms) < 6:
                self.fill_small_ring(atoms, offset, residue.ring_color)
            else:
                self.fill_6ring(atoms, offset, residue.ring_color)

        if ring_count:
            self._graphics_changed |= self._SHAPE_CHANGE

    def fill_small_ring(self, atoms, offset, color):
        # 3-, 4-, and 5- membered rings
        from chimerax.core.geometry import fill_small_ring
        vertices, normals, triangles = fill_small_ring(atoms.coords, offset)
        self._ring_drawing.add_shape(vertices, normals, triangles, color, atoms)

    def fill_6ring(self, atoms, offset, color):
        # 6-membered rings
        from chimerax.core.geometry import fill_6ring
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
        self._ring_drawing.add_shape(vertices, normals, triangles, color, atoms)

    def _add_r2t(self, r, tr):
        try:
            ranges = self._ribbon_r2t[r]
        except KeyError:
            self._ribbon_r2t[r] = [tr]
        else:
            ranges.append(tr)

    def _create_ribbon_graphics(self):
        if self._ribbon_drawing is None:
            self._ribbon_drawing = p = self.new_drawing('ribbon')
            p.display = True
        else:
            p = self._ribbon_drawing
            p.remove_all_drawings()
        self._ribbon_t2r = {}
        self._ribbon_r2t = {}
        self._ribbon_tether = []
        self._ribbon_spline_backbone = {}
        if self.ribbon_display_count == 0:
            return
        from .ribbon import Ribbon
        from .molobject import Residue
        from numpy import concatenate, array, zeros
        polymers = self.polymers(missing_structure_treatment=self.PMS_TRACE_CONNECTS)
        def end_strand(res_class, ss_ranges, end):
            if res_class[-1] == XSectionManager.RC_SHEET_START:
                # Single-residue strands are coils
                res_class[-1] = XSectionManager.RC_COIL
                del ss_ranges[-1]
            else:
                # Multi-residue strands are okay
                res_class[-1] = XSectionManager.RC_SHEET_END
                ss_ranges[-1][1] = end
        def end_helix(res_class, ss_ranges, end):
            if res_class[-1] == XSectionManager.RC_HELIX_START:
                # Single-residue helices are coils
                res_class[-1] = XSectionManager.RC_COIL
                del ss_ranges[-1]
            else:
                # Multi-residue helices are okay
                res_class[-1] = XSectionManager.RC_HELIX_END
                ss_ranges[-1][1] = end
        def is_arc_helix_end(i):
            return is_arc_helix[i]
        def is_arc_helix_middle(i, j):
            if not is_arc_helix[i] or not is_arc_helix[j]:
                return False
            return ssids[i] == ssids[j]
        for rlist, ptype in polymers:
            # Always call get_polymer_spline to make sure hide bits are
            # properly set when ribbons are completely undisplayed
            any_display, atoms, coords, guides = rlist.get_polymer_spline()
            if not any_display:
                continue
            residues = atoms.residues
            rp = RibbonDrawing(self.name + " " + str(residues[0]) + " ribbons")
            p.add_drawing(rp)
            t2r = []
            # Always update all atom visibility so that undisplaying ribbon
            # will bring back previously hidden backbone atoms
            residues.atoms.update_ribbon_visibility()
            # Use residues instead of rlist below because rlist may contain
            # residues that do not participate in ribbon (e.g., because
            # it does not have a CA)
            displays = residues.ribbon_displays
            if displays.sum() == 0:
                continue
            if len(atoms) < 2:
                continue

            # Assign a residue class to each residue and compute the
            # ranges of secondary structures
            from .ribbon import XSectionManager
            polymer_type = residues.polymer_types
            is_helix = residues.is_helix
            is_strand = residues.is_strand
            ssids = residues.secondary_structure_ids
            res_class = []
            was_sheet = was_helix = False
            last_ssid = None
            helix_ranges = []
            sheet_ranges = []
            if self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
                is_arc_helix = array(is_helix)
            else:
                is_arc_helix = zeros(len(is_helix))
            was_nucleic = False

            for i in range(len(residues)):
                if polymer_type[i] == Residue.PT_NUCLEIC:
                    rc = XSectionManager.RC_NUCLEIC
                    am_sheet = am_helix = False
                    was_nucleic = True
                elif polymer_type[i] == Residue.PT_AMINO:
                    if is_strand[i]:
                        # Define sheet SS as having higher priority over helix SS
                        if was_sheet:
                            # Check if this is the start of another sheet
                            # rather than continuation for the current one
                            if ssids[i] != last_ssid:
                                end_strand(res_class, sheet_ranges, i)
                                rc = XSectionManager.RC_SHEET_START
                                sheet_ranges.append([i, -1])
                            else:
                                rc = XSectionManager.RC_SHEET_MIDDLE
                        else:
                            rc = XSectionManager.RC_SHEET_START
                            sheet_ranges.append([i, -1])
                        am_sheet = True
                        am_helix = False
                    elif is_helix[i]:
                        if was_helix:
                            # Check if this is the start of another helix
                            # rather than a continuation for the current one
                            if ssids[i] != last_ssid:
                                end_helix(res_class, helix_ranges, i)
                                rc = XSectionManager.RC_HELIX_START
                                helix_ranges.append([i, -1])
                            else:
                                rc = XSectionManager.RC_HELIX_MIDDLE
                        else:
                            rc = XSectionManager.RC_HELIX_START
                            helix_ranges.append([i, -1])
                        am_sheet = False
                        am_helix = True
                    else:
                        rc = XSectionManager.RC_COIL
                        am_sheet = am_helix = False
                    was_nucleic = False
                else:
                    if was_nucleic:
                        rc = XSectionManager.RC_NUCLEIC
                    else:
                        rc = XSectionManager.RC_COIL
                    am_sheet = am_helix = False
                if was_sheet and not am_sheet:
                    end_strand(res_class, sheet_ranges, i)
                elif was_helix and not am_helix:
                    end_helix(res_class, helix_ranges, i)
                res_class.append(rc)
                was_sheet = am_sheet
                was_helix = am_helix
                last_ssid = ssids[i]
            if was_sheet:
                # 1hxx ends in a strand
                end_strand(res_class, sheet_ranges, len(residues))
            elif was_helix:
                # 1hxx ends in a strand
                end_helix(res_class, helix_ranges, len(residues))

            # Postprocess helix ranges if in arc mode to remove
            # 2-residue helices since we cannot compute an arc
            # from two points.
            if self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
                keep = []
                for r in helix_ranges:
                    if r[1] - r[0] > 2:
                        keep.append(r)
                    else:
                        for i in range(r[0], r[1]):
                            is_arc_helix[i] = False
                helix_ranges = keep

            # Assign front and back cross sections for each residue.
            # The "front" section is between this residue and the previous.
            # The "back" section is between this residue and the next.
            # The front and back sections meet at the control point atom.
            # Compute cross sections and whether we care about a smooth
            # transition between residues.
            # If helices are displayed as tubes, we alter the cross sections
            # at the beginning and end to coils since the wide ribbon looks
            # odd when it is only for half a residue
            from .ribbon import XSectionManager
            xs_front = []
            xs_back = []
            need_twist = []
            rc0 = XSectionManager.RC_COIL
            for i in range(len(residues)):
                rc1 = res_class[i]
                try:
                    rc2 = res_class[i + 1]
                except IndexError:
                    rc2 = XSectionManager.RC_COIL
                f, b = self.ribbon_xs_mgr.assign(rc0, rc1, rc2)
                xs_front.append(f)
                xs_back.append(b)
                need_twist.append(self._need_twist(rc1, rc2))
                rc0 = rc1
            if self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
                for i in range(len(residues)):
                    if not is_helix[i]:
                        continue
                    rc = res_class[i]
                    if rc == XSectionManager.RC_HELIX_START:
                        xs_front[i] = self.ribbon_xs_mgr.xs_coil
                    elif rc == XSectionManager.RC_HELIX_END:
                        xs_back[i] = self.ribbon_xs_mgr.xs_coil
            need_twist[-1] = False

            # Perform any smoothing (e.g., strand smoothing
            # to remove lasagna sheets, pipes and planks
            # display as cylinders and planes, etc.)
            self._smooth_ribbon(residues, coords, guides, atoms, ssids,
                                xs_front, xs_back, p,
                                helix_ranges, sheet_ranges)
            if False:
                # Debugging code to display line from control point to guide
                cp = p.new_drawing(str(self) + " control points")
                from chimerax import surface
                va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
                cp.set_geometry(va, na, ta)
                from numpy import empty, float32
                cp_radii = empty(len(coords), float)
                cp_radii.fill(0.1)
                cp.positions = _tether_placements(coords, guides, cp_radii, self.TETHER_CYLINDER)
                cp_colors = empty((len(coords), 4), float)
                cp_colors[:] = (255,0,0,255)
                cp.colors = cp_colors

            # Generate ribbon
            any_ribbon = True
            ribbon = Ribbon(coords, guides, self.ribbon_orients(residues),
                            self._use_spline_normals)
            # self._show_normal_spline(p, coords, ribbon)
            v_start = 0         # for tracking starting vertex index for each residue
            t_start = 0         # for tracking starting triangle index for each residue
            vertex_list = []
            normal_list = []
            color_list = []
            triangle_list = []
            colors = residues.ribbon_colors
            if self.ribbon_show_spine:
                spine_colors = None
                spine_xyz1 = None
                spine_xyz2 = None
            # Odd number of segments gets blending, even has sharp edge
            rd = self._level_of_detail.ribbon_divisions
            if rd % 2 == 1:
                seg_blend = rd
                seg_cap = seg_blend + 1
            else:
                seg_cap = rd
                seg_blend = seg_cap + 1

            # Draw first and last residue differently because they
            # are each only a single half segment, while the middle
            # residues are each two half segments.
            import sys

            # First residues
            from .ribbon import FLIP_MINIMIZE, FLIP_PREVENT, FLIP_FORCE
            if displays[0] and not is_arc_helix_end(0):
                # print(residues[0], file=sys.__stderr__); sys.__stderr__.flush()
                xs_compat = self.ribbon_xs_mgr.is_compatible(xs_back[0], xs_front[1])
                capped = displays[0] != displays[1] or not xs_compat
                seg = capped and seg_cap or seg_blend
                front_c, front_t, front_n = ribbon.lead_segment(seg_cap // 2)
                back_c, back_t, back_n = ribbon.segment(0, ribbon.FRONT, seg, not need_twist[0])
                centers = concatenate((front_c, back_c))
                tangents = concatenate((front_t, back_t))
                normals = concatenate((front_n, back_n))
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[0],
                                                                                     centers, normals,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                s = xs_back[0].extrude(centers, tangents, normals, colors[0], True, capped, v_start)
                v_start += len(s.vertices)
                t_end = t_start + len(s.triangles)
                vertex_list.append(s.vertices)
                normal_list.append(s.normals)
                triangle_list.append(s.triangles)
                color_list.append(s.colors)
                if displays[1] and xs_compat:
                    prev_band = s.back_band
                else:
                    prev_band = None
                triangle_range = RibbonTriangleRange(t_start, t_end, rp, residues[0])
                t2r.append(triangle_range)
                self._add_r2t(residues[0], triangle_range)
                t_start = t_end
            else:
                capped = True
                prev_band = None
            # Middle residues
            for i in range(1, len(residues) - 1):
                # print(residues[i], file=sys.__stderr__); sys.__stderr__.flush()
                if not displays[i]:
                    continue
                t_end = t_start
                if is_arc_helix_middle(i, i - 1):
                    # Helix is shown separately as a tube, so we do not need to
                    # draw anything
                    mid_cap = True
                    next_cap = True
                    prev_band = None
                else:
                    # Show as ribbon
                    seg = capped and seg_cap or seg_blend
                    mid_cap = not self.ribbon_xs_mgr.is_compatible(xs_front[i], xs_back[i])
                    #print(residues[i], mid_cap, need_twist[i])
                    front_c, front_t, front_n = ribbon.segment(i - 1, ribbon.BACK, seg,
                                                               mid_cap or not need_twist[i], last=mid_cap)
                    if self.ribbon_show_spine:
                        spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[i],
                                                                                         front_c, front_n,
                                                                                         spine_colors,
                                                                                         spine_xyz1,
                                                                                         spine_xyz2)
                    xs_compat = self.ribbon_xs_mgr.is_compatible(xs_back[i], xs_front[i + 1])
                    next_cap = displays[i] != displays[i + 1] or not xs_compat
                    sf = xs_front[i].extrude(front_c, front_t, front_n, colors[i],
                                               capped, mid_cap, v_start)
                    v_start += len(sf.vertices)
                    t_end += len(sf.triangles)
                    vertex_list.append(sf.vertices)
                    normal_list.append(sf.normals)
                    triangle_list.append(sf.triangles)
                    color_list.append(sf.colors)
                    if prev_band is not None:
                        triangle_list.append(xs_front[i].blend(prev_band, sf.front_band))
                        t_end += len(triangle_list[-1])
                if is_arc_helix_middle(i, i + 1):
                    # Helix is shown separately as a tube, so we do not need to
                    # draw anything
                    prev_band = None
                else:
                    seg = next_cap and seg_cap or seg_blend
                    flip_mode = FLIP_MINIMIZE
                    if self.ribbon_mode_helix == self.RIBBON_MODE_DEFAULT and is_helix[i] and is_helix[i + 1]:
                        flip_mode = FLIP_PREVENT
                    # strands generally flip normals at every residue but
                    # beta bulges violate this rule so we cannot always flip
                    # elif is_strand[i] and is_strand[i + 1]:
                    #     flip_mode = FLIP_FORCE
                    back_c, back_t, back_n = ribbon.segment(i, ribbon.FRONT, seg, not need_twist[i],
                                                            flip_mode=flip_mode)
                    if self.ribbon_show_spine:
                        spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[i],
                                                                                         back_c, back_n,
                                                                                         spine_colors,
                                                                                         spine_xyz1,
                                                                                         spine_xyz2)
                    sb = xs_back[i].extrude(back_c, back_t, back_n, colors[i],
                                               mid_cap, next_cap, v_start)
                    v_start += len(sb.vertices)
                    t_end += len(sb.triangles)
                    vertex_list.append(sb.vertices)
                    normal_list.append(sb.normals)
                    triangle_list.append(sb.triangles)
                    color_list.append(sb.colors)
                    if not mid_cap:
                        triangle_list.append(xs_back[i].blend(sf.back_band, sb.front_band))
                        t_end += len(triangle_list[-1])
                    if next_cap:
                        prev_band = None
                    else:
                        prev_band = sb.back_band
                capped = next_cap
                if t_end != t_start:
                    triangle_range = RibbonTriangleRange(t_start, t_end, rp, residues[i])
                    t2r.append(triangle_range)
                    self._add_r2t(residues[i], triangle_range)
                    t_start = t_end
            # Last residue
            if displays[-1] and not is_arc_helix_end(-1):
                # print(residues[-1], file=sys.__stderr__); sys.__stderr__.flush()
                seg = capped and seg_cap or seg_blend
                front_c, front_t, front_n = ribbon.segment(ribbon.num_segments - 1, ribbon.BACK, seg, True)
                back_c, back_t, back_n = ribbon.trail_segment(seg_cap // 2)
                centers = concatenate((front_c, back_c))
                tangents = concatenate((front_t, back_t))
                normals = concatenate((front_n, back_n))
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[-1],
                                                                                     centers, normals,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                s = xs_front[-1].extrude(centers, tangents, normals, colors[-1],
                                    capped, True, v_start)
                v_start += len(s.vertices)
                t_end = t_start + len(s.triangles)
                vertex_list.append(s.vertices)
                normal_list.append(s.normals)
                triangle_list.append(s.triangles)
                color_list.append(s.colors)
                if prev_band is not None:
                    triangle_list.append(xs_front[-1].blend(prev_band, s.front_band))
                    t_end += len(triangle_list[-1])
                triangle_range = RibbonTriangleRange(t_start, t_end, rp, residues[-1])
                t2r.append(triangle_range)
                self._add_r2t(residues[-1], triangle_range)
                t_start = t_end

            # Create drawing from arrays
            if vertex_list:
                rp.display = True
                va = concatenate(vertex_list)
                na = concatenate(normal_list)
                from .ribbon import normalize_vector_array_inplace
                normalize_vector_array_inplace(na)
                ta = concatenate(triangle_list)
                rp.set_geometry(va, na, ta)
                rp.vertex_colors = concatenate(color_list)
            else:
                rp.display = False
            # Save mappings for picking
            self._ribbon_t2r[rp] = t2r

            # Cache position of backbone atoms on ribbon
            # and get list of tethered atoms
            positions = self._ribbon_spline_position(ribbon, residues, self._NonTetherPositions)
            self._ribbon_spline_backbone.update(positions)
            positions = self._ribbon_spline_position(ribbon, residues, self._TetherPositions)
            from numpy.linalg import norm
            from .molarray import Atoms
            tether_atoms = Atoms(list(positions.keys()))
            spline_coords = array(list(positions.values()))
            self._ribbon_spline_backbone.update(positions)
            if len(spline_coords) == 0:
                spline_coords = spline_coords.reshape((0,3))
            atom_coords = tether_atoms.coords
            offsets = array(atom_coords) - spline_coords
            tethered = norm(offsets, axis=1) > self.bond_radius

            # Create tethers if necessary
            from numpy import any
            m = residues[0].structure
            if m.ribbon_tether_scale > 0 and any(tethered):
                tp = p.new_drawing(str(self) + " ribbon_tethers")
                tp.skip_bounds = True   # Don't include in bounds calculation. Optimization.
                tp.no_cofr = True	# Don't use for finding center of rotation. Optimization.
                tp.pickable = False	# Don't allow mouse picking.
                nc = m.ribbon_tether_sides
                from chimerax import surface
                if m.ribbon_tether_shape == self.TETHER_CYLINDER:
                    va, na, ta = surface.cylinder_geometry(nc=nc, nz=2, caps=False)
                else:
                    # Assume it's either TETHER_CONE or TETHER_REVERSE_CONE
                    va, na, ta = surface.cone_geometry(nc=nc, caps=False, points_up=False)
                tp.set_geometry(va, na, ta)
                self._ribbon_tether.append((tether_atoms, tp,
                                            spline_coords[tethered],
                                            tether_atoms.filter(tethered),
                                            m.ribbon_tether_shape,
                                            m.ribbon_tether_scale))
            else:
                self._ribbon_tether.append((tether_atoms, None, None, None, None, None))

            # Create spine if necessary
            if self.ribbon_show_spine:
                sp = p.new_drawing(str(self) + " spine")
                from chimerax import surface
                va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=True)
                sp.set_geometry(va, na, ta)
                from numpy import empty, float32
                spine_radii = empty(len(spine_colors), float32)
                spine_radii.fill(0.3)
                sp.positions = _tether_placements(spine_xyz1, spine_xyz2, spine_radii, self.TETHER_CYLINDER)
                sp.colors = spine_colors
        self._graphics_changed |= self._SHAPE_CHANGE
        self.residues.ribbon_selected = False

    def _show_normal_spline(self, p, coords, ribbon):
        # Normal spline can be shown as spheres on either side (S)
        # or a cylinder across (C)
        num_coords = len(coords)
        try:
            spline = ribbon.normal_spline
            other_spline = ribbon.other_normal_spline
        except AttributeError:
            return
        sp = p.new_drawing(str(self) + " normal spline")
        from chimerax import surface
        from numpy import empty, array, float32, linspace
        from chimerax.core.geometry import Places
        num_pts = num_coords*self._level_of_detail.ribbon_divisions
        #S
        #S va, na, ta = surface.sphere_geometry(20)
        #S xyzr = empty((num_pts*2, 4), float32)
        #S t = linspace(0.0, num_coords, num=num_pts, endpoint=False)
        #S xyzr[:num_pts, :3] = [spline(i) for i in t]
        #S xyzr[num_pts:, :3] = [other_spline(i) for i in t]
        #S xyzr[:, 3] = 0.2
        #S sp.positions = Places(shift_and_scale=xyzr)
        #S sp_colors = empty((len(xyzr), 4), dtype=float32)
        #S sp_colors[:num_pts] = (255, 0, 0, 255)
        #S sp_colors[num_pts:] = (0, 255, 0, 255)
        #S
        #C
        va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=True)
        radii = empty(num_pts, dtype=float32)
        radii.fill(0.2)
        t = linspace(0.0, num_coords, num=num_pts, endpoint=False)
        xyz1 = array([spline(i) for i in t], dtype=float32)
        xyz2 = array([other_spline(i) for i in t], dtype=float32)
        sp.set_geometry(va, na, ta)
        sp.positions = _tether_placements(xyz1, xyz2, radii, self.TETHER_CYLINDER)
        sp_colors = empty((len(xyz1), 4), dtype=float32)
        sp_colors[:] = (255, 0, 0, 255)
        sp.colors = sp_colors
        #C

    def _smooth_ribbon(self, rlist, coords, guides, atoms, ssids,
                       xs_front, xs_back, p, helix_ranges, sheet_ranges):
        ribbon_adjusts = rlist.ribbon_adjusts
        if self.ribbon_mode_helix == self.RIBBON_MODE_DEFAULT:
            # Smooth helices
            # XXX: Skip helix smoothing for now since it does not work well for bent helices
            pass
            # for start, end in helix_ranges:
            #     self._smooth_helix(coords, guides, xs_front, xs_back,
            #                        ribbon_adjusts, start, end, p)
        elif self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
            for start, end in helix_ranges:
                self._arc_helix(rlist, coords, guides, ssids, xs_front, xs_back,
                                ribbon_adjusts, start, end, p)
        elif self.ribbon_mode_helix == self.RIBBON_MODE_WRAP:
            for start, end in helix_ranges:
                self._wrap_helix(rlist, coords, guides, ssids, xs_front, xs_back,
                                 ribbon_adjusts, start, end, p)
        if self.ribbon_mode_strand == self.RIBBON_MODE_DEFAULT:
            # Smooth strands
            for start, end in sheet_ranges:
                self._smooth_strand(rlist, coords, guides, xs_front, xs_back,
                                    ribbon_adjusts, start, end, p)
        elif self.ribbon_mode_strand == self.RIBBON_MODE_ARC:
            for start, end in sheet_ranges:
                self._arc_strand(rlist, coords, guides, xs_front, xs_back,
                                 ribbon_adjusts, start, end, p)

    def _smooth_helix(self, rlist, coords, guides, xs_front, xs_back,
                      ribbon_adjusts, start, end, p):
        # Try to fix up the ribbon orientation so that it is parallel to the helical axis
        from numpy import dot, newaxis, mean
        from numpy.linalg import norm
        from .ribbon import normalize_vector_array
        # We only "optimize" longer helices because short
        # ones do not contain enough information to do
        # things intelligently
        ss_coords = coords[start:end]
        adjusts = ribbon_adjusts[start:end][:, newaxis]
        axis, centroid, rel_coords = self._ss_axes(ss_coords)
        # Compute position of cylinder center corresponding to
        # helix control point atoms
        axis_pos = dot(rel_coords, axis)[:, newaxis]
        cyl_centers = centroid + axis * axis_pos
        if False:
            # Debugging code to display center of secondary structure
            self._ss_display(p, str(self) + " helix " + str(start), cyl_centers)
        # Compute radius of cylinder
        spokes = ss_coords - cyl_centers
        cyl_radius = mean(norm(spokes, axis=1))
        # Compute smoothed position of helix control point atoms
        ideal = cyl_centers + normalize_vector_array(spokes) * cyl_radius
        offsets = adjusts * (ideal - ss_coords)
        new_coords = ss_coords + offsets
        # Update both control point and guide coordinates
        coords[start:end] = new_coords
        if guides is not None:
            # Compute guide atom position relative to control point atom
            delta_guides = guides[start:end] - ss_coords
            # Move the guide location so that it forces the
            # ribbon parallel to the axis
            guides[start:end] = new_coords + axis
        # Originally, we just update the guide location to
        # the same relative place as before
        #   guides[start:end] = new_coords + delta_guides

    def _arc_helix(self, rlist, coords, guides, ssids, xs_front, xs_back,
                   ribbon_adjusts, start, end, p):
        # Only bother if at least one residue is displayed
        displays = rlist.ribbon_displays
        if not any(displays[start:end]):
            return

        from .sse import HelixCylinder
        from numpy import linspace, cos, sin
        from math import pi
        from numpy import empty, tile

        hc = HelixCylinder(coords[start:end], radius=self.ribbon_xs_mgr.tube_radius)
        centers = hc.cylinder_centers()
        radius = hc.cylinder_radius()
        normals, binormals = hc.cylinder_normals()
        icenters, inormals, ibinormals = hc.cylinder_intermediates()
        coords[start:end] = centers

        # Compute unit circle in 2D
        mgr = self.ribbon_xs_mgr
        num_pts = mgr.params[mgr.STYLE_ROUND]["sides"]
        angles = linspace(0.0, pi * 2, num=num_pts, endpoint=False)
        cos_a = radius * cos(angles)
        sin_a = radius * sin(angles)

        # Generate the cylinders and caps for displayed residues
        # Each middle residue consists of three points:
        #   intermediate point (i-1,i)
        #   point i
        #   intermediate point (i,i+1)
        # The first and last residues only have two points.
        # This means there are two bands of triangles for middle
        # residues but only one for the end residues.
        #
        # Note that even though two adjacent residues share
        # a single intermediate point, the intermediate point
        # is duplicated for each residue since they may be
        # colored differently.  XXX: Possible optimization
        # is reusing intermediate points that have the same
        # color instead of duplicating them.

        # First we count up how many caps and bands are displayed
        num_vertices = 0
        num_triangles = 0
        cap_triangles = num_pts - 2
        band_triangles = 2 * num_pts
        # First and last residues are special
        if displays[start]:
            # 3 = 1 for cap, 2 for tube
            num_vertices += 3 * num_pts
            num_triangles += cap_triangles + band_triangles
        was_displayed = displays[start]
        for i in range(start+1, end-1):
            # Middle residues
            if displays[i]:
                if not was_displayed:
                    # front cap
                    num_vertices += num_pts
                    num_triangles += cap_triangles
                # 3 for tube: (i-1,i),i,(i,i+1)
                num_vertices += 3 * num_pts
                num_triangles += 2 * band_triangles
            else:
                if was_displayed:
                    # back cap
                    num_vertices += num_pts
                    num_triangles += cap_triangles
            was_displayed = displays[i]
        # last residue
        if displays[end-1]:
            if was_displayed:
                # 3 = 1 for back cap, 2 for tube
                num_vertices += 3 * num_pts
                num_triangles += cap_triangles + band_triangles
            else:
                # 4 = 2 for caps, 2 for tube
                num_vertices += 4 * num_pts
                num_triangles += 2 * cap_triangles + band_triangles
        elif was_displayed:
            # back cap
            num_vertices += num_pts
            num_triangles += cap_triangles

        # Second, create containers for vertices, normals and triangles
        va = empty((num_vertices, 3), dtype=float)
        na = empty((num_vertices, 3), dtype=float)
        ca = empty((num_vertices, 4), dtype=float)
        ta = empty((num_triangles, 3), dtype=int)

        # Third, add vertices, normals and triangles for each residue
        # In the following functions, "i" = [start:end] and
        # "offset" = [0, end-start]
        colors = rlist.ribbon_colors
        vi = 0
        ti = 0
        def _make_circle(c, n, bn):
            nonlocal cos_a, sin_a
            from numpy import tile, cross
            from numpy.linalg import norm
            count = (len(cos_a), 1)
            normals = (tile(n, count) * cos_a.reshape(count) +
                       tile(bn, count) * sin_a.reshape(count))
            circle = normals + c
            return circle, normals
        def _make_tangent(n, bn):
            from numpy import cross
            return cross(n, bn)
        def _add_cap(c, n, bn, color, back):
            nonlocal vi, ti, va, na, ca, ta, num_pts
            circle, normals = _make_circle(c, n, bn)
            tangent = _make_tangent(n, bn)
            if back:
                tangent = -tangent
            va[vi:vi+num_pts] = circle
            na[vi:vi+num_pts] = tangent
            ca[vi:vi+num_pts] = color
            if back:
                ta[ti:ti+cap_triangles,0] = range(vi+2,vi+num_pts)
                ta[ti:ti+cap_triangles,1] = range(vi+1,vi+num_pts-1)
                ta[ti:ti+cap_triangles,2] = vi
            else:
                ta[ti:ti+cap_triangles,0] = vi
                ta[ti:ti+cap_triangles,1] = range(vi+1,vi+num_pts-1)
                ta[ti:ti+cap_triangles,2] = range(vi+2,vi+num_pts)
            vi += num_pts
            ti += cap_triangles
        def _add_band_vertices(circlef, normalsf, color):
            nonlocal vi, ti, va, na, ca, num_pts
            save = vi
            va[vi:vi+num_pts] = circlef
            na[vi:vi+num_pts] = normalsf
            ca[vi:vi+num_pts] = color
            vi += num_pts
            return save
        def _add_band_triangles(f, b):
            nonlocal ti, ta, num_pts
            ta[ti:ti+num_pts, 0] = range(f, f+num_pts)
            ta[ti:ti+num_pts, 1] = [f + (n+1) % num_pts for n in range(num_pts)]
            ta[ti:ti+num_pts, 2] = range(b, b+num_pts)
            ti += num_pts
            ta[ti:ti+num_pts, 0] = [f + (n+1) % num_pts for n in range(num_pts)]
            ta[ti:ti+num_pts, 1] = [b + (n+1) % num_pts for n in range(num_pts)]
            ta[ti:ti+num_pts, 2] = range(b, b+num_pts)
            ti += num_pts
        def add_front_cap(i):
            nonlocal start, centers, normals, binormals
            nonlocal icenters, inormals, ibinormals, colors
            if i == start:
                # First residue is special
                offset = 0
                c = centers[offset]
                n = normals[offset]
                bn = binormals[offset]
            else:
                offset = (i - 1) - start
                c = icenters[offset]
                n = inormals[offset]
                bn = ibinormals[offset]
            _add_cap(c, n, bn, colors[i], False)
        def add_back_cap(i):
            nonlocal start, end, centers, normals, binormals
            nonlocal icenters, inormals, ibinormals, colors
            if i == (end - 1):
                # Last residue is special
                offset = -1
                c = centers[offset]
                n = normals[offset]
                bn = binormals[offset]
            else:
                offset = i - start
                c = icenters[offset]
                n = inormals[offset]
                bn = ibinormals[offset]
            _add_cap(c, n, bn, colors[i], True)
        def add_front_band(i):
            nonlocal start, centers, normals, binormals
            nonlocal icenters, inormals, ibinormals, colors
            offset = i - start
            cf = icenters[offset-1]
            nf = inormals[offset-1]
            bnf = ibinormals[offset-1]
            circlef, normalsf = _make_circle(cf, nf, bnf)
            fi = _add_band_vertices(circlef, normalsf, colors[i])
            cb = centers[offset]
            nb = normals[offset]
            bnb = binormals[offset]
            circleb, normalsb = _make_circle(cb, nb, bnb)
            bi = _add_band_vertices(circleb, normalsb, colors[i])
            _add_band_triangles(fi, bi)
        def add_back_band(i):
            nonlocal start, centers, normals, binormals
            nonlocal icenters, inormals, ibinormals, colors
            offset = i - start
            cf = centers[offset]
            nf = normals[offset]
            bnf = binormals[offset]
            circlef, normalsf = _make_circle(cf, nf, bnf)
            fi = _add_band_vertices(circlef, normalsf, colors[i])
            cb = icenters[offset]
            nb = inormals[offset]
            bnb = ibinormals[offset]
            circleb, normalsb = _make_circle(cb, nb, bnb)
            bi = _add_band_vertices(circleb, normalsb, colors[i])
            _add_band_triangles(fi, bi)
        def add_both_bands(i):
            nonlocal start, centers, normals, binormals
            nonlocal icenters, inormals, ibinormals, colors
            offset = i - start
            cf = icenters[offset-1]
            nf = inormals[offset-1]
            bnf = ibinormals[offset-1]
            circlef, normalsf = _make_circle(cf, nf, bnf)
            fi = _add_band_vertices(circlef, normalsf, colors[i])
            cm = centers[offset]
            nm = normals[offset]
            bnm = binormals[offset]
            circlem, normalsm = _make_circle(cm, nm, bnm)
            mi = _add_band_vertices(circlem, normalsm, colors[i])
            cb = icenters[offset]
            nb = inormals[offset]
            bnb = ibinormals[offset]
            circleb, normalsb = _make_circle(cb, nb, bnb)
            bi = _add_band_vertices(circleb, normalsb, colors[i])
            _add_band_triangles(fi, mi)
            _add_band_triangles(mi, bi)

        # Third (still), create the caps and bands
        t_range = {}
        if displays[start]:
            add_front_cap(start)
            add_back_band(start)
            t_range[start] = [0, ti]
        was_displayed = displays[start]
        for i in range(start+1, end-1):
            if displays[i]:
                t_start = ti
                if not was_displayed:
                    add_front_cap(i)
                add_both_bands(i)
                t_range[i] = [t_start, ti]
            else:
                if was_displayed:
                    add_back_cap(i-1)
                    t_range[i-1][1] = ti
            was_displayed = displays[i]
        # last residue
        if displays[end-1]:
            t_start = ti
            if was_displayed:
                add_front_band(end-1)
                add_back_cap(end-1)
            else:
                add_front_cap(end-1)
                add_front_band(end-1)
                add_back_cap(end-1)
            t_range[end-1] = [t_start, ti]
        elif was_displayed:
            add_back_cap(end-2)
            t_range[end-2][1] = ti

        # Fourth, create graphics object of vertices, normals,
        # colors and triangles
        name = "helix-%d" % ssids[start]
        ssp = RibbonDrawing(name)
        p.add_drawing(ssp)
        ssp.set_geometry(va, na, ta)
        ssp.vertex_colors = ca

        # Finally, update selection data structures
        t2r = []
        for i, r in t_range.items():
            res = rlist[i]
            triangle_range = RibbonTriangleRange(r[0], r[1], ssp, res)
            t2r.append(triangle_range)
            self._add_r2t(res, triangle_range)
        self._ribbon_t2r[ssp] = t2r

    def _wrap_helix(self, rlist, coords, guides, ssids, xs_front, xs_back,
                    ribbon_adjusts, start, end, p):
        # Only bother if at least one residue is displayed
        displays = rlist.ribbon_displays
        if not any(displays[start:end]):
            return

        from .sse import HelixCylinder
        hc = HelixCylinder(coords[start:end])
        directions = hc.cylinder_directions()
        coords[start:end] = hc.cylinder_surface()
        guides[start:end] = coords[start:end] + directions
        if False:
            # Debugging code to display guides of secondary structure
            self._ss_guide_display(p, str(self) + " helix guide " + str(start),
                                   coords[start:end], guides[start:end])


    def _smooth_strand(self, rlist, coords, guides, xs_front, xs_back,
                       ribbon_adjusts, start, end, p):
        if (end - start + 1) <= 2:
            # Short strands do not need smoothing
            return
        from numpy import zeros, empty, dot, newaxis
        from numpy.linalg import norm
        from .ribbon import normalize
        ss_coords = coords[start:end]
        if len(ss_coords) < 3:
            # short strand, no smoothing
            ideal = ss_coords
            offsets = zeros(ss_coords.shape, dtype=float)
        else:
            # The "ideal" coordinates for a residue is computed by averaging
            # with the previous and next residues.  The first and last
            # residues are treated specially by moving in the opposite
            # direction as their neighbors.
            ideal = empty(ss_coords.shape, dtype=float)
            ideal[1:-1] = (ss_coords[1:-1] * 2 + ss_coords[:-2] + ss_coords[2:]) / 4
            # If there are exactly three residues in the strand, then they
            # should end up on a line.  We use a 0.99 factor to make sure
            # that we do not "cross the line" due to floating point round-off.
            if len(ss_coords) == 3:
                ideal[0] = ss_coords[0] - 0.99 * (ideal[1] - ss_coords[1])
                ideal[-1] = ss_coords[-1] - 0.99 * (ideal[-2] - ss_coords[-2])
            else:
                ideal[0] = ss_coords[0] - (ideal[1] - ss_coords[1])
                ideal[-1] = ss_coords[-1] - (ideal[-2] - ss_coords[-2])
            adjusts = ribbon_adjusts[start:end][:, newaxis]
            offsets = adjusts * (ideal - ss_coords)
            new_coords = ss_coords + offsets
            # Update both control point and guide coordinates
            if guides is not None:
                # Compute guide atom position relative to control point atom
                delta_guides = guides[start:end] - ss_coords
                guides[start:end] = new_coords + delta_guides
            coords[start:end] = new_coords
        if False:
            # Debugging code to display center of secondary structure
            self._ss_display(p, str(self) + " strand " + str(start), ideal)
            self._ss_guide_display(p, str(self) + " strand guide " + str(start),
                                   coords[start:end], guides[start:end])

    def _arc_strand(self, rlist, coords, guides, xs_front, xs_back,
                       ribbon_adjusts, start, end, p):
        if (end - start + 1) <= 2:
            # Short strands do not need to be shown as planks
            return
        # Only bother if at least one residue is displayed
        displays = rlist.ribbon_displays
        if not any(displays[start:end]):
            return
        from .sse import StrandPlank
        from numpy.linalg import norm
        atoms = rlist[start:end].atoms
        oxygens = atoms.filter(atoms.names == 'O')
        print(len(oxygens), "oxygens of", len(atoms), "atoms in", end - start, "residues")
        sp = StrandPlank(coords[start:end], oxygens.coords)
        centers = sp.plank_centers()
        normals, binormals = sp.plank_normals()
        if True:
            # Debugging code to display guides of secondary structure
            from numpy import newaxis
            g = sp.tilt_centers + sp.tilt_x[:,newaxis] * normals + sp.tilt_y[:,newaxis] * binormals
            self._ss_guide_display(p, str(self) + " strand guide " + str(start),
                                   sp.tilt_centers, g)
        coords[start:end] = centers
        #delta = guides[start:end] - coords[start:end]
        #guides[start:end] = coords[start:end] + delta
        guides[start:end] = coords[start:end] + binormals
        if True:
            # Debugging code to display center of secondary structure
            self._ss_display(p, str(self) + " strand " + str(start), centers)

    def _ss_axes(self, ss_coords):
        from numpy import mean, argmax
        from numpy.linalg import svd
        centroid = mean(ss_coords, axis=0)
        rel_coords = ss_coords - centroid
        ignore, vals, vecs = svd(rel_coords)
        axes = vecs[argmax(vals)]
        return axes, centroid, rel_coords

    def _ss_display(self, p, name, centers):
        from chimerax import surface
        from numpy import empty, float32
        ssp = p.new_drawing(name)
        va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
        ssp.set_geometry(va, na, ta)
        ss_radii = empty(len(centers) - 1, float32)
        ss_radii.fill(0.2)
        ssp.positions = _tether_placements(centers[:-1], centers[1:], ss_radii, self.TETHER_CYLINDER)
        ss_colors = empty((len(ss_radii), 4), float32)
        ss_colors[:] = (0,255,0,255)
        ssp.colors = ss_colors

    def _ss_guide_display(self, p, name, centers, guides):
        from chimerax import surface
        from numpy import empty, float32
        ssp = p.new_drawing(name)
        va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
        ssp.set_geometry(va, na, ta)
        ss_radii = empty(len(centers), float32)
        ss_radii.fill(0.2)
        ssp.positions = _tether_placements(centers, guides, ss_radii, self.TETHER_CYLINDER)
        ss_colors = empty((len(ss_radii), 4), float32)
        ss_colors[:] = (255,255,0,255)
        ssp.colors = ss_colors

    def _need_twist(self, rc0, rc1):
        # Determine if we need to twist ribbon smoothly from rc0 to rc1
        if rc0 == rc1:
            return True
        from .ribbon import XSectionManager
        if rc0 in XSectionManager.RC_ANY_SHEET and rc1 in XSectionManager.RC_ANY_SHEET:
            return True
        if rc0 in XSectionManager.RC_ANY_HELIX and rc1 in XSectionManager.RC_ANY_HELIX:
            return True
        if rc0 is XSectionManager.RC_HELIX_END or rc0 is XSectionManager.RC_SHEET_END:
            return True
        return False

    def _ribbon_update_spine(self, c, centers, normals, spine_colors, spine_xyz1, spine_xyz2):
        from numpy import empty
        xyz1 = centers + normals
        xyz2 = centers - normals
        color = empty((len(xyz1), 4), int)
        color[:] = c
        if spine_colors is None:
            return color, xyz1, xyz2
        else:
            from numpy import concatenate
            return (concatenate([spine_colors, color]), concatenate([spine_xyz1, xyz1]),
                    concatenate([spine_xyz2, xyz2]))

    def _update_ribbon_graphics(self):
        # Set selected ribbons in graphics
        from .molarray import Residues
        atoms = self.atoms
        if atoms.num_selected > 0:
            residues = atoms.filter(atoms.selected).unique_residues
            from numpy import array
            mask = array([r in self._ribbon_r2t for r in residues], dtype=bool)
            rsel = residues.filter(mask)
        else:
            rsel = Residues()
        residues = self.residues
        selected_residues = residues.filter(residues.ribbon_selected)
        hide = selected_residues - rsel
        keep = selected_residues & rsel
        show = rsel - selected_residues
        hide.ribbon_selected = False
        show.ribbon_selected = True
        # Change the selected triangles in drawings
        da = {}         # actions - 0=hide, 1=keep, 2=show
        residues = [hide, keep, show]
        # Partition by drawing
        for i in range(len(residues)):
            for r in residues[i]:
                try:
                    tr_list = self._ribbon_r2t[r]
                except KeyError:
                    continue
                for tr in tr_list:
                    try:
                        a = da[tr.drawing]
                    except KeyError:
                        a = da[tr.drawing] = ([], [], [])
                    a[i].append((tr.start, tr.end))
        for p, residues in da.items():
            if not residues[1] and not residues[2]:
                # No residues being kept or added
                p.highlighted_triangles_mask = None
            else:
                m = p.highlighted_triangles_mask
                if m is None:
                    import numpy
                    m = numpy.zeros((p.number_of_triangles(),), bool)
                for start, end in residues[0]:
                    m[start:end] = False
                for start, end in residues[2]:
                    m[start:end] = True
                p.highlighted_triangles_mask = m

    # Position of atoms on ribbon in spline parameter units.
    # These should correspond to the "minimum" backbone atoms
    # listed in atomstruct/Residue.cpp.
    # Negative means on the spline between previous residue
    # and this one; positive between this and next.
    # These are copied from Chimera.  May want to do a survey
    # of closest spline parameters across many structures instead.
    _TetherPositions = {
        # Amino acid
        "N":  -1/3.,
        "CA":  0.,
        "C":   1/3.,
        # Nucleotide
        "P":   -2/6.,
        "O5'": -1/6.,
        "C5'":  0.,
        "C4'":  1/6.,
        "C3'":  2/6.,
        "O3'":  3/6.,
    }
    _NonTetherPositions = {
        # Amino acid
        "O":    1/3.,
        "OXT":  1/3.,
        "OT1":  1/3.,
        "OT2":  1/3.,
        # Nucleotide
        "OP1": -2/6.,
        "O1P": -2/6.,
        "OP2": -2/6.,
        "O2P": -2/6.,
        "OP3": -2/6.,
        "O3P": -2/6.,
        "O2'": -1/6.,
        "C2'":  2/6.,
        "O4'":  1/6.,
        "C1'":  1.5/6.,
        "O3'":  2/6.,
    }

    def _ribbon_spline_position(self, ribbon, residues, pos_map):
        positions = {}
        for n, r in enumerate(residues):
            first = (r == residues[0])
            last = (r == residues[-1])
            for atom_name, position in pos_map.items():
                a = r.find_atom(atom_name)
                if a is None or not a.is_backbone():
                    continue
                if last:
                    p = ribbon.position(n - 1, 1 + position)
                elif position >= 0 or first:
                    p = ribbon.position(n, position)
                else:
                    p = ribbon.position(n - 1, 1 + position)
                positions[a] = p
        return positions

    def ribbon_coord(self, a):
        return self._ribbon_spline_backbone[a]

    def _update_ribbon_tethers(self):
        from numpy import around
        for all_atoms, tp, xyz1, tether_atoms, shape, scale in self._ribbon_tether:
            all_atoms.update_ribbon_visibility()
            if tp:
                xyz2 = tether_atoms.coords
                radii = self._atom_display_radii(tether_atoms) * scale
                tp.positions = _tether_placements(xyz1, xyz2, radii, shape)
                tp.display_positions = tether_atoms.visibles & tether_atoms.residues.ribbon_displays
                colors = tether_atoms.colors
                colors[:,3] = around(colors[:,3] * self.ribbon_tether_opacity).astype(int)
                tp.colors = colors

    def bounds(self):
        #
        # TODO: Updating ribbon graphics during bounds() calc is precarious.
        # This caused bug #1717 where ribbons got unexpectedly deleted during a call
        # to bounds() by the shadow calculation code during rendering.
        # Probably should define bounds to mean the displayed graphics bounds
        # which may not include pending updates to graphics.  Code that needs to
        # include the pending graphics changes would need to explicitly update the graphics.
        #
        self._update_graphics_if_needed()       # Ribbon bounds computed from graphics
        # import sys, time
        # start = time.time()
        b = super().bounds()
        # stop = time.time()
        # print('structure bounds time:', (stop - start) * 1e6, file=sys.__stderr__)
        return b

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
        drawings = [ad]
        drawings.extend(rd for rd in self.child_drawings() if isinstance(rd, RibbonDrawing))
        from chimerax.core.geometry import union_bounds
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
        self._update_graphics_if_needed()       # Ribbon drawing lazily computed
        super().x3d_needs(x3d_scene)

    def write_x3d(self, *args, **kw):
        self._update_graphics_if_needed()       # Ribbon drawing lazily computed
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
        return []

    def clear_selection(self):
        self.selected = False
        self.atoms.selected = False
        self.bonds.selected = False
        self.residues.ribbon_selected = False
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
            level = 1005
            psel = asel | atoms.has_selected_bonds
        else:
            r = atoms.residues
            rids = r.unique_ids
            from numpy import unique, in1d
            sel_rids = unique(rids[asel])
            ares = in1d(rids, sel_rids)
            if ares.sum() > na:
                # Promote to entire residues
                level = 1004
                psel = ares
            else:
                ssids = r.secondary_structure_ids
                sel_ssids = unique(ssids[asel])
                ass = in1d(ssids, sel_ssids)
                if ass.sum() > na:
                    # Promote to secondary structure
                    level = 1003
                    psel = ass
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

    def atomspec_atoms(self):
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
            s = numpy.vectorize(choose)(objects)
            selected = numpy.logical_and(selected, s)
        return selected


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
                    selected = numpy.logical_or(selected, s)
                else:
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
        from chimerax.core.geometry import find_close_points
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
        # TODO: use the next two lines instead of the following four for a 5% speedup
        # should be okay to change since Structure.bounds does _update_graphics_if_needed first
        # xyzr = self.positions.shift_and_scale_array()
        # coords, radii = xyzr[:, :3], xyzr[:, 3]
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
        from chimerax.core.geometry import sphere_bounds, copies_bounding_box
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

        xyzr = self.positions.shift_and_scale_array()
        coords, radii = xyzr[:,:3], xyzr[:,3]

        # Check for atom sphere intercept
        from chimerax.core import geometry
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
        from chimerax.core import geometry
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
    # Used for both bonds and pseudoonds
    # can't have any child drawings

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
        if bonds is None:
            return None
        ba1, ba2 = bonds.atoms
        c1, c2, r = ba1.coords, ba2.coords, bonds.radii
        r.shape = (r.shape[0], 1)
        from numpy import amin, amax
        xyz_min = amin([amin(c1 - r, axis=0), amin(c2 - r, axis=0)], axis=0)
        xyz_max = amax([amax(c1 + r, axis=0), amax(c2 + r, axis=0)], axis=0)
        from chimerax.core.geometry import Bounds, copies_bounding_box
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
        cyl_info = _halfbond_cylinder_x3d(ba1.coords, ba2.coords, bonds.radii)
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


class RibbonDrawing(Drawing):
    # TODO: eliminate need for parent.parent._ribbon_t2r

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None
        p = super().first_intercept(mxyz1, mxyz2)
        if p is None:
            return None
        t2r = self.parent.parent._ribbon_t2r[self]
        from bisect import bisect_right
        n = bisect_right(t2r, p.triangle_number)
        if n > 0:
            triangle_range = t2r[n - 1]
            return PickedResidue(triangle_range.residue, p.distance)
        return None

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []
        t2r = self.parent.parent._ribbon_t2r[self]
        picks = []
        rp = super().planes_pick(planes)
        for p in rp:
            if isinstance(p, PickedTriangles) and p.drawing() is self:
                tmask = p._triangles_mask
                res = [rtr.residue for rtr in t2r if tmask[rtr.start:rtr.end].sum() > 0]
                if res:
                    from .molarray import Residues
                    rc = Residues(res)
                    picks.append(PickedResidues(rc))
        return picks


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

    def apply_auto_styling(self, set_lighting = False, style=None):
        if style is None:
            if self.num_chains == 0:
                style = "non-polymer"
            elif self.num_chains < 5:
                style = "small polymer"
            elif self.num_chains < 250:
                style = "medium polymer"
            else:
                style = "large polymer"

        color = self.initial_color(self.session.main_view.background_color)
        self.set_color(color)

        atoms = self.atoms
        if style == "non-polymer":
            lighting = "default"
            from .molobject import Atom, Bond
            atoms.draw_modes = Atom.STICK_STYLE
            from .colors import element_colors
            het_atoms = atoms.filter(atoms.element_numbers != 6)
            het_atoms.colors = element_colors(het_atoms.element_numbers)
        elif style == "small polymer":
            lighting = "default"
            from .molobject import Atom, Bond, Residue
            atoms.draw_modes = Atom.STICK_STYLE
            from .colors import element_colors
            het_atoms = atoms.filter(atoms.element_numbers != 6)
            het_atoms.colors = element_colors(het_atoms.element_numbers)
            ribbonable = self.chains.existing_residues
            # 10 residues or less is basically a trivial depiction if ribboned
            if len(ribbonable) > 10:
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
                    from .nucleotides.cmd import nucleotides
                    if len(nucleic) < 100:
                        nucleotides(self.session, 'tube/slab', objects=nucleic)
                    else:
                        nucleotides(self.session, 'ladder', objects=nucleic)
                    from .colors import nucleotide_colors
                    nucleic.ring_colors = nucleotide_colors(nucleic)[0]
                if ligand:
                    # show residues interacting with ligand
                    lig_points = ligand.atoms.coords
                    mol_points = atoms.coords
                    from chimerax.core.geometry import find_close_points
                    close_indices = find_close_points(lig_points, mol_points, 3.6)[1]
                    display |= atoms.filter(close_indices).residues
                display_atoms = display.atoms
                if self.num_residues > 1:
                    display_atoms = display_atoms.filter(display_atoms.idatm_types != "HC")
                display_atoms.displays = True
                ribbonable.ribbon_displays = True
        elif style == "medium polymer":
            lighting = "full" if self.num_atoms < 300000 else "full multiShadow 16"
            from .colors import chain_colors, element_colors
            residues = self.residues
            residues.ribbon_colors = residues.ring_colors = chain_colors(residues.chain_ids)
            atoms.colors = chain_colors(atoms.residues.chain_ids)
            from .molobject import Atom
            ligand_atoms = atoms.filter(atoms.structure_categories == "ligand")
            ligand_atoms.draw_modes = Atom.STICK_STYLE
            ligand_atoms.colors = element_colors(ligand_atoms.element_numbers)
            solvent_atoms = atoms.filter(atoms.structure_categories == "solvent")
            solvent_atoms.draw_modes = Atom.BALL_STYLE
            solvent_atoms.colors = element_colors(solvent_atoms.element_numbers)
        else:
            # since this is now available as a preset, allow for possibly a smaller number of atoms
            lighting = "soft" if self.num_atoms < 300000 else "soft multiShadow 16"

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
            from chimerax.core.commands import Command
            cmd = Command(self.session)
            cmd.run("lighting " + lighting, log=False)

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        from .molobject import has_custom_attrs
        return has_custom_attrs(Structure, self) or has_custom_attrs(AtomicStructure, self)

    def take_snapshot(self, session, flags):
        from .molobject import get_custom_attrs
        data = {
            'AtomicStructure version': 2,
            'structure state': Structure.take_snapshot(self, session, flags),
            'custom attrs': get_custom_attrs(AtomicStructure, self)
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = AtomicStructure(session, auto_style = False, log_info = False)
        if data.get('AtomicStructure version', 1) == 1:
            Structure.set_state_from_snapshot(s, session, data)
        else:
            from .molobject import set_custom_attrs
            Structure.set_state_from_snapshot(s, session, data['structure state'])
            set_custom_attrs(s, data)
        return s

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
        from .pdb import process_chem_name
        for k, v in hnd.items():
            hnd[k] = process_chem_name(v)

    def _set_chain_descriptions(self, session):
        from . import mmcif
        chain_to_desc = {}
        struct_asym, entity = mmcif.get_mmcif_tables_from_metadata(self, ['struct_asym', 'entity'])
        if struct_asym:
            entity, = mmcif.get_mmcif_tables_from_metadata(self, ['entity'])
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
                    chain_to_desc[ch.chain_id] = (
                        entity_to_description[mmcif_chain_to_entity[mmcif_cid]], False)
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
            from chimerax.atomic.pdb import process_chem_name
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
               chain_res_range(chain), chain.chain_id)
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
            return '<a title="Select chain" href="cxcmd:select %s">%s/%s</a>' % (
                chain_res_range(chain), chain.structure.id_string, chain.chain_id)
        self._report_chain_summary(session, descripts, chain_text, True)

    def _report_res_info(self, session):
        if hasattr(self, 'get_formatted_res_info'):
            res_info = self.get_formatted_res_info(standalone=True)
            if res_info:
                session.logger.info(res_info, is_html=True)

    def _report_chain_summary(self, session, descripts, chain_text, is_ensemble):
        def descript_text(description, chains):
            from html import escape
            return '<a title="Show sequence" href="cxcmd:sequence chain %s">%s</a>' % (
                ''.join(["#%s/%s" % (chain.structure.id_string, chain.chain_id)
                    for chain in chains]), escape(description))
        from chimerax.core.logger import html_table_params
        summary = '\n<table %s>\n' % html_table_params
        summary += '  <thead>\n'
        summary += '    <tr>\n'
        summary += '      <th colspan="2">Chain information for %s</th>\n' % (
            self.name if is_ensemble else self)
        summary += '    </tr>\n'
        summary += '    <tr>\n'
        summary += '      <th>Chain</th>\n'
        summary += '      <th>Description</th>\n'
        summary += '    </tr>\n'
        summary += '  </thead>\n'
        summary += '  <tbody>\n'
        for key, chains in descripts.items():
            description, characters = key
            summary += '    <tr>\n'
            summary += '      <td style="text-align:center">'
            summary += ' '.join([chain_text(chain) for chain in chains])
            summary += '      </td>'
            summary += '      <td>'
            summary += descript_text(description, chains)
            summary += '      </td>'
            summary += '    </tr>\n'
        summary += '  </tbody>\n'
        summary += '</table>'
        session.logger.info(summary, is_html=True)

    def _report_assemblies(self, session):
        if getattr(self, 'ignore_assemblies', False):
            return
        html = assembly_html_table(self)
        if html:
            session.logger.info(html, is_html=True)

def assembly_html_table(mol):
    '''HTML table listing assemblies using info from metadata instead of reparsing mmCIF file.'''
    from chimerax.atomic import mmcif 
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
    first, last = existing[0], existing[-1]
    return "%s-%s" % (first.string(style="command"), last.string(residue_only=True, style="command")[1:])


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
        self._last_ribbon_divisions = 20
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        self._display_handler = t.add_handler(MODEL_DISPLAY_CHANGED, self._model_display_changed)
        self._need_update = False

    def __del__(self):
        self.session.triggers.remove_handler(self._handler)
        self.session.triggers.remove_handler(self._display_handler)

    def add_structure(self, s):
        self._structures.add(s)
        self._structures_array = None
        self.num_atoms_shown = 0	# Make sure new structure gets a level of detail update

    def remove_structure(self, s):
        self._structures.remove(s)
        self._structures_array = None

    def need_update(self):
        self._need_update = True

    def _model_display_changed(self, tname, model):
        if isinstance(model, Structure) or _has_structure_descendant(model):
            self._need_update = True

    def _update_graphics_if_needed(self, *_):
        s = self._array()
        gc = s._graphics_changeds	# Includes pseudobond group changes.
        if gc.any() or self._need_update:
            for i in gc.nonzero()[0]:
                s[i]._update_graphics_if_needed()

            # Update level of detail
            n = sum(m.num_atoms_visible * m.num_displayed_positions
                    for m in s if m.visible)
            if n > 0 and n != self.num_atoms_shown:
                self.num_atoms_shown = n
                self.update_level_of_detail()
            self._need_update = False
            if (gc & StructureData._SELECT_CHANGE).any():
                from chimerax.core.selection import SELECTION_CHANGED
                self.session.triggers.activate_trigger(SELECTION_CHANGED, None)
                # XXX: No data for now.  What should be passed?

    def update_level_of_detail(self):
        n = self.num_atoms_shown

        lod = self.level_of_detail
        ribbon_changed = (lod.ribbon_divisions != self._last_ribbon_divisions)
        if ribbon_changed:
            self._last_ribbon_divisions = lod.ribbon_divisions

        for m in self._structures:
            if m.display:
                m._update_level_of_detail(n)
                if ribbon_changed:
                    m._graphics_changed |= m._RIBBON_CHANGE

    def _array(self):
        sa = self._structures_array
        if sa is None:
            from .molarray import StructureDatas, object_pointers
            self._structures_array = sa = StructureDatas(object_pointers(self._structures))
        return sa

    def set_subdivision(self, subdivision):
        lod = self.level_of_detail
        lod.quality = subdivision
        lod.atom_fixed_triangles = None
        lod.bond_fixed_triangles = None
        self.update_level_of_detail()

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

        self._atom_min_triangles = 10
        self._atom_max_triangles = 2000
        self._atom_default_triangles = 200
        self._atom_max_total_triangles = 5000000
        self._step_factor = 1.2
        self.atom_fixed_triangles = None	# If not None use fixed number of triangles
        self._sphere_geometries = {}	# Map ntri to (va,na,ta)

        self._bond_min_triangles = 24
        self._bond_max_triangles = 160
        self._bond_default_triangles = 60
        self._bond_max_total_triangles = 5000000
        self.bond_fixed_triangles = None	# If not None use fixed number of triangles
        self._cylinder_geometries = {}	# Map ntri to (va,na,ta)

        self.ribbon_divisions = 20

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
            from chimerax.core.geometry.sphere import sphere_triangulation
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
            # Update instanced sphere triangulation
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
        atoms = self._structure.atoms
        atoms.selected = asel = self._atom_sel_mask
        atoms[asel].intra_bonds.selected = True
    def demote(self):
        s = self._structure
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
        return self.atom.string(style='command')
    @property
    def residue(self):
        return self.atom.residue
    def select(self, mode = 'add'):
        select_atom(self.atom, mode)

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
    cxyz1, cxyz2 = (a1.scene_coords, a2.scene_coords) if scene_coordinates else (a1.coords, a2.coords)
    r = bs.radii

    # Check for atom sphere intercept
    from chimerax.core import geometry
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
    from chimerax.core import geometry
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
#
class RibbonTriangleRange:
    __slots__ = ["start", "end", "drawing", "residue"]
    def __init__(self, start, end, drawing, residue):
        self.start = start
        self.end = end
        self.drawing = drawing
        self.residue = residue
    def __lt__(self, other): return self.start < other
    def __le__(self, other): return self.start <= other
    def __eq__(self, other): return self.start == other
    def __ne__(self, other): return self.start != other
    def __gt__(self, other): return self.start > other
    def __gt__(self, other): return self.start >= other

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking one prototype cylinder to each bond location.
#
def _bond_cylinder_placements(axyz0, axyz1, radii):

  n = len(axyz0)
  from numpy import empty, float32
  p = empty((n,4,4), float32)

  from chimerax.core.geometry import cylinder_rotations
  cylinder_rotations(axyz0, axyz1, radii, p)

  p[:,3,:3] = 0.5*(axyz0 + axyz1)

  from chimerax.core.geometry import Places
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

  from chimerax.core.geometry import half_cylinder_rotations
  half_cylinder_rotations(axyz0, axyz1, radii, p)

  from chimerax.core.geometry import Places
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

  from chimerax.core.geometry import cylinder_rotations_x3d
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
def _tether_placements(xyz0, xyz1, radius, shape):
    if shape == StructureData.TETHER_REVERSE_CONE:
        return _bond_cylinder_placements(xyz1, xyz0, radius)
    else:
        return _bond_cylinder_placements(xyz0, xyz1, radius)

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
    return [m for m in session.models.list() if isinstance(m,AtomicStructure)]

# -----------------------------------------------------------------------------
#
def all_structures(session):
    '''List of all :class:`.Structure` objects.'''
    return [m for m in session.models.list() if isinstance(m,Structure)]

# -----------------------------------------------------------------------------
#
def all_atoms(session, atomic_only=False):
    '''All atoms in all structures as an :class:`.Atoms` collection.'''
    func = all_atomic_structures if atomic_only else all_structures
    return structure_atoms(func(session))

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
    alist = []
    for m in session.models.list(type = Structure):
        alist.extend(m.selected_items('atoms'))
    from .molarray import concatenate, Atoms
    atoms = concatenate(alist, Atoms)
    return atoms

# -----------------------------------------------------------------------------
#
def selected_bonds(session):
    '''All selected bonds in all structures as an :class:`.Bonds` collection.'''
    blist = []
    for m in session.models.list(type = Structure):
        for b in m.selected_items('bonds'):
            blist.append(b)
    from .molarray import concatenate, Bonds
    bonds = concatenate(blist, Bonds)
    return bonds

# -----------------------------------------------------------------------------
#
def structure_residues(structures):
    '''Return all residues in specified atomic structures as an :class:`.Atoms` collection.'''
    from .molarray import Residues
    res = Residues()
    for m in structures:
        res = res | m.residues
    return res
