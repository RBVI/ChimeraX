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

from .. import toolshed
from ..models import Model
from ..state import State
from .molobject import StructureData

CATEGORY = toolshed.STRUCTURE

class Structure(Model, StructureData):

    def __init__(self, session, *, name = "structure", c_pointer = None, restore_data = None,
                 smart_initial_display = True):
        # Cross section coordinates are 2D and counterclockwise
        # Use C++ version of XSection instead of Python version
        from .molobject import RibbonXSection as XSection
        from .molarray import Residues
        from numpy import array
        from .ribbon import XSectionManager
       
        # attrs that should be saved in sessions, along with their initial values...
        self._session_attrs = {
            'ball_scale': 0.3,		# Scales sphere radius in ball and stick style
            'bond_radius': 0.2,
            'pseudobond_radius': 0.05,
            #'_ribbon_selected_residues': Residues(),
        }

        StructureData.__init__(self, c_pointer)
        for attr_name, val in self._session_attrs.items():
            setattr(self, attr_name, val)
        Model.__init__(self, name, session)
        self._smart_initial_display = smart_initial_display

        # for now, restore attrs to default initial values even for sessions...
        self._deleted = False
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._cached_atom_bounds = None
        self._atom_bounds_needs_update = True
        self._ribbon_drawing = None
        self._ribbon_t2r = {}         # ribbon triangles-to-residue map
        self._ribbon_r2t = {}         # ribbon residue-to-triangles map
        self._ribbon_tether = []      # ribbon tethers from ribbon to floating atoms
        self.ribbon_xs_mgr = XSectionManager(self)
        # TODO: move back into _session_attrs when Collection instances
        # handle session saving/restoring
        self._ribbon_selected_residues = Residues()

        from . import molobject
        molobject.add_to_object_map(self)

        self._ses_handlers = []
        t = self.session.triggers
        for ses_func, trig_name in [("save_setup", "begin save session"),
                ("save_teardown", "end save session")]:
            self._ses_handlers.append(t.add_handler(trig_name,
                    lambda *args, qual=ses_func: self._ses_call(qual)))

        self._make_drawing()

    def __str__(self):
        from ..core_settings import settings
        id = '#' + self.id_string()
        if settings.atomspec_contents == "command-line specifier" or not self.name:
            return id
        return '%s %s' % (self.name, id)

    def delete(self):
        '''Delete this structure.'''
        self._deleted = True
        t = self.session.triggers
        for handler in self._ses_handlers:
            t.remove_handler(handler)
        Model.delete(self)	# Delete children (pseudobond groups) before deleting structure
        StructureData.delete(self)

    def deleted(self):
        '''Has this atomic structure been deleted.'''
        return self._deleted

    def copy(self, name = None):
        '''
        Return a copy of this structure with a new name.
        No atoms or other components of the structure
        are shared between the original and the copy.
        '''
        if name is None:
            name = self.name
        m = self.__class__(self.session, name = name, c_pointer = StructureData._copy(self),
                           smart_initial_display = False)
        m.positions = self.positions
        return m

    def added_to_session(self, session):
        if self._smart_initial_display:
            color = self.initial_color(session.main_view.background_color)
            self.set_color(color)

            atoms = self.atoms
            if self.num_chains == 0:
                lighting = "default"
                from .molobject import Atom, Bond
                atoms.draw_modes = Atom.STICK_STYLE
                from .colors import element_colors
                het_atoms = atoms.filter(atoms.element_numbers != 6)
                het_atoms.colors = element_colors(het_atoms.element_numbers)
            elif self.num_chains < 5:
                lighting = "default"
                from .molobject import Atom, Bond
                atoms.draw_modes = Atom.STICK_STYLE
                from .colors import element_colors
                het_atoms = atoms.filter(atoms.element_numbers != 6)
                het_atoms.colors = element_colors(het_atoms.element_numbers)
                physical_residues = self.chains.existing_residues
                ribbonable = physical_residues.filter(physical_residues.num_atoms > 1)
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
                    pas = ribbonable.existing_principal_atoms
                    nucleic = pas.residues.filter(pas.names != "CA")
                    display |= nucleic
                    if ligand:
                        # show residues interacting with ligand
                        lig_points = ligand.atoms.coords
                        mol_points = atoms.coords
                        from ..geometry import find_closest_points
                        close_indices = find_closest_points(lig_points, mol_points, 3.6)[1]
                        display |= atoms.filter(close_indices).residues
                    display_atoms = display.atoms
                    if self.num_residues > 1:
                        display_atoms = display_atoms.filter(display_atoms.idatm_types != "HC")
                    display_atoms.displays = True
                    ribbonable.ribbon_displays = True
                elif len(ribbonable) == 0:
                    # CA only?
                    atoms.draw_modes = Atom.BALL_STYLE
            elif self.num_chains < 250:
                lighting = "full"
                from .colors import chain_colors, element_colors
                residues = self.residues
                residues.ribbon_colors = chain_colors(residues.chain_ids)
                atoms.colors = chain_colors(atoms.residues.chain_ids)
                from .molobject import Atom
                ligand_atoms = atoms.filter(atoms.structure_categories == "ligand")
                ligand_atoms.draw_modes = Atom.STICK_STYLE
                ligand_atoms.colors = element_colors(ligand_atoms.element_numbers)
                solvent_atoms = atoms.filter(atoms.structure_categories == "solvent")
                solvent_atoms.draw_modes = Atom.BALL_STYLE
                solvent_atoms.colors = element_colors(solvent_atoms.element_numbers)
            else:
                lighting = "soft multiShadow 16"
            from ..commands import Command
            if len([m for m in session.models.list()
                    if isinstance(m, self.__class__)]) == 1:
                cmd = Command(session)
                cmd.run("lighting " + lighting, log=False)

        self._start_change_tracking(session.change_tracker)

        # Setup handler to manage C++ data changes that require graphics updates.
        gu = structure_graphics_updater(session)
        gu.add_structure(self)

    def removed_from_session(self, session):
        gu = structure_graphics_updater(session)
        gu.remove_structure(self)

    def take_snapshot(self, session, flags):
        data = {'model state': Model.take_snapshot(self, session, flags),
                'structure state': StructureData.take_snapshot(self, session, flags)}
        for attr_name in self._session_attrs.keys():
            data[attr_name] = getattr(self, attr_name)
        from ..state import CORE_STATE_VERSION
        data['version'] = CORE_STATE_VERSION
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = Structure(session, smart_initial_display = False)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        # Model restores self.name, which is a property of the C++ StructureData
        # so initialize StructureData first.
        StructureData.set_state_from_snapshot(self, session, data['structure state'])
        Model.set_state_from_snapshot(self, session, data['model state'])

        for attr_name, default_val in self._session_attrs.items():
            setattr(self, attr_name, data.get(attr_name, default_val))

        # Create Python pseudobond group models so they are added as children.
        list(self.pbg_map.values())

        # TODO: For some reason ribbon drawing does not update automatically.
        # TODO: Also marker atoms do not draw without this.
        self._graphics_changed |= (self._SHAPE_CHANGE | self._RIBBON_CHANGE)

    def reset_state(self, session):
        pass

    def initial_color(self, bg_color):
        from .colors import structure_color
        return structure_color(self.id, bg_color)

    def set_color(self, color):
        from ..colors import Color
        if isinstance(color, Color):
            rgba = color.uint8x4()
        else:
            rgba = color
        StructureData.set_color(self, rgba)
        Model.set_color(self, rgba)

    def _get_single_color(self):
        residues = self.residues
        ribbon_displays = residues.ribbon_displays
        from ..colors import most_common_color
        if ribbon_displays.any():
            return most_common_color(residues.filter(ribbon_displays).ribbon_colors)
        atoms = self.atoms
        shown = atoms.filter(atoms.displays)
        if shown:
            return most_common_color(shown.colors)
        return most_common_color(atoms.colors)
    def _set_single_color(self, color):
        self.atoms.colors = color
        self.residues.ribbon_colors = color
    single_color = property(_get_single_color, _set_single_color)

    def _make_drawing(self):
        # Create graphics
        self._update_atom_graphics()
        self._update_bond_graphics()
        for pbg in self.pbg_map.values():
            pbg._update_graphics()
        self._create_ribbon_graphics()

    @property
    def _level_of_detail(self):
        gu = structure_graphics_updater(self.session)
        return gu.level_of_detail

    def new_atoms(self):
        # TODO: Handle instead with a C++ notification that atoms added or deleted
        self._atom_bounds_needs_update = True

    def _update_graphics_if_needed(self, *_):
        gc = self._graphics_changed
        if gc == 0:
            return
        
        if gc & self._RIBBON_CHANGE:
            self._create_ribbon_graphics()
            # Displaying ribbon can set backbone atom hide bits producing shape change.
            gc |= self._graphics_changed
        
        # Update graphics
        self._graphics_changed = 0
        s = (gc & self._SHAPE_CHANGE)
        if gc & (self._COLOR_CHANGE | self._RIBBON_CHANGE) or s:
            self._update_ribbon_tethers()
        self._update_graphics(gc)
        self.redraw_needed(shape_changed = s,
                           selection_changed = (gc & self._SELECT_CHANGE))
        if s:
            self._atom_bounds_needs_update = True

        if gc & self._SELECT_CHANGE:
            # Update selection on molecular surfaces
            # TODO: Won't work for surfaces spanning multiple molecules
            from .molsurf import MolecularSurface
            for surf in self.child_drawings():
                if isinstance(surf, MolecularSurface):
                    surf.update_selection()

    def _update_graphics(self, changes = StructureData._ALL_CHANGE):
        self._update_atom_graphics(changes)
        self._update_bond_graphics(changes)
        for pbg in self.pbg_map.values():
            pbg._update_graphics(changes)
        self._update_ribbon_graphics()

    def _update_atom_graphics(self, changes = StructureData._ALL_CHANGE):
        atoms = self.atoms  # optimzation, avoid making new numpy array from C++
        avis = atoms.visibles
        p = self._atoms_drawing
        if p is None:
            if avis.sum() == 0:
                return
            self._atoms_drawing = p = self.new_drawing('atoms')
            self._atoms_drawing.custom_x3d = self._custom_atom_x3d
            # Update level of detail of spheres
            self._level_of_detail.set_atom_sphere_geometry(p)

        if changes & self._SHAPE_CHANGE:
            # Set instanced sphere center position and radius
            n = len(atoms)
            from numpy import empty, float32, multiply
            xyzr = empty((n, 4), float32)
            xyzr[:, :3] = atoms.coords
            xyzr[:, 3] = self._atom_display_radii(atoms)

            from ..geometry import Places
            p.positions = Places(shift_and_scale=xyzr)
            p.display_positions = avis

        if changes & (self._COLOR_CHANGE | self._SHAPE_CHANGE):
            # Set atom colors
            p.colors = atoms.colors

        if changes & (self._SELECT_CHANGE | self._SHAPE_CHANGE):
            # Set selected
            p.selected_positions = atoms.selected if atoms.num_selected > 0 else None

    def _custom_atom_x3d(self, stream, x3d_scene, indent, place):
        from numpy import empty, float32
        p = self._atoms_drawing
        atoms = self.atoms
        radii = self._atom_display_radii(atoms)
        tab = ' ' * indent
        for v, xyz, r, c in zip(p.display_positions, atoms.coords, radii, p.colors):
            if not v:
                continue
            print('%s<Transform translation="%g %g %g">' % (tab, xyz[0], xyz[1], xyz[2]), file=stream)
            print('%s <Shape>' % tab, file=stream)
            self.reuse_appearance(stream, x3d_scene, indent + 2, c)
            print('%s  <Sphere radius="%g"/>' % (tab, r), file=stream)
            print('%s </Shape>' % tab, file=stream)
            print('%s</Transform>' % tab, file=stream)

    def _atom_display_radii(self, atoms):
        r = atoms.radii.copy()
        dm = atoms.draw_modes
        from .molobject import Atom
        r[dm == Atom.BALL_STYLE] *= self.ball_scale
        r[dm == Atom.STICK_STYLE] = self.bond_radius
        return r

    def _update_bond_graphics(self, changes = StructureData._ALL_CHANGE):
        bonds = self.bonds  # optimzation, avoid making new numpy array from C++
        p = self._bonds_drawing
        if p is None:
            if bonds.num_shown == 0:
                return
            self._bonds_drawing = p = self.new_drawing('bonds')
            self._bonds_drawing.custom_x3d = self._custom_bond_x3d
            # Update level of detail of cylinders
            self._level_of_detail.set_bond_cylinder_geometry(p)

        if changes & (self._SHAPE_CHANGE | self._SELECT_CHANGE):
            bond_atoms = bonds.atoms
        if changes & self._SHAPE_CHANGE:
            ba1, ba2 = bond_atoms
            p.positions = _halfbond_cylinder_placements(ba1.coords, ba2.coords, bonds.radii)
            p.display_positions = _shown_bond_cylinders(bonds)
        if changes & (self._COLOR_CHANGE | self._SHAPE_CHANGE):
            p.colors = c = bonds.half_colors
        if changes & (self._SELECT_CHANGE | self._SHAPE_CHANGE):
            p.selected_positions = _selected_bond_cylinders(bond_atoms)

    def _custom_bond_x3d(self, stream, x3d_scene, indent, place):
        from numpy import empty, float32
        p = self._bonds_drawing
        bonds = self.bonds
        ba1, ba2 = bonds.atoms
        cyl_info = _halfbond_cylinder_x3d(ba1.coords, ba2.coords, bonds.radii)
        tab = ' ' * indent
        for v, ci, c in zip(p.display_positions, cyl_info, p.colors):
            if not v:
                continue
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

    def _update_level_of_detail(self, total_atoms):
        lod = self._level_of_detail
        bd = self._bonds_drawing
        if bd:
            lod.set_bond_cylinder_geometry(bd, total_atoms)
        ad = self._atoms_drawing
        if ad:
            lod.set_atom_sphere_geometry(ad, total_atoms)

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
        if self.ribbon_display_count == 0:
            return
        from .ribbon import Ribbon
        from .molobject import Residue
        from numpy import concatenate, array, zeros
        polymers = self.polymers(False, False)
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
        for rlist in polymers:
            rp = p.new_drawing(self.name + " ribbons")
            t2r = []
            # Always call get_polymer_spline to make sure hide bits are
            # properly set when ribbons are completely undisplayed
            any_display, atoms, coords, guides = rlist.get_polymer_spline(self.ribbon_orientation)
            if not any_display:
                continue
            residues = atoms.residues
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

            # Assign front and back cross sections for each residue.
            # The "front" section is between this residue and the previous.
            # The "back" section is between this residue and the next.
            # The front and back sections meet at the control point atom.
            # Compute cross sections and whether we care about a smooth
            # transition between residues.
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
            need_twist[-1] = False

            # Perform any smoothing (e.g., strand smoothing
            # to remove lasagna sheets, pipes and planks
            # display as cylinders and planes, etc.)
            tethered = zeros(len(atoms), bool)
            self._smooth_ribbon(residues, coords, guides, atoms, ssids,
                                tethered, xs_front, xs_back, p,
                                helix_ranges, sheet_ranges)
            tethered &= displays
            if False:
                # Debugging code to display line from control point to guide
                cp = p.new_drawing(str(self) + " control points")
                from .. import surface
                va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
                cp.geometry = va, ta
                cp.normals = na
                from numpy import empty, float32
                cp_radii = empty(len(coords), float)
                cp_radii.fill(0.1)
                cp.positions = _tether_placements(coords, guides, cp_radii, self.TETHER_CYLINDER)
                cp_colors = empty((len(coords), 4), float)
                cp_colors[:] = (255,0,0,255)
                cp.colors = cp_colors

            # Generate ribbon
            any_ribbon = True
            ribbon = Ribbon(coords, guides, self.ribbon_orientation)
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
            rd = self._level_of_detail._ribbon_divisions
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
            if displays[0] and not (is_helix[0] and self.ribbon_mode_helix == self.RIBBON_MODE_ARC):
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
                self._ribbon_r2t[residues[0]] = triangle_range
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
                if is_helix[i] and is_helix[i-1] and self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
                    # Helix is shown separately as a tube, so we do not need to
                    # draw anything
                    next_cap = True
                    prev_band = None
                else:
                    # Show as ribbon
                    seg = capped and seg_cap or seg_blend
                    mid_cap = not self.ribbon_xs_mgr.is_compatible(xs_front[i], xs_back[i])
                    front_c, front_t, front_n = ribbon.segment(i - 1, ribbon.BACK, seg, mid_cap, last=mid_cap)
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
                if is_helix[i] and is_helix[i+1] and self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
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
                    self._ribbon_r2t[residues[i]] = triangle_range
                    t_start = t_end
            # Last residue
            if displays[-1] and not (is_helix[-1] and self.ribbon_mode_helix == self.RIBBON_MODE_ARC):
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
                self._ribbon_r2t[residues[-1]] = triangle_range
                t_start = t_end

            # Create drawing from arrays
            rp.display = True
            rp.vertices = concatenate(vertex_list)
            rp.normals = concatenate(normal_list)
            rp.triangles = concatenate(triangle_list)
            rp.vertex_colors = concatenate(color_list)
            # Save mappings for picking
            self._ribbon_t2r[rp] = t2r

            # Create tethers if necessary
            from numpy import any
            m = residues[0].structure
            if m.ribbon_tether_scale > 0 and any(tethered):
                tp = p.new_drawing(str(self) + " ribbon_tethers")
                nc = m.ribbon_tether_sides
                from .. import surface
                if m.ribbon_tether_shape == self.TETHER_CYLINDER:
                    va, na, ta = surface.cylinder_geometry(nc=nc, nz=2, caps=False)
                else:
                    # Assume it's either TETHER_CONE or TETHER_REVERSE_CONE
                    va, na, ta = surface.cone_geometry(nc=nc, caps=False)
                tp.geometry = va, ta
                tp.normals = na
                self._ribbon_tether.append((atoms, tp, coords[tethered], atoms.filter(tethered),
                                            m.ribbon_tether_shape, m.ribbon_tether_scale))
            else:
                self._ribbon_tether.append((atoms, None, None, None, None, None))

            # Create spine if necessary
            if self.ribbon_show_spine:
                sp = p.new_drawing(str(self) + " spine")
                from .. import surface
                va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=True)
                sp.geometry = va, ta
                sp.normals = na
                from numpy import empty, float32
                spine_radii = empty(len(spine_colors), float32)
                spine_radii.fill(0.3)
                sp.positions = _tether_placements(spine_xyz1, spine_xyz2, spine_radii, self.TETHER_CYLINDER)
                sp.colors = spine_colors
        self._graphics_changed |= self._SHAPE_CHANGE
        from .molarray import Residues
        self._ribbon_selected_residues = Residues()

    def _smooth_ribbon(self, rlist, coords, guides, atoms, ssids, tethered,
                       xs_front, xs_back, p, helix_ranges, sheet_ranges):
        ribbon_adjusts = rlist.ribbon_adjusts
        if self.ribbon_mode_helix == self.RIBBON_MODE_DEFAULT:
            # Smooth helices
            # XXX: Skip helix smoothing for now since it does not work well for bent helices
            pass
            # for start, end in helix_ranges:
            #     self._smooth_helix(coords, guides, tethered, xs_front, xs_back,
            #                        ribbon_adjusts, start, end, p)
        elif self.ribbon_mode_helix == self.RIBBON_MODE_ARC:
            for start, end in helix_ranges:
                self._arc_helix(rlist, coords, guides, ssids, tethered, xs_front, xs_back,
                                ribbon_adjusts, start, end, p)
        if self.ribbon_mode_strand == self.RIBBON_MODE_DEFAULT:
            # Smooth strands
            for start, end in sheet_ranges:
                self._smooth_strand(rlist, coords, guides, tethered, xs_front, xs_back,
                                    ribbon_adjusts, start, end, p)
        elif self.ribbon_mode_strand == self.RIBBON_MODE_ARC:
            # TODO: not implemented yet
            pass
            # for start, end in sheet_ranges:
            #     self._arc_strand(coords, guides, tethered, xs_front, xs_back,
            #                      ribbon_adjusts, start, end, p)

    def _smooth_helix(self, rlist, coords, guides, tethered, xs_front, xs_back,
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
        # Update the tethered array (we compare against self.bond_radius
        # because we want to create cones for the "worst" case which is
        # when the atoms are displayed in stick mode, with radius self.bond_radius)
        tethered[start:end] = norm(offsets, axis=1) > self.bond_radius

    def _arc_helix(self, rlist, coords, guides, ssids, tethered, xs_front, xs_back,
                   ribbon_adjusts, start, end, p):
        # Only bother if at least one residue is displayed
        displays = rlist.ribbon_displays
        if not any(displays[start:end]):
            return

        from .sse import HelixCylinder
        from numpy import linspace, cos, sin
        from math import pi
        from numpy import empty, tile

        hc = HelixCylinder(coords[start:end])
        centers = hc.cylinder_centers()
        radius = hc.cylinder_radius()
        normals, binormals = hc.cylinder_normals()
        icenters, inormals, ibinormals = hc.cylinder_intermediates()
        coords[start:end] = centers
        tethered[start:end] = True

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
                num_vertices += 4 * num_ptes
                num_triangles += 2 * cap_triangles + band_triangles
        elif was_displayed:
            # back cap
            num_vertices += num_pts
            num_triangles += cap_triangles

        # Second, create containers for vertices, normals and triangles
        va = empty((num_vertices, 3))
        na = empty((num_vertices, 3))
        ca = empty((num_vertices, 4))
        ta = empty((num_triangles, 3))

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
        if displays[start]:
            add_front_cap(start)
            add_back_band(start)
        was_displayed = displays[start]
        for i in range(start+1, end-1):
            if displays[i]:
                if not was_displayed:
                    add_front_cap(i)
                add_both_bands(i)
            else:
                if was_displayed:
                    add_back_cap(i-1)
            was_displayed = displays[i]
        # last residue
        if displays[end-1]:
            if was_displayed:
                add_front_band(end-1)
                add_back_cap(end-1)
            else:
                add_front_cap(end-1)
                add_front_band(end-1)
                add_back_cap(end-1)
        elif was_displayed:
            add_back_cap(end-2)

        # Finally, create graphics object of vertices, normals,
        # colors and triangles
        name = "helix-%d" % ssids[start]
        ssp = p.new_drawing(name)
        ssp.geometry = va, ta
        ssp.normals = na
        ssp.vertex_colors = ca

    def _smooth_strand(self, rlist, coords, guides, tethered, xs_front, xs_back,
                       ribbon_adjusts, start, end, p):
        if (end - start + 1) <= 2:
            # Short strands do not need smoothing
            return
        from numpy import empty, dot, newaxis
        from numpy.linalg import norm
        from .ribbon import normalize
        ss_coords = coords[start:end]
        adjusts = ribbon_adjusts[start:end][:, newaxis]
        ideal = empty(ss_coords.shape, dtype=float)
        if len(ideal) == 2:
            # Two-residue strand, no smoothing
            ideal[0] = ss_coords[0]
            ideal[-1] = ss_coords[-1]
        else:
            ideal[1:-1] = (ss_coords[1:-1] * 2 + ss_coords[:-2] + ss_coords[2:]) / 4
            ideal[0] = ss_coords[0] - (ideal[1] - ss_coords[1])
            ideal[-1] = ss_coords[-1] - (ideal[-2] - ss_coords[-2])
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
        # Update the tethered array
        tethered[start:end] = norm(offsets, axis=1) > self.bond_radius

    def _ss_axes(self, ss_coords):
        from numpy import mean, argmax
        from numpy.linalg import svd
        centroid = mean(ss_coords, axis=0)
        rel_coords = ss_coords - centroid
        ignore, vals, vecs = svd(rel_coords)
        axes = vecs[argmax(vals)]
        return axes, centroid, rel_coords

    def _ss_display(self, p, name, centers):
        ssp = p.new_drawing(name)
        from .. import surface
        va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
        ssp.geometry = va, ta
        ssp.normals = na
        from numpy import empty, float32
        ss_radii = empty(len(centers) - 1, float32)
        ss_radii.fill(0.2)
        ssp.positions = _tether_placements(centers[:-1], centers[1:], ss_radii, self.TETHER_CYLINDER)
        ss_colors = empty((len(ss_radii), 4), float32)
        ss_colors[:] = (0,255,0,255)
        ssp.colors = ss_colors

    def _ss_guide_display(self, p, name, centers, guides):
        ssp = p.new_drawing(name)
        from .. import surface
        va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
        ssp.geometry = va, ta
        ssp.normals = na
        from numpy import empty, float32
        ss_radii = empty(len(centers), float32)
        ss_radii.fill(0.4)
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
        hide = self._ribbon_selected_residues - rsel
        keep = self._ribbon_selected_residues & rsel
        show = rsel - self._ribbon_selected_residues
        self._ribbon_selected_residues = keep | show
        # Change the selected triangles in drawings
        da = {}         # actions - 0=hide, 1=keep, 2=show
        residues = [hide, keep, show]
        # Partition by drawing
        for i in range(len(residues)):
            for r in residues[i]:
                try:
                    tr = self._ribbon_r2t[r]
                except KeyError:
                    continue
                try:
                    a = da[tr.drawing]
                except KeyError:
                    a = da[tr.drawing] = ([], [], [])
                a[i].append((tr.start, tr.end))
        for p, residues in da.items():
            if not residues[1] and not residues[2]:
                # No residues being kept or added
                p.selected_triangles_mask = None
            else:
                m = p.selected_triangles_mask
                if m is None:
                    import numpy
                    m = numpy.zeros((p.number_of_triangles(),), bool)
                for start, end in residues[0]:
                    m[start:end] = False
                for start, end in residues[2]:
                    m[start:end] = True
                p.selected_triangles_mask = m

    def _update_ribbon_tethers(self):
        from numpy import around
        for all_atoms, tp, xyz1, tether_atoms, shape, scale in self._ribbon_tether:
            all_atoms.update_ribbon_visibility()
            if tp:
                xyz2 = tether_atoms.coords
                radii = self._atom_display_radii(tether_atoms) * scale
                tp.positions = _tether_placements(xyz1, xyz2, radii, shape)
                tp.display_positions = tether_atoms.visibles
                colors = tether_atoms.colors
                colors[:,3] = around(colors[:,3] * self.ribbon_tether_opacity).astype(int)
                tp.colors = colors

    def bounds(self, positions = True):
        # TODO: Cache bounds
        self._update_graphics_if_needed()       # Ribbon bound computed from graphics
        ab = self._atom_bounds()
        rb = self._ribbon_bounds()
        sb = tuple(s.bounds() for s in self.surfaces())
        from ..geometry import bounds
        b = bounds.union_bounds((ab, rb) + sb)
        if positions:
            b = bounds.copies_bounding_box(b, self.positions)
        return b

    def _atom_bounds(self):
        if not self._atom_bounds_needs_update:
            return self._cached_atom_bounds
        a = self.atoms
        disp = a.displays
        xyz = a.coords[disp]
        radii = a.radii[disp]
        from .. import geometry
        b = geometry.sphere_bounds(xyz, radii)
        self._cached_atom_bounds = b
        self._atom_bounds_needs_update = False
        return b

    def _ribbon_bounds(self):
        rd = self._ribbon_drawing
        if rd is None or not rd.display:
            return None
        return rd.bounds()

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and hasattr(self, exclude)):
            return None
        # TODO: check intercept of bounding box as optimization
        # TODO: Handle molecule placed at multiple positions
        xyz1, xyz2 = self.position.inverse() * (mxyz1, mxyz2)
        pa = self._atom_first_intercept(xyz1, xyz2)
        pb = self._bond_first_intercept(xyz1, xyz2)
        if pb and pa:
            a = pa.atom
            if a.draw_mode == a.STICK_STYLE and a in pb.bond.atoms:
                pb = None	# Pick atom if stick bond and its atom are picked.
        ppb = self._pseudobond_first_intercept(xyz1, xyz2)
        pr = self._ribbon_first_intercept(xyz1, xyz2)
        # Handle molecular surfaces
        ps = self.first_intercept_children(self.child_models(), mxyz1, mxyz2, exclude)
        picks = [pa, pb, ppb, pr, ps]

        # TODO: for now, tethers pick nothing, but it should either pick
        #       the residue or the guide atom.

        pclosest = None
        for p in picks:
            if p and (pclosest is None or p.distance < pclosest.distance):
                pclosest = p
        return pclosest

    def _atom_first_intercept(self, mxyz1, mxyz2):
        d = self._atoms_drawing
        if d is None or not d.display:
            return None

        xyzr = d.positions.shift_and_scale_array()
        dp = d.display_positions
        xyz,r = (xyzr[:,:3], xyzr[:,3]) if dp is None else (xyzr[dp,:3], xyzr[dp,3])

        # Check for atom sphere intercept
        from .. import geometry
        fmin, anum = geometry.closest_sphere_intercept(xyz, r, mxyz1, mxyz2)

        if fmin is None:
            return None

        if not dp is None:
            anum = dp.nonzero()[0][anum]    # Remap index to include undisplayed positions
        atom = self.atoms[anum]

        # Create pick object
        s = PickedAtom(atom, fmin)

        return s

    def _bond_first_intercept(self, mxyz1, mxyz2):
        d = self._bonds_drawing
        if d and d.display:
            b,f = _bond_intercept(self.bonds, mxyz1, mxyz2)
            if b:
                return PickedBond(b, f)
        return None

    def _pseudobond_first_intercept(self, mxyz1, mxyz2):
        fc = bc = None
        for pbg in self.pbg_map.values():
            d = pbg._pbond_drawing
            if d and d.display:
                b,f = _bond_intercept(pbg.pseudobonds, mxyz1, mxyz2)
                if f is not None and (fc is None or f < fc):
                    fc = f
                    bc = b
                    
        p = PickedPseudobond(bc, fc) if bc else None
        return p

    def _ribbon_first_intercept(self, mxyz1, mxyz2):
        pclosest = None
        for d, t2r in self._ribbon_t2r.items():
            if d.display:
                p = d.first_intercept(mxyz1, mxyz2)
                if p and (pclosest is None or p.distance < pclosest.distance):
                    from bisect import bisect_right
                    n = bisect_right(t2r, p.triangle_number)
                    if n > 0:
                        triangle_range = t2r[n - 1]
                        pclosest = PickedResidue(triangle_range.residue, p.distance)
        return pclosest

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and hasattr(self, exclude):
            return []

        p = self._atoms_planes_pick(planes)
        rp = self._ribbon_planes_pick(planes)
        if rp:
            p.extend(rp)
        return p

    def _atoms_planes_pick(self, planes):
        d = self._atoms_drawing
        if d is None or not d.display:
            return []

        xyz = d.positions.shift_and_scale_array()[:,:3]
        dp = d.display_positions
        if dp is not None:
            xyz = xyz[dp,:]

        picks = []
        from .. import geometry
        pmask = geometry.points_within_planes(xyz, planes)
        if pmask.sum() == 0:
            return []

        a = self.atoms
        if not dp is None:
            anum = dp.nonzero()[0][pmask]    # Remap index to include undisplayed positions
            atoms = a.filter(anum)
        else:
            atoms = a.filter(pmask)

        p = PickedAtoms(atoms)

        return [p]

    def _ribbon_planes_pick(self, planes):
        picks = []
        for d, t2r in self._ribbon_t2r.items():
            if d.display:
                rp = d.planes_pick(planes)
                from ..graphics import TrianglesPick
                for p in rp:
                    if isinstance(p, TrianglesPick) and p.drawing() is d:
                        tmask = p._triangles_mask
                        res = [rtr.residue for rtr in t2r if tmask[rtr.start:rtr.end].sum() > 0]
                        if res:
                            from .molarray import Residues
                            rc = Residues(res)
                            picks.append(PickedResidues(rc))
        return picks

    def set_selected(self, sel):
        self.atoms.selected = sel
        Model.set_selected(self, sel)
    selected = property(Model.get_selected, set_selected)

    def set_selected_positions(self, spos):
        self.atoms.selected = (spos is not None and spos.sum() > 0)
        Model.set_selected_positions(self, spos)
    selected_positions = property(Model.get_selected_positions, set_selected_positions)

    def selected_items(self, itype):
        if itype == 'atoms':
            atoms = self.atoms
            if atoms.num_selected > 0:
                return [atoms.filter(atoms.selected)]
        return []

    def any_part_selected(self):
        if self.atoms.num_selected > 0:
            return True
        for c in self.child_models():
            if c.any_part_selected():
                return True
        return False
        
    def clear_selection(self):
        self.selected = False
        self.atoms.selected = False

    def selection_promotion(self):
        atoms = self.atoms
        n = atoms.num_selected
        if n == 0 or n == len(atoms):
            return None
        asel = atoms.selected

        r = atoms.residues
        rids = r.unique_ids
        from numpy import unique, in1d
        sel_rids = unique(rids[asel])
        ares = in1d(rids, sel_rids)
        if ares.sum() > n:
            # Promote to entire residues
            level = 1004
            psel = ares
        else:
            ssids = r.secondary_structure_ids
            sel_ssids = unique(ssids[asel])
            ass = in1d(ssids, sel_ssids)
            if ass.sum() > n:
                # Promote to secondary structure
                level = 1003
                psel = ass
            else:
                from numpy import array
                cids = array(r.chain_ids)
                sel_cids = unique(cids[asel])
                ac = in1d(cids, sel_cids)
                if ac.sum() > n:
                    # Promote to entire chains
                    level = 1002
                    psel = ac
                else:
                    # Promote to entire molecule
                    level = 1001
                    ac[:] = True
                    psel = ac

        return PromoteAtomSelection(self, level, psel, asel)

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
        chain_ids = atoms.residues.chain_ids
        try:
            case_insensitive = self._atomspec_chain_ci
        except AttributeError:
            any_upper = any([c.isupper() for c in chain_ids])
            any_lower = any([c.islower() for c in chain_ids])
            case_insensitive = not any_upper or not any_lower
            self._atomspec_chain_ci = case_insensitive
        import numpy
        selected = numpy.zeros(num_atoms)
        # TODO: account for attrs in addition to parts
        for part in parts:
            if part.end is None:
                if case_insensitive:
                    def choose(chain_id, v=part.start.lower()):
                        return chain_id.lower() == v
                else:
                    def choose(chain_id, v=part.start):
                        return chain_id == v
            else:
                if case_insensitive:
                    def choose(chain_id, s=part.start.lower(), e=part.end.lower()):
                        cid = chain_id.lower()
                        return cid >= s and cid <= e
                else:
                    def choose(chain_id, s=part.start, e=part.end):
                        return chain_id >= s and chain_id <= e
            s = numpy.vectorize(choose)(chain_ids)
            selected = numpy.logical_or(selected, s)
        # print("AtomicStructure._atomspec_filter_chain", selected)
        return selected

    def _atomspec_filter_residue(self, atoms, num_atoms, parts, attrs):
        import numpy
        res_names = numpy.array(atoms.residues.names)
        res_numbers = atoms.residues.numbers
        res_ics = atoms.residues.insertion_codes
        selected = numpy.zeros(num_atoms)
        # TODO: account for attrs in addition to parts
        for part in parts:
            start_number, start_ic = self._res_parse(part.start)
            if part.end is None:
                end_number = None

                def choose_type(value, v=part.start.lower()):
                    return value.lower() == v
            else:
                end_number, end_ic = self._res_parse(part.end)

                def choose_type(value, s=part.start.lower(), e=part.end.lower()):
                    v = value.lower()
                    return v >= s and v <= e
            if start_number:
                if end_number is None:
                    def choose_id(n, ic, test_val=str(start_number)+start_ic):
                        return str(n)+ic == test_val
                else:
                    def choose_id(n, ic, sn=start_number, sic=start_ic, en=end_number, eic=end_ic):
                        if n < start_number or n > end_number:
                            return False
                        if n > start_number and n < end_number:
                            return True
                        if n == start_number:
                            if not ic and not sic:
                                return True
                            if ic and not sic:
                                # blank insertion code is before non-blanks
                                # res has insertion code, but test string doesn't...
                                return True
                            if sic and not ic:
                                # blank insertion code is before non-blanks
                                # test string has insertion code, but res doesn't...
                                return False
                            return sic <= ic
                        if n == end_number:
                            if not ic and not eic:
                                return True
                            if ic and not eic:
                                # blank insertion code is before non-blanks
                                # res has insertion code, but test string doesn't...
                                return False
                            if eic and not ic:
                                # blank insertion code is before non-blanks
                                # test string has insertion code, but res doesn't...
                                return True
                            return eic >= ic
                        return True
            else:
                choose_id = None
            s = numpy.vectorize(choose_type)(res_names)
            selected = numpy.logical_or(selected, s)
            if choose_id:
                s = numpy.vectorize(choose_id)(res_numbers, res_ics)
                selected = numpy.logical_or(selected, s)
        # print("AtomicStructure._atomspec_filter_residue", selected)
        return selected

    def _res_parse(self, n):
        try:
            return int(n), ""
        except ValueError:
            if not n:
                return None, ""
            try:
                return int(n[:-1]), n[-1]
            except ValueError:
                return None, ""

    def _atomspec_filter_atom(self, atoms, num_atoms, parts, attrs):
        import numpy
        names = numpy.array(atoms.names)
        selected = numpy.zeros(num_atoms)
        # TODO: account for attrs in addition to parts
        for part in parts:
            if part.end is None:
                def choose(name, v=part.start.lower()):
                    return name.lower() == v
            else:
                def choose(name, s=part.start.lower(), e=part.end.lower()):
                    n = name.lower()
                    return n >= s and n <= e
            s = numpy.vectorize(choose)(names)
            selected = numpy.logical_or(selected, s)
        # print("AtomicStructure._atomspec_filter_atom", selected)
        return selected

class AtomicStructure(Structure):
    """
    Bases: :class:`.StructureData`, :class:`.Model`, :class:`.Structure`

    Molecular model including atomic coordinates.
    The data is managed by the :class:`.StructureData` base class
    which provides access to the C++ structures.
    """

    @staticmethod
    def restore_snapshot(session, data):
        s = AtomicStructure(session, smart_initial_display = False)
        Structure.set_state_from_snapshot(s, session, data)
        return s

    def added_to_session(self, session):
        # run dssp?  Is it a protein with no SS assignments?
        if not self.ss_assigned:
            pas = self.residues.existing_principal_atoms
            if len(pas.residues.filter(pas.names=="CA")) > 0:
                session.logger.info("Model %s (%s) has no secondary structure assignments. "
                    ' Running <a href="help:user/commands/dssp.html">dssp</a>'
                    " using default settings." % (self.id_string(), self.name), is_html=True)
                session.logger.status("Computing secondary structure assignments...")
                from ..commands.dssp import compute_ss
                try:
                    compute_ss(session, self)
                except ValueError as e:
                    if "normalize" in str(e):
                        msg = "Unable to compute secondary structure assigments" \
                            " due to degenerate geometry in structure"
                        session.logger.status(msg, color="red")
                        session.logger.warning(msg)
                    else:
                        raise
                else:
                    session.logger.status("Computed secondary structure assignments (see log)")
        super().added_to_session(session)
        self._set_chain_descriptions(session)
        self._determine_het_res_descriptions(session)

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
        chain_to_desc = {}
        if 'pdbx_poly_seq_scheme' in self.metadata:
            scheme = self.metadata.get('pdbx_poly_seq_scheme', [])
            id_to_index = {}
            for sch in scheme:
                id_to_index[sch['pdb_strand_id']] = int(sch['entity_id']) - 1
            entity = self.metadata.get('entity', [])
            entity_name_com = self.metadata.get('entity_name_com', [])
            name_com_lookup = {}
            for enc in entity_name_com:
                name_com_lookup[int(enc['entity_id'])-1] = enc['name']
            for chain_id, index in id_to_index.items():
                description = None
                # try SYNONYM equivalent first
                if index in name_com_lookup:
                    syn = name_com_lookup[index]
                    if syn != '?':
                        description = syn
                        synonym = True
                if not description and len(entity) > index:
                    description = entity[index]['pdbx_description']
                    synonym = False
                if description:
                    chain_to_desc[chain_id] = (description, synonym)
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
            from .pdb import process_chem_name
            for k, v in chain_to_desc.items():
                description, synonym = v
                chain_to_desc[k] = process_chem_name(description, probable_abbrs=synonym)
            chains = sorted(self.chains, key=lambda c: c.chain_id)
            for chain in chains:
                chain.description = chain_to_desc.get(chain.chain_id, None)
                if chain.description:
                    session.logger.info("%s, chain %s: %s" % (self, chain.chain_id,
                        chain.description))


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
        from ..models import MODEL_DISPLAY_CHANGED
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
            n = sum(tuple(m.num_atoms_visible for m in s if m.visible))
            if n > 0 and n != self.num_atoms_shown:
                self.num_atoms_shown = n
                self._update_level_of_detail()
            self._need_update = False
            
    def _update_level_of_detail(self):
        n = self.num_atoms_shown
        for m in self._structures:
            if m.display:
                m._update_level_of_detail(n)

    def _array(self):
        sa = self._structures_array
        if sa is None:
            from .molarray import StructureDatas, object_pointers
            self._structures_array = sa = StructureDatas(object_pointers(self._structures))
        return sa

    def set_subdivision(self, subdivision):
        self.level_of_detail.quality = subdivision
        self._update_level_of_detail()

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
        self._atom_max_total_triangles = 10000000
        self._step_factor = 1.2
        self._sphere_geometries = {}	# Map ntri to (va,na,ta)

        self._bond_min_triangles = 24
        self._bond_max_triangles = 160
        self._bond_default_triangles = 60
        self._bond_max_total_triangles = 5000000
        self._cylinder_geometries = {}	# Map ntri to (va,na,ta)

        self._ribbon_divisions = 20

    def take_snapshot(self, session, flags):
        return {'quality': self.quality,
                'version': 1}

    @staticmethod
    def restore_snapshot(session, data):
        lod = LevelOfDetail()
        lod.quality = data['quality']
        return lod

    def reset_state(self):
        self.quality = 1

    def set_atom_sphere_geometry(self, drawing, natoms = None):
        if natoms == 0:
            return
        ntri = self.atom_sphere_triangles(natoms)
        ta = drawing.triangles
        if ta is None or len(ta) != ntri:
            # Update instanced sphere triangulation
            w = len(ta) if ta is not None else 0
            va, na, ta = self.sphere_geometry(ntri)
            drawing.vertices = va
            drawing.normals = na
            drawing.triangles = ta

    def sphere_geometry(self, ntri):
        # Cache sphere triangulations of different sizes.
        sg = self._sphere_geometries
        if not ntri in sg:
            from ..geometry.sphere import sphere_triangulation
            va, ta = sphere_triangulation(ntri)
            sg[ntri] = (va,va,ta)
        return sg[ntri]

    def atom_sphere_triangles(self, natoms):
        if natoms is None:
            return self._atom_default_triangles
        ntri = self.quality * self._atom_max_total_triangles // natoms
        nmin, nmax = self._atom_min_triangles, self._atom_max_triangles
        ntri = self.clamp_geometric(ntri, nmin, nmax)
        ntri = 2*(ntri//2)	# Require multiple of 2.
        return ntri

    def clamp_geometric(self, n, nmin, nmax):
        f = self._step_factor
        from math import log, pow
        n1 = int(nmin*pow(f,int(log(n/nmin,f))))
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
            drawing.vertices = va
            drawing.normals = na
            drawing.triangles = ta

    def cylinder_geometry(self, div):
        # Cache cylinder triangulations of different sizes.
        cg = self._cylinder_geometries
        if not div in cg:
            from .. import surface
            cg[div] = surface.cylinder_geometry(nc = div, caps = False, height = 0.5)
        return cg[div]

    def bond_cylinder_triangles(self, nbonds):
        if nbonds is None:
            return self._bond_default_triangles
        ntri = self.quality * self._bond_max_total_triangles // nbonds
        nmin, nmax = self._bond_min_triangles, self._bond_max_triangles
        ntri = self.clamp_geometric(ntri, nmin, nmax)
        ntri = 4*(ntri//4)	# Require multiple of 4
        return ntri

# -----------------------------------------------------------------------------
#
from ..selection import SelectionPromotion
class PromoteAtomSelection(SelectionPromotion):
    def __init__(self, structure, level, atom_sel_mask, prev_sel_mask):
        SelectionPromotion.__init__(self, level)
        self._structure = structure
        self._atom_sel_mask = atom_sel_mask
        self._prev_sel_mask = prev_sel_mask
    def promote(self):
        self._structure.atoms.selected = self._atom_sel_mask
    def demote(self):
        self._structure.atoms.selected = self._prev_sel_mask

# -----------------------------------------------------------------------------
#
from ..graphics import Pick
class PickedAtom(Pick):
    def __init__(self, atom, distance):
        Pick.__init__(self, distance)
        self.atom = atom
    def description(self):
        return str(self.atom)
    @property
    def residue(self):
        return self.atom.residue
    def select(self, toggle = False):
        a = self.atom
        a.selected = (not a.selected) if toggle else True

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
    def select(self, toggle = False):
        a = self.atoms
        if toggle:
            from numpy import logical_not
            a.selected = logical_not(a.selected)
        else:
            a.selected = True

# -----------------------------------------------------------------------------
# Handles bonds and pseudobonds.
#
def _bond_intercept(bonds, mxyz1, mxyz2):

    bshown = bonds.showns
    bs = bonds.filter(bshown)
    a1, a2 = bs.atoms
    xyz1, xyz2, r = a1.coords, a2.coords, bs.radii

    # Check for atom sphere intercept
    from .. import geometry
    f, bnum = geometry.closest_cylinder_intercept(xyz1, xyz2, r, mxyz1, mxyz2)

    if f is None:
        return None, None

    # Remap index to include undisplayed positions
    bnum = bshown.nonzero()[0][bnum]

    return bonds[bnum], f

# -----------------------------------------------------------------------------
#
class PickedBond(Pick):
    def __init__(self, bond, distance):
        Pick.__init__(self, distance)
        self.bond = bond
    def description(self):
        return str(self.bond)
    @property
    def residue(self):
        a1, a2 = self.bond.atoms
        if a1.residue == a2.residue:
            return a1.residue
        return None
    def select(self, toggle = False):
        for a in self.bond.atoms:
            a.selected = (not a.selected) if toggle else True

# -----------------------------------------------------------------------------
#
class PickedPseudobond(Pick):
    def __init__(self, pbond, distance):
        Pick.__init__(self, distance)
        self.pbond = pbond
    def description(self):
        return str(self.pbond)
    @property
    def residue(self):
        a1, a2 = self.pbond.atoms
        if a1.residue == a2.residue:
            return a1.residue
        return None
    def select(self, toggle = False):
        for a in self.pbond.atoms:
            a.selected = (not a.selected) if toggle else True

# -----------------------------------------------------------------------------
#
from ..graphics import Pick
class PickedResidue(Pick):
    def __init__(self, residue, distance):
        Pick.__init__(self, distance)
        self.residue = residue
    def description(self):
        return str(self.residue)
    def select(self, toggle=False):
        atoms = self.residue.atoms
        atoms.selected = (not atoms.selected.any()) if toggle else True

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
    def select(self, toggle = False):
        a = self.residues.atoms
        if toggle:
            from numpy import logical_not
            a.selected = logical_not(a.selected)
        else:
            a.selected = True

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

  from ..geometry import cylinder_rotations
  cylinder_rotations(axyz0, axyz1, radii, p)

  p[:,3,:3] = 0.5*(axyz0 + axyz1)

  from ..geometry import Places
  pl = Places(opengl_array = p)
  return pl

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking two prototype cylinders to each bond location.
#
def _halfbond_cylinder_placements(axyz0, axyz1, radii):

  n = len(axyz0)
  from numpy import empty, float32
  p = empty((2*n,4,4), float32)
  
  from ..geometry import cylinder_rotations
  cylinder_rotations(axyz0, axyz1, radii, p[:n,:,:])
  p[n:,:,:] = p[:n,:,:]

  # Translations
  p[:n,3,:3] = 0.75*axyz0 + 0.25*axyz1
  p[n:,3,:3] = 0.25*axyz0 + 0.75*axyz1

  from ..geometry import Places
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
  
  from ..geometry import cylinder_rotations_x3d
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
def _selected_bond_cylinders(bond_atoms):
    ba1, ba2 = bond_atoms
    if ba1.num_selected > 0 and ba2.num_selected > 0:
        from numpy import logical_and, concatenate
        sel = logical_and(ba1.selected,ba2.selected)
        sel = concatenate((sel,sel))
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
def all_atoms(session):
    '''All atoms in all structures as an :class:`.Atoms` collection.'''
    return structure_atoms(all_structures(session))

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
        for a in m.selected_items('atoms'):
            blist.append(a.inter_bonds)
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
