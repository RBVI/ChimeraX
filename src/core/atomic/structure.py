# vim: set expandtab shiftwidth=4 softtabstop=4:
from .. import io
from ..models import Model
from ..state import State
from ..session import RestoreError
from .molobject import AtomicStructureData

CATEGORY = io.STRUCTURE


class AtomicStructure(AtomicStructureData, Model):
    """
    Bases: :class:`.AtomicStructureData`, :class:`.Model`

    Molecular model including atomic coordinates.
    The data is managed by the :class:`.AtomicStructureData` base class
    which provides access to the C++ structures.
    """

    ATOMIC_COLOR_NAMES = ["tan", "sky blue", "plum", "light green",
        "salmon", "light gray", "deep pink", "gold", "dodger blue", "purple"]

    STRUCTURE_STATE_VERSION = 0

    def __init__(self, session, *, name = "structure", c_pointer = None, restore_data = None,
                 level_of_detail = None, smart_initial_display = True):
        # Cross section coordinates are 2D and counterclockwise
        # Use C++ version of XSection instead of Python version
        from .molobject import RibbonXSection as XSection
        from .molarray import Residues
        from numpy import array
        # from .ribbon import XSection
        xsc_helix = array([( 0.5, 0.1),(0.0, 0.2),(-0.5, 0.1),(-0.6,0.0),
                           (-0.5,-0.1),(0.0,-0.2),( 0.5,-0.1),( 0.6,0.0)]) * 1.5
        xsc_strand = array([(0.5,0.1),(-0.5,0.1),(-0.5,-0.1),(0.5,-0.1)]) * 1.5
        xsc_turn = array([(0.1,0.1),(-0.1,0.1),(-0.1,-0.1),(0.1,-0.1)]) * 1.5
        xsc_arrow_head = array([(1.0,0.1),(-1.0,0.1),(-1.0,-0.1),(1.0,-0.1)]) * 1.5
        xsc_arrow_tail = array([(0.1,0.1),(-0.1,0.1),(-0.1,-0.1),(0.1,-0.1)]) * 1.5
       
        # attrs that should be saved in sessions, along with their initial values...
        self._session_attrs = {
            'ball_scale': 0.3,		# Scales sphere radius in ball and stick style
            'bond_radius': 0.2,
            'pseudobond_radius': 0.05,
            '_level_of_detail': LevelOfDetail() if level_of_detail is None else level_of_detail,
            #'_ribbon_selected_residues': Residues(),
        }

        if restore_data:
            # from session
            (tool_info, version, as_data) = restore_data
            model_data, python_data, c_data = as_data
            #
            # Model will attempt to restore self.name, which is a property of the C++
            # layer for an AtomicStructure, so initialize AtomicStructureData first...
            AtomicStructureData.__init__(self, logger=session.logger)
            for attr_name, default_val in self._session_attrs.items():
                setattr(self, attr_name, python_data.get(attr_name, default_val))
            Model.restore_snapshot_init(self, session, tool_info, *model_data)
            #as_version, ints, floats, misc = c_data
            self.session_restore(*c_data)
            self._smart_initial_display = False
        else:
            AtomicStructureData.__init__(self, c_pointer)
            for attr_name, val in self._session_attrs.items():
                setattr(self, attr_name, val)
            Model.__init__(self, name, session)
            self._smart_initial_display = smart_initial_display

        # for now, restore attrs to default initial values even for sessions...
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._pseudobond_group_drawings = {}    # Map PseudobondGroup to drawing
        self._cached_atom_bounds = None
        self._atom_bounds_needs_update = True
        self._ribbon_drawing = None
        self._ribbon_t2r = {}         # ribbon triangles-to-residue map
        self._ribbon_r2t = {}         # ribbon residue-to-triangles map
        self._ribbon_tether = []      # ribbon tethers from ribbon to floating atoms
        self._ribbon_xs_helix = XSection(xsc_helix, faceted=False)
        self._ribbon_xs_strand = XSection(xsc_strand, faceted=True)
        self._ribbon_xs_strand_start = XSection(xsc_turn, xsc_strand, faceted=True)
        self._ribbon_xs_turn = XSection(xsc_turn, faceted=True)
        self._ribbon_xs_arrow = XSection(xsc_arrow_head, xsc_arrow_tail, faceted=True)
        # TODO: move back into _session_attrs when Collection instances
        # handle session saving/restoring
        self._ribbon_selected_residues = Residues()

        from . import molobject
        molobject.add_to_object_map(self)

        self._ses_handlers = [
            self.session.triggers.add_handler("begin save session", self._begin_ses_save),
            self.session.triggers.add_handler("end save session", self._end_ses_save)
        ]
        self._make_drawing()

    def delete(self):
        '''Delete this structure.'''
        AtomicStructureData.delete(self)
        for handler in self._ses_handlers:
            self.session.triggers.delete_handler(handler)
        Model.delete(self)

    def copy(self, name):
        '''
        Return a copy of this structure with a new name.
        No atoms or other components of the structure
        are shared between the original and the copy.
        '''
        m = AtomicStructure(AtomicStructureData._copy(self), name = name,
                            level_of_detail = self._level_of_detail)
        m.positions = self.positions
        return m

    def added_to_session(self, session):
        if self._smart_initial_display:
            color = self.initial_color(session.main_view.background_color)
            self.set_color(color.uint8x4())

            atoms = self.atoms
            if self.num_chains == 0:
                lighting = "default"
                from .molobject import Atom, Bond
                atoms.draw_modes = Atom.STICK_STYLE
                from ..colors import element_colors
                het_atoms = atoms.filter(atoms.element_numbers != 6)
                het_atoms.colors = element_colors(het_atoms.element_numbers)
            elif self.num_chains < 5:
                lighting = "default"
                from .molobject import Atom, Bond
                atoms.draw_modes = Atom.STICK_STYLE
                from ..colors import element_colors
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
                    display.atoms.displays = True
                    ribbonable.ribbon_displays = True

            elif self.num_chains < 250:
                lighting = "full"
                from ..colors import chain_colors
                residues = self.residues
                residues.ribbon_colors = chain_colors(residues.chain_ids)
                atoms.colors = chain_colors(atoms.residues.chain_ids)
            else:
                lighting = "shadows true"
            from ..commands import Command
            if len([m for m in session.models.list()
                    if isinstance(m, self.__class__)]) == 1:
                Command(session, "lighting " + lighting, final=True).execute(log=False)

        self._start_change_tracking(session.change_tracker)
        self.handler = session.triggers.add_handler('graphics update', self._update_graphics_if_needed)

    def removed_from_session(self, session):
        session.triggers.delete_handler(self.handler)

    def take_snapshot(self, session, flags):
        from ..state import CORE_STATE_VERSION
        # TODO: also need to save this class's own state
        ints = []
        floats = []
        misc = []
        as_version = self.session_info(ints, floats, misc)
        return CORE_STATE_VERSION, [
            Model.take_snapshot(self, session, flags),
            { attr_name: getattr(self, attr_name) for attr_name in self._session_attrs.keys() },
            (as_version, ints, floats, misc)
        ]

    def restore_snapshot_init(self, session, tool_info, version, data):
        AtomicStructure.__init__(self, session, restore_data=(tool_info, version, data))

    def reset_state(self, session):
        pass

    def initial_color(self, bg_color):
        from ..colors import BuiltinColors, distinguish_from, Color
        try:
            cname = self.ATOMIC_COLOR_NAMES[self.id[0]-1]
            model_color = BuiltinColors[cname]
            if (model_color.rgba[:3] == bg_color[:3]).all():
                # force use of another color...
                raise IndexError("Same as background color")
        except IndexError:
            # pick a color that distinguishes from the standard list
            # as well as white and black and green (highlight), and hope...
            avoid = [BuiltinColors[cn].rgba[:3] for cn in self.ATOMIC_COLOR_NAMES]
            avoid.extend([(0,0,0), (0,1,0), (1,1,1), bg_color[:3]])
            model_color = Color(distinguish_from(avoid, num_candidates=7, seed=14))
        return model_color

    def _make_drawing(self):
        # Create graphics
        self._update_atom_graphics(self.atoms)
        self._update_bond_graphics(self.bonds)
        for name, pbg in self.pbg_map.items():
            self._update_pseudobond_graphics(name, pbg)
        self._create_ribbon_graphics()

    def set_subdivision(self, subdivision):
        self._level_of_detail.quality = subdivision
        self._update_graphics()

    def new_atoms(self):
        # TODO: Handle instead with a C++ notification that atoms added or deleted
        self._atom_bounds_needs_update = True

    def _update_graphics_if_needed(self, *_):
        if self._gc_ribbon:
            # Do this before fetching bits because ribbon creation changes some
            # display and hide bits
            try:
                self._create_ribbon_graphics()
            finally:
                self._gc_ribbon = False
        # Molecule changes
        c, s, se = self._gc_color, self._gc_shape, self._gc_select
        # Check for pseudobond changes
        for pbg in self.pbg_map.values():
            c |= pbg._gc_color
            s |= pbg._gc_shape
            se |= pbg._gc_select
            pbg._gc_color = pbg._gc_shape = pbg._gc_select = False
        # Update graphics
        if c or s or se:
            self._gc_color = self._gc_shape = self._gc_select = False
            if s:
                self._update_ribbon_tethers()
            self._update_graphics()
            self.redraw_needed(shape_changed = s, selection_changed = se)
            if s:
                self._atom_bounds_needs_update = True

    def _update_graphics(self):
        self._update_atom_graphics(self.atoms)
        self._update_bond_graphics(self.bonds)
        for name, pbg in self.pbg_map.items():
            self._update_pseudobond_graphics(name, pbg)
        self._update_ribbon_graphics()

    def _update_atom_graphics(self, atoms):
        avis = atoms.visibles
        ndisp = avis.sum()
        p = self._atoms_drawing
        if p is None:
            if ndisp == 0:
                return
            self._atoms_drawing = p = self.new_drawing('atoms')

        # Update level of detail of spheres
        self._level_of_detail.set_atom_sphere_geometry(ndisp, p)

        # Set instanced sphere center position and radius
        n = len(atoms)
        from numpy import empty, float32, multiply
        xyzr = empty((n, 4), float32)
        xyzr[:, :3] = atoms.coords
        xyzr[:, 3] = self._atom_display_radii(atoms)

        from ..geometry import Places
        p.positions = Places(shift_and_scale=xyzr)
        p.display_positions = avis

        # Set atom colors
        p.colors = atoms.colors

        # Set selected
        p.selected_positions = atoms.selected if atoms.num_selected > 0 else None

    def _atom_display_radii(self, atoms):
        r = atoms.radii.copy()
        dm = atoms.draw_modes
        from .molobject import Atom
        r[dm == Atom.BALL_STYLE] *= self.ball_scale
        r[dm == Atom.STICK_STYLE] = self.bond_radius
        return r

    def _update_bond_graphics(self, bonds):
        nshown = bonds.num_shown
        p = self._bonds_drawing
        if p is None:
            if nshown == 0:
                return
            self._bonds_drawing = p = self.new_drawing('bonds')

        # Update level of detail of spheres
        self._level_of_detail.set_bond_cylinder_geometry(nshown, p)

        ba1, ba2 = bond_atoms = bonds.atoms
        p.positions = _halfbond_cylinder_placements(ba1.coords, ba2.coords, bonds.radii)
        p.display_positions = _shown_bond_cylinders(bonds)
        p.colors = c = bonds.half_colors
        p.selected_positions = _selected_bond_cylinders(bond_atoms)

    def _update_pseudobond_graphics(self, name, pbgroup):

        pg = self._pseudobond_group_drawings
        if pbgroup in pg:
            p = pg[pbgroup]
        else:
            pg[pbgroup] = p = self.new_drawing(name)
            va, na, ta = _pseudobond_geometry()
            p.geometry = va, ta
            p.normals = na

        pbonds = pbgroup.pseudobonds
        ba1, ba2 = bond_atoms = pbonds.atoms
        p.positions = _halfbond_cylinder_placements(ba1.coords, ba2.coords, pbonds.radii)
        p.display_positions = _shown_bond_cylinders(pbonds)
        p.colors = pbonds.half_colors
        p.selected_positions = _selected_bond_cylinders(bond_atoms)

    def _create_ribbon_graphics(self):
        from .ribbon import Ribbon
        from numpy import concatenate, array, zeros
        polymers = self.polymers(False, False)
        if self._ribbon_drawing is None:
            self._ribbon_drawing = p = self.new_drawing('ribbon')
            p.display = True
        else:
            p = self._ribbon_drawing
            p.remove_all_drawings()
        self._ribbon_t2r = {}
        self._ribbon_r2t = {}
        self._ribbon_tether = []
        import sys
        for rlist in polymers:
            rp = p.new_drawing(rlist.strs[0])
            t2r = []
            # Always call get_polymer_spline to make sure hide bits are
            # properly set when ribbons are completely undisplayed
            atoms, coords, guides = rlist.get_polymer_spline()
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
            if len(atoms) < 4:
                continue
            # Perform any smoothing (e.g., strand smoothing
            # to remove lasagna sheets, pipes and planks
            # display as cylinders and planes, etc.)
            tethered = zeros(len(atoms), bool)
            self._smooth_ribbon(residues, coords, guides, atoms, tethered, p)
            if False:
                # Debugging code to display line from control point to guide
                cp = p.new_drawing(rlist.strs[0] + " control points")
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
            ribbon = Ribbon(coords, guides)
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
            # Assign cross sections
            is_helix = residues.is_helix
            is_sheet = residues.is_sheet
            xss = []
            was_strand = False
            for i in range(len(residues)):
                if is_sheet[i]:
                    if was_strand:
                        xss.append(self._ribbon_xs_strand)
                    else:
                        xss.append(self._ribbon_xs_strand_start)
                        was_strand = True
                else:
                    if was_strand:
                        xss[-1] = self._ribbon_xs_arrow
                    if is_helix[i]:
                        xss.append(self._ribbon_xs_helix)
                    else:
                        xss.append(self._ribbon_xs_turn)
                    was_strand = False

            # Draw first and last residue differently because they
            # are each only a single half segment, while the middle
            # residues are each two half segments.

            # First residues
            if displays[0]:
                capped = displays[0] != displays[1] or xss[0] != xss[1]
                seg = capped and seg_cap or seg_blend
                centers, tangents, normals = ribbon.segment(0, ribbon.FRONT, seg)
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[0],
                                                                                     centers, normals,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                s = xss[0].extrude(centers, tangents, normals, colors[0],
                                   True, capped, v_start)
                v_start += len(s.vertices)
                t_end = t_start + len(s.triangles)
                vertex_list.append(s.vertices)
                normal_list.append(s.normals)
                triangle_list.append(s.triangles)
                color_list.append(s.colors)
                if displays[1] and xss[0] == xss[1]:
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
                if not displays[i]:
                    continue
                seg = capped and seg_cap or seg_blend
                front_c, front_t, front_n = ribbon.segment(i - 1, ribbon.BACK, seg)
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[0],
                                                                                     front_c, front_n,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                next_cap = displays[i] != displays[i + 1] or xss[i] != xss[i + 1]
                seg = next_cap and seg_cap or seg_blend
                back_c, back_t, back_n = ribbon.segment(i, ribbon.FRONT, seg)
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[0],
                                                                                     back_c, back_n,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                centers = concatenate((front_c, back_c))
                tangents = concatenate((front_t, back_t))
                normals = concatenate((front_n, back_n))
                s = xss[i].extrude(centers, tangents, normals, colors[i],
                                   capped, next_cap, v_start)
                v_start += len(s.vertices)
                t_end = t_start + len(s.triangles)
                vertex_list.append(s.vertices)
                normal_list.append(s.normals)
                triangle_list.append(s.triangles)
                color_list.append(s.colors)
                if prev_band is not None:
                    triangle_list.append(xss[i].blend(prev_band, s.front_band))
                    t_end += len(triangle_list[-1])
                if next_cap:
                    prev_band = None
                else:
                    prev_band = s.back_band
                capped = next_cap
                triangle_range = RibbonTriangleRange(t_start, t_end, rp, residues[i])
                t2r.append(triangle_range)
                self._ribbon_r2t[residues[i]] = triangle_range
                t_start = t_end
            # Last residue
            if displays[-1]:
                seg = capped and seg_cap or seg_blend
                centers, tangents, normals = ribbon.segment(ribbon.num_segments - 1, ribbon.BACK, seg, last=True)
                if self.ribbon_show_spine:
                    spine_colors, spine_xyz1, spine_xyz2 = self._ribbon_update_spine(colors[0],
                                                                                     centers, normals,
                                                                                     spine_colors,
                                                                                     spine_xyz1,
                                                                                     spine_xyz2)
                s = xss[-1].extrude(centers, tangents, normals, colors[-1],
                                    capped, True, v_start)
                v_start += len(s.vertices)
                t_end = t_start + len(s.triangles)
                vertex_list.append(s.vertices)
                normal_list.append(s.normals)
                triangle_list.append(s.triangles)
                color_list.append(s.colors)
                if prev_band is not None:
                    triangle_list.append(xss[-1].blend(prev_band, s.front_band))
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
                tp = p.new_drawing(residues.strs[0] + "_tethers")
                nc = m.ribbon_tether_sides
                from .. import surface
                if m.ribbon_tether_shape == AtomicStructureData.TETHER_CYLINDER:
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
                sp = p.new_drawing(rlist.strs[0] + " spine")
                from .. import surface
                va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=True)
                sp.geometry = va, ta
                sp.normals = na
                from numpy import empty, float32
                spine_radii = empty(len(spine_colors), float32)
                spine_radii.fill(0.3)
                sp.positions = _tether_placements(spine_xyz1, spine_xyz2, spine_radii, self.TETHER_CYLINDER)
                sp.colors = spine_colors
        self._gc_shape = True

    def _smooth_ribbon(self, rlist, coords, guides, atoms, tethered, p):
        from numpy import logical_and, logical_not
        from numpy import dot, newaxis, mean
        from numpy.linalg import norm
        import math
        from .ribbon import normalize, normalize_vector_array
        ribbon_adjusts = rlist.ribbon_adjusts
        # Smooth helices
        ss_ids = rlist.ss_ids
        helices = rlist.is_helix
#         for start, end in self._ss_ranges(helices, ss_ids, 8):
#             # We only "optimize" longer helices because short
#             # ones do not contain enough information to do
#             # things intelligently
#             ss_coords = coords[start:end]
#             adjusts = ribbon_adjusts[start:end][:, newaxis]
#             axis, centroid, rel_coords = self._ss_axes(ss_coords)
#             # Compute position of cylinder center corresponding to
#             # helix control point atoms
#             axis_pos = dot(rel_coords, axis)[:, newaxis]
#             cyl_centers = centroid + axis * axis_pos
#             if False:
#                 # Debugging code to display center of secondary structure
#                 self._ss_display(p, rlist.strs[0] + " helix " + str(start), cyl_centers)
#             # Compute radius of cylinder
#             spokes = ss_coords - cyl_centers
#             cyl_radius = mean(norm(spokes, axis=1))
#             # Compute smoothed position of helix control point atoms
#             ideal = cyl_centers + normalize_vector_array(spokes) * cyl_radius
#             offsets = adjusts * (ideal - ss_coords)
#             new_coords = ss_coords + offsets
#             # Compute guide atom position relative to control point atom
#             delta_guides = guides[start:end] - ss_coords
#             # Update both control point and guide coordinates
#             coords[start:end] = new_coords
#             # Move the guide location so that it forces the
#             # ribbon parallel to the axis
#             guides[start:end] = new_coords + axis
#             # Originally, we just update the guide location to
#             # the same relative place as before
#             #   guides[start:end] = new_coords + delta_guides
#             # Update the tethered array (we compare against self.bond_radius
#             # because we want to create cones for the "worst" case which is
#             # when the atoms are displayed in stick mode, with radius self.bond_radius)
#             tethered[start:end] = norm(offsets, axis=1) > self.bond_radius
        # Smooth strands
        strands = logical_and(rlist.is_sheet, logical_not(helices))
        for start, end in self._ss_ranges(strands, ss_ids, 4):
            ss_coords = coords[start:end]
            adjusts = ribbon_adjusts[start:end][:, newaxis]
            axis, centroid, rel_coords = self._ss_axes(ss_coords)
            # Compute position for strand control point atom on
            # axis by projection
            axis = normalize(axis)
            axis_pos = dot(rel_coords, axis)[:, newaxis]
            ideal = centroid + axis * axis_pos
            if False:
                # Debugging code to display center of secondary structure
                self._ss_display(p, rlist.strs[0] + " helix " + str(start), ideal)
            offsets = adjusts * (ideal - ss_coords)
            new_coords = ss_coords + offsets
            # Compute guide atom position relative to control point atom
            delta_guides = guides[start:end] - ss_coords
            # Update both control point and guide coordinates
            coords[start:end] = new_coords
            guides[start:end] = new_coords + delta_guides
            # Update the tethered array
            tethered[start:end] = norm(offsets, axis=1) > self.bond_radius

    def _ss_ranges(self, ba, ss_ids, min_length):
        # Return ranges of True in boolean array "ba"
        ranges = []
        start = -1
        start_id = None
        for n, bn in enumerate(ba):
            if bn:
                if start < 0:
                    start_id = ss_ids[n]
                    start = n
                elif ss_ids[n] != start_id:
                    if n - start >= min_length:
                        ranges.append((start, n))
                    start_id = ss_ids[n]
                    start = n
            else:
                if start >= 0:
                    if n - start >= min_length:
                        ranges.append((start, n))
                    start = -1
        if start >= 0 and len(ba) - start >= min_length:
            ranges.append((start, len(ba)))
        return ranges

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
        ps = self.first_intercept_children(self.child_models(), xyz1, xyz2, exclude)
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
        for pbg, d in self._pseudobond_group_drawings.items():
            if d.display:
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
                            rc = Residues(residues=res)
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

    def select_atom(self, atom, toggle=False, selected=True):
        '''
        Select or unselect a specified :class:`.Atom`.
        If selected is false then it unselects this atom.
        '''
        atom.selected = (not atom.selected) if toggle else selected
        self._selection_changed()

    def select_atoms(self, atoms, toggle=False, selected=True):
        '''
        Select or unselect :class:`.Atoms`.
        If selected is false then it unselects the atoms.
        '''
        a = self.atoms
        asel = a.selected
        m = a.mask(atoms)
        from numpy import logical_not
        asel[m] = logical_not(asel[m]) if toggle else selected
        self.select_mask_atoms(asel)

    def select_mask_atoms(self, atom_mask):
        self.atoms.selected = atom_mask
        self._selection_changed()

    def select_residue(self, residue, toggle=False, selected=True):
        '''
        Select a specified :class:`.Residue`.
        If selected is false then it unselects the residue.
        Selecting a residue is equivalent to select all the residue atoms.
        '''
        if toggle:
            selected = self._ribbon_selected_residues.index(residue) < 0
        self.select_atoms(residue.atoms, toggle=False, selected=selected)

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
        self._selection_changed()

    def _selection_changed(self):
        # Update selection on molecular surfaces
        # TODO: Won't work for surfaces spanning multiple molecules
        from .molsurf import MolecularSurface
        for s in self.child_drawings():
            if isinstance(s, MolecularSurface):
                s.update_selection()

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

    def _begin_ses_save(self, *args):
        self.session_save_setup()

    def _end_ses_save(self, *args):
        self.session_save_teardown()

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
        selected = numpy.zeros(num_atoms)
        # TODO: account for attrs in addition to parts
        for part in parts:
            start_number = self._number(part.start)
            if part.end is None:
                end_number = None

                def choose_type(value, v=part.start.lower()):
                    return value.lower() == v
            else:
                end_number = self._number(part.end)

                def choose_type(value, s=part.start.lower(), e=part.end.lower()):
                    v = value.lower()
                    return v >= s and v <= e
            if start_number:
                if end_number is None:
                    def choose_number(value, v=start_number):
                        return value == v
                else:
                    def choose_number(value, s=start_number, e=end_number):
                        return value >= s and value <= e
            else:
                choose_number = None
            s = numpy.vectorize(choose_type)(res_names)
            selected = numpy.logical_or(selected, s)
            if choose_number:
                s = numpy.vectorize(choose_number)(res_numbers)
                selected = numpy.logical_or(selected, s)
        # print("AtomicStructure._atomspec_filter_residue", selected)
        return selected

    def _number(self, n):
        try:
            return int(n)
        except ValueError:
            return None

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

# -----------------------------------------------------------------------------
#
class LevelOfDetail(State):

    def __init__(self, restore_data=None):
        if restore_data is not None:
            self.quality = restore_data[0]
        else:
            self.quality = 1

        self._atom_min_triangles = 10
        self._atom_max_triangles = 400
        self._atom_max_total_triangles = 10000000
        self._step_factor = 1.2
        self._sphere_geometries = {}	# Map ntri to (va,na,ta)

        self._bond_min_triangles = 24
        self._bond_max_triangles = 160
        self._bond_max_total_triangles = 5000000
        self._cylinder_geometries = {}	# Map ntri to (va,na,ta)

        self._ribbon_divisions = 10

    def take_snapshot(self, session, flags):
        return 1, [self.quality]

    def restore_snapshot_init(self, session, tool_info, version, data):
        self.__init__(restore_data=data)

    def reset_state(self):
        self.quality = 1

    def set_atom_sphere_geometry(self, natoms, drawing):
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

    def set_bond_cylinder_geometry(self, nbonds, drawing):
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
            cg[div] = surface.cylinder_geometry(nc = div, caps = False)
        return cg[div]

    def bond_cylinder_triangles(self, nbonds):
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
        self._structure.select_mask_atoms(self._atom_sel_mask)
    def demote(self):
        self._structure.select_mask_atoms(self._prev_sel_mask)

# -----------------------------------------------------------------------------
#
from ..graphics import Pick
class PickedAtom(Pick):
    def __init__(self, atom, distance):
        Pick.__init__(self, distance)
        self.atom = atom
    def description(self):
        return atom_description(self.atom)
    def select(self, toggle = False):
        a = self.atom
        a.structure.select_atom(a, toggle)

# -----------------------------------------------------------------------------
#
class PickedAtoms(Pick):
    def __init__(self, atoms):
        Pick.__init__(self)
        self.atoms = atoms
    def description(self):
        return '%d atoms' % len(self.atoms)
    def select(self, toggle = False):
        a = self.atoms
        if toggle:
            from numpy import logical_not
            a.selected = logical_not(a.selected)
        else:
            a.selected = True

# -----------------------------------------------------------------------------
#
def atom_description(atom):
    m = atom.structure
    r = atom.residue
    d = '%s #%s/%s %s %d %s' % (m.name, m.id_string(), r.chain_id, r.name, r.number, atom.name)
    return d

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
        return bond_description(self.bond)
    def select(self, toggle = False):
        for a in self.bond.atoms:
            a.structure.select_atom(a, toggle)

# -----------------------------------------------------------------------------
#
class PickedPseudobond(Pick):
    def __init__(self, pbond, distance):
        Pick.__init__(self, distance)
        self.pbond = pbond
    def description(self):
        return bond_description(self.pbond)
    def select(self, toggle = False):
        for a in self.pbond.atoms:
            a.structure.select_atom(a, toggle)

# -----------------------------------------------------------------------------
#
def bond_description(bond):
    a1, a2 = bond.atoms
    m1, m2 = a1.structure, a2.structure
    mid1, mid2 = m1.id_string(), m2.id_string()
    r1, r2 = a1.residue, a2.residue
    from .molobject import Bond
    t = 'bond' if isinstance(bond, Bond) else 'pseudobond'
    if r1 == r2:
        d = '%s %s #%s/%s %s %d %s - %s' % (t, m1.name, mid1, r1.chain_id,
                                            r1.name, r1.number, a1.name, a2.name)
    elif r1.chain_id == r2.chain_id:
        d = '%s %s #%s/%s %s %d %s - %s %d %s' % (t, m1.name, mid1, r1.chain_id,
                                                  r1.name, r1.number, a1.name,
                                                  r2.name, r2.number, a2.name)
    elif m1 == m2:
        d = '%s %s #%s/%s %s %d %s - /%s %s %d %s' % (t, m1.name, mid1,
                                                      r1.chain_id, r1.name, r1.number, a1.name,
                                                      r2.chain_id, r2.name, r2.number, a2.name)
    else:
        d = '%s %s #%s/%s %s %d %s - %s #%s/%s %s %d %s' % (t, m1.name, mid1, r1.chain_id, r1.name, r1.number, a1.name,
                                                            m2.name, mid2, r2.chain_id, r2.name, r2.number, a2.name)

    return d

# -----------------------------------------------------------------------------
#
from ..graphics import Pick
class PickedResidue(Pick):
    def __init__(self, residue, distance):
        Pick.__init__(self, distance)
        self.residue = residue
    def description(self):
        return residue_description(self.residue)
    def select(self, toggle=False):
        r = self.residue
        r.structure.select_residue(r, toggle)

# -----------------------------------------------------------------------------
#
class PickedResidues(Pick):
    def __init__(self, residues):
        Pick.__init__(self)
        self.residues = residues
    def description(self):
        return '%d residues' % len(self.residues)
    def select(self, toggle = False):
        a = self.residues.atoms
        if toggle:
            from numpy import logical_not
            a.selected = logical_not(a.selected)
        else:
            a.selected = True

# -----------------------------------------------------------------------------
#
def residue_description(r):
    m = r.structure
    d = '%s #%s.%s %s %d' % (m.name, m.id_string(), r.chain_id, r.name, r.number)
    return d

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
def _bond_cylinder_placements(axyz0, axyz1, radius):

  n = len(axyz0)
  from numpy import empty, float32, transpose, sqrt, array
  p = empty((n,4,4), float32)
  
  p[:,3,:] = (0,0,0,1)
  p[:,:3,3] = 0.5*(axyz0 + axyz1)

  v = axyz1 - axyz0
  d = sqrt((v*v).sum(axis = 1))
  for a in (0,1,2):
    v[:,a] /= d

  c = v[:,2]
  # TODO: Handle degenerate -z axis case
#  if c <= -1:
#    return ((1,0,0),(0,-1,0),(0,0,-1))      # Rotation by 180 about x
  wx, wy = -v[:,1],v[:,0]
  c1 = 1.0/(1+c)
  cx,cy = c1*wx, c1*wy
  r = radius
  h = d
  rs = array(((r*(cx*wx + c), r*cx*wy,  h*wy),
              (r*cy*wx, r*(cy*wy + c), -h*wx),
              (-r*wy, r*wx, h*c)), float32).transpose((2,0,1))
  p[:,:3,:3] = rs
  pt = transpose(p,(0,2,1))
  from ..geometry import Places
  pl = Places(opengl_array = pt)
  return pl

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking two prototype cylinders to each bond location.
#
def _halfbond_cylinder_placements(axyz0, axyz1, radius):

  n = len(axyz0)
  from numpy import empty, float32, transpose, sqrt, array
  p = empty((2*n,4,4), float32)
  
  p[:,3,:] = (0,0,0,1)
  p[:n,:3,3] = 0.75*axyz0 + 0.25*axyz1
  p[n:,:3,3] = 0.25*axyz0 + 0.75*axyz1

  v = axyz1 - axyz0
  d = sqrt((v*v).sum(axis = 1))
  for a in (0,1,2):
    v[:,a] /= d

  c = v[:,2]
  # TODO: Handle degenerate -z axis case
#  if c <= -1:
#    return ((1,0,0),(0,-1,0),(0,0,-1))      # Rotation by 180 about x
  wx, wy = -v[:,1],v[:,0]
  c1 = 1.0/(1+c)
  cx,cy = c1*wx, c1*wy
  r = radius
  h = 0.5*d
  rs = array(((r*(cx*wx + c), r*cx*wy,  h*wy),
              (r*cy*wx, r*(cy*wy + c), -h*wx),
              (-r*wy, r*wx, h*c)), float32).transpose((2,0,1))
  p[:n,:3,:3] = rs
  p[n:,:3,:3] = rs
  pt = transpose(p,(0,2,1))
  from ..geometry import Places
  pl = Places(opengl_array = pt)
  return pl

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
    if shape == AtomicStructureData.TETHER_REVERSE_CONE:
        return _bond_cylinder_placements(xyz1, xyz0, radius)
    else:
        return _bond_cylinder_placements(xyz0, xyz1, radius)

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9):
    from .. import surface
    return surface.dashed_cylinder_geometry(segments)

# -----------------------------------------------------------------------------
#
def all_atomic_structures(session):
    '''List of all :class:`.AtomicStructure` objects.'''
    return [m for m in session.models.list() if isinstance(m,AtomicStructure)]

# -----------------------------------------------------------------------------
#
def all_atoms(session):
    '''All atoms in all structures as an :class:`.Atoms` collection.'''
    return structure_atoms(all_atomic_structures(session))

# -----------------------------------------------------------------------------
#
def structure_atoms(structures):
    '''Return all atoms in specified atomic structures as an :class:`.Atoms` collection.'''
    from .molarray import Atoms
    atoms = Atoms()
    for m in structures:
        atoms = atoms | m.atoms
    return atoms

# -----------------------------------------------------------------------------
#
def selected_atoms(session):
    '''All selected atoms in all structures as an :class:`.Atoms` collection.'''
    from .molarray import Atoms
    atoms = Atoms()
    for m in session.models.list():
        if isinstance(m, AtomicStructure):
            for matoms in m.selected_items('atoms'):
                atoms = atoms | matoms
    return atoms

# -----------------------------------------------------------------------------
#
def structure_residues(structures):
    '''Return all residues in specified atomic structures as an :class:`.Atoms` collection.'''
    from .molarray import Residues
    res = Residues()
    for m in structures:
        res = res | m.residues
    return res
