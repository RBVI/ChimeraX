# vi: set expandtab shiftwidth=4 softtabstop=4:
from .. import io
from ..models import Model
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

    STRUCTURE_STATE_VERSION = 0

    def __init__(self, name, atomic_structure_pointer = None,
                 initialize_graphical_attributes = True, logger = None):

        AtomicStructureData.__init__(self, atomic_structure_pointer, logger)
        from . import molobject
        molobject.add_to_object_map(self)

        Model.__init__(self, name)

        self._atoms = None              # Cached atoms array

        self.ball_scale = 0.3		# Scales sphere radius in ball and stick style
        self.bond_radius = 0.2
        self.pseudobond_radius = 0.05
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._pseudobond_group_drawings = {}    # Map PseudobondGroup to drawing
        self.triangles_per_sphere = None
        self._cached_atom_bounds = None
        self._atom_bounds_needs_update = True

        self.ribbon_divisions = 10
        self._ribbon_drawing = None
        self._ribbon_t2r = {}         # ribbon triangles-to-residue map
        self._ribbon_r2t = {}         # ribbon residue-to-triangles map
        self._ribbon_tether = []      # ribbon tethers from ribbon to floating atoms
        # Cross section coordinates are 2D and counterclockwise
        from .ribbon import XSection
        xsc_helix = [( 0.5, 0.1),(0.0, 0.2),(-0.5, 0.1),(-0.6,0.0),
                     (-0.5,-0.1),(0.0,-0.2),( 0.5,-0.1),( 0.6,0.0)]
        xsc_strand = [(0.5,0.1),(-0.5,0.1),(-0.5,-0.1),(0.5,-0.1)]
        xsc_turn = [(0.1,0.1),(-0.1,0.1),(-0.1,-0.1),(0.1,-0.1)]
        xsc_arrow_head = [(1.0,0.1),(-1.0,0.1),(-1.0,-0.1),(1.0,-0.1)]
        xsc_arrow_tail = [(0.1,0.1),(-0.1,0.1),(-0.1,-0.1),(0.1,-0.1)]
        self._ribbon_xs_helix = XSection(xsc_helix, faceted=False)
        self._ribbon_xs_strand = XSection(xsc_strand, faceted=True)
        self._ribbon_xs_strand_start = XSection(xsc_turn, xsc_strand, faceted=True)
        self._ribbon_xs_turn = XSection(xsc_turn, faceted=True)
        self._ribbon_xs_arrow = XSection(xsc_arrow_head, xsc_arrow_tail, faceted=True)
        self._ribbon_selected_residues = set()

        self._make_drawing(initialize_graphical_attributes)

    def delete(self):
        '''Delete this structure.'''
        self._atoms = None
        AtomicStructureData.delete(self)
        Model.delete(self)

    def copy(self, name):
        '''
        Return a copy of this structure with a new name.
        No atoms or other components of the structure
        are shared between the original and the copy.
        '''
        m = AtomicStructure(name, AtomicStructureData._copy(self),
                            initialize_graphical_attributes = False)
        m.positions = self.positions
        return m

    def added_to_session(self, session):
        self.handler = session.triggers.add_handler('graphics update',
            self._update_graphics_if_needed)

    def removed_from_session(self, session):
        session.triggers.delete_handler(self.handler)

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

    @property
    def atoms(self):
        '''All atoms of the structure in an :py:class:`.Atoms` instance.'''
        if self._atoms is None:
            self._atoms = AtomicStructureData.atoms.fget(self)
        return self._atoms

    @property
    def pseudobond_groups(self):
        '''Dictionary mapping name to :class:`.PseudobondGroup` for pseudobond groups
        belonging to this structure. Read only.'''
        return self.pbg_map

    def shown_atom_count(self):
        '''Number of atoms displayed.'''
        na = sum(self.atoms.displays) if self.display else 0
        return na

    def _initialize_graphical_attributes(self):
        a = self.atoms
        from .molobject import Atom
        a.draw_modes = Atom.SPHERE_STYLE
        from ..colors import element_colors
        a.colors = element_colors(a.element_numbers)
        b = self.bonds
        b.colors = (170,170,170,255)
        b.radii = self.bond_radius
        pb_colors = {'metal coordination bonds':(147,112,219,255)}
        for name, pbg in self.pseudobond_groups.items():
            pb = pbg.pseudobonds
            pb.radii = self.pseudobond_radius
            pb.colors = pb_colors.get(name, (255,255,0,255))

    def _make_drawing(self, initialize_graphical_attributes):
        if initialize_graphical_attributes:
            self._initialize_graphical_attributes()

        # Create graphics
        a = self.atoms
        self._create_atom_spheres(a.coords, self._atom_display_radii(), a.colors, a.displays)
        self._update_bond_graphics(self.bonds)
        for name, pbg in self.pseudobond_groups.items():
            self._update_pseudobond_graphics(name, pbg)
        self._create_ribbon_graphics()

    def _create_atom_spheres(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            self._atoms_drawing = p = self.new_drawing('atoms')

        n = len(coords)
        self.triangles_per_sphere = 320 if n < 30000 else 80 if n < 120000 else 20

        # Set instanced sphere triangulation
        from .. import surface
        va, na, ta = surface.sphere_geometry(self.triangles_per_sphere)
        p.geometry = va, ta
        p.normals = na

        self._update_atom_graphics(coords, radii, colors, display)

    def new_atoms(self):
        # TODO: Handle instead with a C++ notification that atoms added or deleted
        self._atoms = None
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
        for pbg in self.pseudobond_groups.values():
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
        a = self.atoms
        self._update_atom_graphics(a.coords, self._atom_display_radii(), a.colors, a.visibles)
        self._update_bond_graphics(self.bonds)
        pbgs = self.pseudobond_groups
        for name, pbg in pbgs.items():
            self._update_pseudobond_graphics(name, pbg)
        self._update_ribbon_graphics()

    def _update_atom_graphics(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            return

        # Set instanced sphere center position and radius
        n = len(coords)
        from numpy import empty, float32, multiply
        xyzr = empty((n, 4), float32)
        xyzr[:, :3] = coords
        xyzr[:, 3] = radii

        from ..geometry import Places
        p.positions = Places(shift_and_scale=xyzr)
        p.display_positions = display

        # Set atom colors
        p.colors = colors

        # Set selected
        a = self.atoms
        p.selected_positions = a.selected if a.num_selected > 0 else None

    def _atom_display_radii(self, a=None):
        if a is None:
            a = self.atoms
        r = a.radii.copy()
        dm = a.draw_modes
        from .molobject import Atom
        r[dm == Atom.BALL_STYLE] *= self.ball_scale
        r[dm == Atom.STICK_STYLE] = self.bond_radius
        return r

    def _update_bond_graphics(self, bonds):
        p = self._bonds_drawing
        if p is None:
            if bonds.num_shown == 0:
                return
            self._bonds_drawing = p = self.new_drawing('bonds')
            from .. import surface
            # Use 3 z-sections so cylinder ends match in half-bond mode.
            va, na, ta = surface.cylinder_geometry(nz = 3, caps = False)
            p.geometry = va, ta
            p.normals = na

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
        from .ribbon import Ribbon, XSection
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
        for rlist in polymers:
            rp = p.new_drawing(rlist.strs[0])
            t2r = []
            # Always call get_polymer_spline to make sure hide bits are
            # properly set when ribbons are completely undisplayed
            atoms, coords, guides = rlist.get_polymer_spline()
            residues = atoms.residues
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
            if self.ribbon_divisions % 2 == 1:
                seg_blend = self.ribbon_divisions
                seg_cap = seg_blend + 1
            else:
                seg_cap = self.ribbon_divisions
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
                prev_band = s.back_band
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
                if prev_band:
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
                if prev_band:
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
                self._ribbon_tether.append((tp, coords[tethered], atoms.filter(tethered), atoms,
                                            m.ribbon_tether_shape, m.ribbon_tether_scale))

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
        ss_ids = rlist.ss_id
        helices = rlist.is_helix
        for start, end in self._ss_ranges(helices, ss_ids):
            ss_coords = coords[start:end]
            adjusts = ribbon_adjusts[start:end][:, newaxis]
            axes, vals, centroid, rel_coords = self._ss_axes(ss_coords)
            # Compute position of cylinder center corresponding to
            # helix control point atoms
            axis = axes[0]
            axis_pos = dot(rel_coords, axis)[:, newaxis]
            cyl_centers = centroid + axis * axis_pos
            if False:
                # Debugging code to display center of secondary structure
                self._ss_display(rlist.strs[0] + " helix " + str(start), cyl_centers)
            # Compute radius of cylinder
            spokes = ss_coords - cyl_centers
            cyl_radius = mean(norm(spokes, axis=1))
            #from math import sqrt
            #cyl_radius = sqrt(vals[1] * vals[1] + vals[2] * vals[2])
            # Compute smoothed position of helix control point atoms
            ideal = cyl_centers + normalize_vector_array(spokes) * cyl_radius
            offsets = adjusts * (ideal - ss_coords)
            new_coords = ss_coords + offsets
            # Compute guide atom position relative to control point atom
            delta_guides = guides[start:end] - ss_coords
            # Update both control point and guide coordinates
            coords[start:end] = new_coords
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
        # Smooth strands
        strands = logical_and(rlist.is_sheet, logical_not(helices))
        for start, end in self._ss_ranges(strands, ss_ids):
            ss_coords = coords[start:end]
            adjusts = ribbon_adjusts[start:end][:, newaxis]
            axes, vals, centroid, rel_coords = self._ss_axes(ss_coords)
            # Compute position for strand control point atom on
            # axis by projection
            axis = normalize(axes[0])
            axis_pos = dot(rel_coords, axis)[:, newaxis]
            ideal = centroid + axis * axis_pos
            if False:
                # Debugging code to display center of secondary structure
                self._ss_display(rlist.strs[0] + " helix " + str(start), ideal)
            offsets = adjusts * (ideal - ss_coords)
            new_coords = ss_coords + offsets
            # Compute guide atom position relative to control point atom
            delta_guides = guides[start:end] - ss_coords
            # Update both control point and guide coordinates
            coords[start:end] = new_coords
            guides[start:end] = new_coords + delta_guides
            # Update the tethered array
            tethered[start:end] = norm(offsets, axis=1) > self.bond_radius

    def _ss_ranges(self, ba, ss_ids):
        # Return ranges of True in boolean array "ba"
        ranges = []
        start = -1
        start_id = None
        for n, bn in enumerate(ba):
            if bn:
                if start < 0:
                    start = n
                elif ss_ids[n] != start_id:
                    if n - start > 2:
                        ranges.append((start, n))
                    start_id = ss_ids[n]
                    start = n
            else:
                if start >= 0:
                    if n - start > 2:
                        ranges.append((start, n))
                    start = -1
        if start >= 0 and len(ba) - start > 2:
            ranges.append((start, len(ba)))
        return ranges

    def _ss_axes(self, ss_coords):
        from numpy import mean
        from numpy.linalg import svd
        centroid = mean(ss_coords, axis=0)
        rel_coords = ss_coords - centroid
        ignore, vals, vecs = svd(rel_coords)
        axes = vecs.take(vals.argsort()[::-1], 1)
        return axes, vals, centroid, rel_coords


    def _ss_display(self, name, centers):
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
        if self.atoms.num_selected > 0:
            rsel = set([r for r in self.atoms.filter(self.atoms.selected).unique_residues
                        if r in self._ribbon_r2t])
        else:
            rsel = set()
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
        for tp, xyz1, atoms, all_atoms, shape, scale in self._ribbon_tether:
            all_atoms.update_ribbon_visibility()
            xyz2 = atoms.coords
            radii = self._atom_display_radii(atoms) * scale
            tp.positions = _tether_placements(xyz1, xyz2, radii, shape)
            tp.display_positions = atoms.visibles
            colors = atoms.colors
            colors[:,3] *= self.ribbon_tether_opacity
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
        # TODO check intercept of bounding box as optimization
        pa = self._atom_first_intercept(mxyz1, mxyz2)
        pb = self._bond_first_intercept(mxyz1, mxyz2)
        if pb and pa:
            a = pa.atom
            if a.draw_mode == a.STICK_STYLE and a in pb.bond.atoms:
                pb = None	# Pick atom if stick bond and its atom are picked.
        ppb = self._pseudobond_first_intercept(mxyz1, mxyz2)
        pr = self._ribbon_first_intercept(mxyz1, mxyz2)
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
        asel = self.atoms.selected
        m = self.atoms.mask(atoms)
        from numpy import logical_not
        asel[m] = logical_not(asel[m]) if toggle else selected
        self.atoms.selected = asel
        self._selection_changed()

    def select_residue(self, residue, toggle=False, selected=True):
        '''
        Select a specified :class:`.Residue`.
        If selected is false then it unselects the residue.
        Selecting a residue is equivalent to select all the residue atoms.
        '''
        if toggle:
            selected = residue not in self._ribbon_selected_residues
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
        return Model.any_part_selected(self)

    def clear_selection(self):
        self.atoms.selected = False
        self._selection_changed()

    def _selection_changed(self, promotion = False):
        if not promotion:
            self._selection_promotion_history = []

        # Update selection on molecular surfaces
        # TODO: Won't work for surfaces spanning multiple molecules
        from .molsurf import MolecularSurface
        for s in self.child_drawings():
            if isinstance(s, MolecularSurface):
                s.update_selection()

    def promote_selection(self):
        n = self.atoms.num_selected
        if n == 0 or n == len(self.atoms):
            return
        asel = self.atoms.selected
        self._selection_promotion_history.append(asel)

        atoms = self.atoms
        r = atoms.residues
        rids = r.unique_ids
        from numpy import unique, in1d
        sel_rids = unique(rids[asel])
        ares = in1d(rids, sel_rids)
        if ares.sum() > n:
            # Promote to entire residues
            psel = ares
        else:
            from numpy import array
            cids = array(r.chain_ids)
            sel_cids = unique(cids[asel])
            ac = in1d(cids, sel_cids)
            if ac.sum() > n:
                # Promote to entire chains
                psel = ac
            else:
                # Promote to entire molecule
                psel = True
        self.atoms.selected = psel
        self._selection_changed(promotion = True)

    def demote_selection(self):
        pt = self._selection_promotion_history
        if len(pt) > 0:
            asel = pt.pop()
            if len(asel) == len(self.atoms):
                self.atoms.selected = asel
                self._selection_changed(promotion = True)

    def clear_selection_promotion_history(self):
        self._selection_promotion_history = []

    def surfaces(self):
        '''List of :class:`.MolecularSurface` objects for this structure.'''
        from .molsurf import MolecularSurface
        surfs = [s for s in self.child_models() if isinstance(s, MolecularSurface)]
        return surfs

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
