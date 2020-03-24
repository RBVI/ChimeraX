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

timing = False
#timing = True
if timing:
    from time import time
    rsegtime = 0
    spathtime = 0

EPSILON = 1e-6

# Must match values in molc.cpp
FLIP_MINIMIZE = 0
FLIP_PREVENT = 1
FLIP_FORCE = 2

from .molobject import StructureData
TETHER_CYLINDER = StructureData.TETHER_CYLINDER

def _make_ribbon_graphics(structure, ribbons_drawing):
    '''Update ribbons drawing.'''
    
    ribbons_drawing.clear()

    if structure.ribbon_display_count == 0:
        return

    if timing:
        t0 = time()

    polymers = structure.polymers(missing_structure_treatment=structure.PMS_TRACE_CONNECTS)

    if timing:
        poltime = time()-t0

    rangestime = xstime = smootime = tubetime = spltime = geotime = drtime = tethertime = 0
    global rsegtime, spathtime
    rsegtime = spathtime = 0
    for rlist, ptype in polymers:
        # Always call get_polymer_spline to make sure hide bits are
        # properly set when ribbons are completely undisplayed
        any_display, atoms, coords, guides = rlist.get_polymer_spline()
        if not any_display:
            continue

        # Use residues instead of rlist below because rlist may contain
        # residues that do not participate in ribbon (e.g., because
        # it does not have a CA)
        residues = atoms.residues

        # Always update all atom visibility so that undisplaying ribbon
        # will bring back previously hidden backbone atoms
        residues.atoms.update_ribbon_visibility()

        if len(atoms) < 2:
            continue
        
        displays = residues.ribbon_displays
        if displays.sum() == 0:
            continue

        if timing:
            t0 = time()
            
        # Assign a residue class to each residue and compute the
        # ranges of secondary structures
        is_helix = residues.is_helix
        ssids = residues.secondary_structure_ids
        arc_helix = (structure.ribbon_mode_helix == structure.RIBBON_MODE_ARC)
        res_class, helix_ranges, sheet_ranges, non_tube_ranges = \
            _ribbon_ranges(is_helix, residues.is_strand, ssids, residues.polymer_types, arc_helix)

        if timing:
            rangestime += time()-t0
            t0 = time()

        # Assign front and back cross sections for each residue.
        xs_mgr = structure.ribbon_xs_mgr
        xs_front, xs_back, smooth_twist = \
            _ribbon_crosssections(res_class, xs_mgr, is_helix, arc_helix)

        if timing:
            xstime += time()-t0
            t0 = time()
            
        # Perform any smoothing (e.g., strand smoothing
        # to remove lasagna sheets, pipes and planks
        # display as cylinders and planes, etc.)
        _smooth_ribbon(residues, coords, guides, helix_ranges, sheet_ranges,
                       structure.ribbon_mode_helix, structure.ribbon_mode_strand)

        if timing:
            smootime += time()-t0
            t0 = time()

        # Create tube helices, one new RibbonDrawing for each helix.
        if arc_helix:
            ribbon_adjusts = residues.ribbon_adjusts
            for start, end in helix_ranges:
                _make_arc_helix_drawing(residues, coords, guides, ssids, xs_front, xs_back, xs_mgr,
                                        ribbon_adjusts, start, end, ribbons_drawing)

        if timing:
            tubetime += time()-t0
            t0 = time()

        # _ss_control_point_display(ribbons_drawing, coords, guides)

        # Create spline path
        orients = structure.ribbon_orients(residues)
        segment_divisions = structure._level_of_detail.ribbon_divisions
        flip_modes = _ribbon_flip_modes(structure, is_helix)
        ribbon = Ribbon(coords, guides, orients, flip_modes, smooth_twist, segment_divisions,
                        structure.spline_normals)
        # _debug_show_normal_spline(ribbons_drawing, coords, ribbon, num_divisions)

        if timing:
            spltime += time() - t0
            t0 = time()
            
        # Compute ribbon triangles
        colors = residues.ribbon_colors
        geometry = TriangleAccumulator()
        _ribbon_geometry(ribbon, non_tube_ranges, displays, colors, xs_front, xs_back,
                         geometry)
        if timing:
            geotime += time() - t0
            t0 = time()

        # Create ribbon drawing
        if not geometry.empty():
            rp = RibbonDrawing(structure.name + " " + str(residues[0]) + " ribbons", len(residues))
            ribbons_drawing.add_drawing(rp)

            triangle_ranges = [RibbonTriangleRange(ts, te, vs, ve, residues[i])
                               for i,ts,te,vs,ve in geometry.triangle_ranges]
            rp.triangle_ranges = triangle_ranges     # Save triangle ranges for picking
            ribbons_drawing.add_residue_triangle_ranges(triangle_ranges, rp)

            # Set drawing geometry
            va, na, vc, ta = geometry.vertex_normal_color_triangle_arrays()
            rp.set_geometry(va, na, ta)
            rp.vertex_colors = vc

        if timing:
            drtime += time() - t0
            t0 = time()

        # Cache position of backbone atoms on ribbon.
        # Get list of tethered atoms, and create tether cone drawing.
        min_tether_offset = structure.bond_radius
        tether_atoms, tether_positions, tethers_drawing = \
            _ribbon_tethers(ribbon, residues, ribbons_drawing,
                            min_tether_offset,
                            structure.ribbon_tether_scale,
                            structure.ribbon_tether_sides,
                            structure.ribbon_tether_shape)
        ribbons_drawing.add_tethers(tether_atoms, tether_positions, tethers_drawing)

        if timing:
            tethertime += time()-t0
        
        # Create spine if necessary
#        if structure.ribbon_show_spine:
#            spine_colors, spine_xyz1, spine_xyz2 = spine
#            _ss_spine_display(ribbons_drawing, spine_colors, spine_xyz1, spine_xyz2)

    if timing:
        print('ribbon times %d polymers, %d residues, polymers %.4g, ranges %.4g, xsect %.4g, smooth %.4g, tube %.4g, spline %.4g, triangles %.4g (segspline %.4g (path %.4g))), makedrawing %.4g, tethers %.4g'
              % (len(polymers), len(ribbons_drawing._residue_triangle_ranges),
                 poltime, rangestime, xstime, smootime, tubetime,
                 spltime, geotime, rsegtime, spathtime,  drtime, tethertime))

def _ribbon_flip_modes(structure, is_helix):
    nres = len(is_helix)
    if structure.ribbon_mode_helix == structure.RIBBON_MODE_DEFAULT:
        last = nres - 1
        flip_modes = [(FLIP_PREVENT if is_helix[i] and (i == last or is_helix[i + 1]) else FLIP_MINIMIZE)
                      for i in range(nres)]
        # strands generally flip normals at every residue but
        # beta bulges violate this rule so we cannot always flip
        # elif is_strand[i] and is_strand[i + 1]:
        #     flip_mode = FLIP_FORCE
    else:
        flip_modes = [FLIP_MINIMIZE] * nres
    return flip_modes

#
# Assign a residue class to each residue and compute the
# ranges of secondary structures.
# Returned helix and strand ranges (r0,r1) start with residue index r0 and end with index r1-1.
# Returned non_tube_ranges (r0,r1) start with residue r0 and end with residue r1.
#
def _ribbon_ranges(is_helix, is_strand, ssids, polymer_type, arc_helix):
    res_class = []
    helix_ranges = []
    sheet_ranges = []

    was_sheet = was_helix = was_nucleic = False
    last_ssid = None

    from .molobject import Residue
    
    nr = len(is_helix)
    for i in range(nr):
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
                        _end_strand(res_class, sheet_ranges, i)
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
                        _end_helix(res_class, helix_ranges, i)
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
            _end_strand(res_class, sheet_ranges, i)
        elif was_helix and not am_helix:
            _end_helix(res_class, helix_ranges, i)
        res_class.append(rc)
        was_sheet = am_sheet
        was_helix = am_helix
        last_ssid = ssids[i]
    if was_sheet:
        # 1hxx ends in a strand
        _end_strand(res_class, sheet_ranges, nr)
    elif was_helix:
        # 1hxx ends in a strand
        _end_helix(res_class, helix_ranges, nr)

    # Postprocess helix ranges if in arc mode to remove
    # 2-residue helices since we cannot compute an arc
    # from two points.
    if arc_helix:
        helix_ranges = [r for r in helix_ranges if r[1] - r[0] > 2]
        non_tube_ranges = []
        s = 0
        for r0,r1 in helix_ranges:
            if r0 > 0:
                non_tube_ranges.append((s,r0))
            s = r1-1
        if s < nr-1:
            non_tube_ranges.append((s,nr-1))
    else:
        non_tube_ranges = [(0,nr-1)]
        
    return res_class, helix_ranges, sheet_ranges, non_tube_ranges

def _end_strand(res_class, ss_ranges, end):
    if res_class[-1] == XSectionManager.RC_SHEET_START:
        # Single-residue strands are coils
        res_class[-1] = XSectionManager.RC_COIL
        del ss_ranges[-1]
    else:
        # Multi-residue strands are okay
        res_class[-1] = XSectionManager.RC_SHEET_END
        ss_ranges[-1][1] = end

def _end_helix(res_class, ss_ranges, end):
    if res_class[-1] == XSectionManager.RC_HELIX_START:
        # Single-residue helices are coils
        res_class[-1] = XSectionManager.RC_COIL
        del ss_ranges[-1]
    else:
        # Multi-residue helices are okay
        res_class[-1] = XSectionManager.RC_HELIX_END
        ss_ranges[-1][1] = end
        
def _ribbon_crosssections(res_class, ribbon_xs_mgr, is_helix, arc_helix):
    # Assign front and back cross sections for each residue.
    # The "front" section is between this residue and the previous.
    # The "back" section is between this residue and the next.
    # The front and back sections meet at the control point atom.
    # Compute cross sections and whether we care about a smooth
    # transition between residues.
    # If helices are displayed as tubes, we alter the cross sections
    # at the beginning and end to coils since the wide ribbon looks
    # odd when it is only for half a residue
    xs_front = []
    xs_back = []
    smooth_twist = []
    rc0 = XSectionManager.RC_COIL
    nr = len(res_class)
    for i in range(nr):
        rc1 = res_class[i]
        try:
            rc2 = res_class[i + 1]
        except IndexError:
            rc2 = XSectionManager.RC_COIL
        f, b = ribbon_xs_mgr.assign(rc0, rc1, rc2)
        xs_front.append(f)
        xs_back.append(b)
        smooth_twist.append(_smooth_twist(rc1, rc2))
        rc0 = rc1
    if arc_helix:
        for i in range(nr):
            if not is_helix[i]:
                continue
            rc = res_class[i]
            if rc == XSectionManager.RC_HELIX_START:
                xs_front[i] = ribbon_xs_mgr.xs_coil
            elif rc == XSectionManager.RC_HELIX_END:
                xs_back[i] = ribbon_xs_mgr.xs_coil
    smooth_twist[-1] = False
    return xs_front, xs_back, smooth_twist

def _smooth_twist(rc0, rc1):
    # Determine if we need to twist ribbon smoothly from crosssection rc0 to rc1.
    # Twist smoothly if cross-section does not change and also within helices and
    # sheets including cross-section changes for arrows.
    if rc0 == rc1:
        return True
    if rc0 in XSectionManager.RC_ANY_SHEET and rc1 in XSectionManager.RC_ANY_SHEET:
        return True
    if rc0 in XSectionManager.RC_ANY_HELIX and rc1 in XSectionManager.RC_ANY_HELIX:
        return True
    if rc0 is XSectionManager.RC_HELIX_END or rc0 is XSectionManager.RC_SHEET_END:
        return True
    return False
        
# Compute triangle geometry for ribbon.
# Only certain ranges of residues are considered, since tube helix
# geometry is created by other code.
# TODO: This routine is taking half the ribbon compute time.  Probably a
#  big contributor is that 17 numpy arrays are being made per residue.
#  Might want to put TriangleAccumulator into C++ to get rid of half of those
#  and have extrude() and blend() put results directly into it.
#  Maybe Ribbon spline coords, tangents, normals could use recycled numpy arrays.
def _ribbon_geometry(ribbon, ranges, displays, colors, xs_front, xs_back, geometry):

    nr = len(displays)
    
    # Each residue has left and right half (also called front and back)
    # with the residue centered in the middle.
    # The two halfs can have different crosssections, e.g. turn and helix.
    # At the ends of the polymer the spline is extended to make the first residue
    # have a left half and the last residue have a right half.
    # If an interior range is shown only half segments are shown at the ends
    # since other code (e.g. tube cylinders) will render the other halfs.
    for r0,r1 in ranges:

        capped = True
        prev_band = None
        
        for i in range(r0, r1+1):
            if not displays[i]:
                continue

            # Left half
            if i > r0 or i == 0:
                mid_cap = (xs_front[i] != xs_back[i])
                front_c, front_t, front_n = ribbon.segment(i - 1, Ribbon.SECOND_HALF, mid_cap)
                sf = xs_front[i].extrude(front_c, front_t, front_n, colors[i],
                                         capped, mid_cap, geometry.v_offset)
                geometry.add_extrusion(sf)

                if prev_band is not None:
                    tjoin = xs_front[i].blend(prev_band, sf.front_band)
                    geometry.add_triangles(tjoin)

            # Right half
            if i < r1 or i == nr-1:
                if i == nr-1:
                    next_cap = True
                else:
                    next_cap = (displays[i] != displays[i + 1] or xs_back[i] != xs_front[i + 1])
                back_c, back_t, back_n = ribbon.segment(i, Ribbon.FIRST_HALF, next_cap)
                sb = xs_back[i].extrude(back_c, back_t, back_n, colors[i],
                                        mid_cap, next_cap, geometry.v_offset)
                geometry.add_extrusion(sb)

                if not mid_cap:
                    tjoin = xs_back[i].blend(sf.back_band, sb.front_band)
                    geometry.add_triangles(tjoin)

                prev_band = None if next_cap else sb.back_band
                capped = next_cap

            geometry.add_range(i)

class TriangleAccumulator:
    '''Accumulate triangles from segments of a ribbon.'''
    def __init__(self):
        self._v_start = 0         # for tracking starting vertex index for each residue
        self._v_end = 0
        self._t_start = 0         # for tracking starting triangle index for each residue
        self._t_end = 0
        self._vertex_list = []
        self._normal_list = []
        self._color_list = []
        self._triangle_list = []
        self._triangle_ranges = []	# List of 5-tuples (residue_index, tstart, tend, vstart, vend)

    def empty(self):
        return len(self._triangle_list) == 0
    
    @property
    def v_offset(self):
        return self._v_end
    
    def add_extrusion(self, extrusion):
        e = extrusion
        self._v_end += len(e.vertices)
        self._t_end += len(e.triangles)
        self._vertex_list.append(e.vertices)
        self._normal_list.append(e.normals)
        self._triangle_list.append(e.triangles)
        self._color_list.append(e.colors)

    def add_triangles(self, triangles):
        self._triangle_list.append(triangles)
        self._t_end += len(triangles)

    def add_range(self, residue_index):
        ts,te,vs,ve = self._t_start, self._t_end, self._v_start, self._v_end
        if te == ts and ve == vs:
            return
        self._triangle_ranges.append((residue_index, ts, te, vs, ve))
        self._t_start = te
        self._v_start = ve

    def vertex_normal_color_triangle_arrays(self):
        if self._vertex_list:
            from numpy import concatenate
            va = concatenate(self._vertex_list)
            na = concatenate(self._normal_list)
            from chimerax.geometry import normalize_vectors
            normalize_vectors(na)
            ta = concatenate(self._triangle_list)
            vc = concatenate(self._color_list)
        else:
            va = na = ta = vc = None
        return va, na, vc, ta

    @property
    def triangle_ranges(self):
        return self._triangle_ranges

# -----------------------------------------------------------------------------
#
class RibbonTriangleRange:
    __slots__ = ["start", "end", "vstart", "vend", "drawing", "residue"]
    def __init__(self, start, end, vstart, vend, residue):
        self.start = start
        self.end = end
        self.vstart = vstart
        self.vend = vend
        self.residue = residue
        self.drawing = None
    def __lt__(self, other): return self.start < other
    def __le__(self, other): return self.start <= other
    def __eq__(self, other): return self.start == other
    def __ne__(self, other): return self.start != other
    def __gt__(self, other): return self.start > other
    def __gt__(self, other): return self.start >= other

from chimerax.graphics import Drawing
class RibbonsDrawing(Drawing):
    def __init__(self, name, structure_name):
        Drawing.__init__(self, name)
        self.structure_name = structure_name
        self._residue_triangle_ranges = {}       # map residue to list of RibbonTriangleRanges
        self._atom_tether_base = {}             # Map atom to tether position on ribbon
        self._tether_atoms = []                 # List of collections of all backbone atoms
        self._tether_drawings = []		# List of TethersDrawing

    def clear(self):
        self.remove_all_drawings()
        self._residue_triangle_ranges.clear()
        self._tether_atoms.clear()
        self._atom_tether_base.clear()
        self._tether_drawings.clear()

    def compute_ribbons(self, structure):
        if timing:
            t0 = time()
        _make_ribbon_graphics(structure, self)
        if timing:
            t1 = time()
            print ('compute_ribbons(): %.4g' % (t1-t0))
    
    def update_ribbon_colors(self, structure):
        if timing:
            t0 = time()

        vertex_colors = {}	# Map RibbonDrawing to vertex color array
        for r, tranges in self._residue_triangle_ranges.items():
            color = r.ribbon_color
            for tr in tranges:
                d = tr.drawing
                if d in vertex_colors:
                    vc = vertex_colors[d]
                else:
                    vertex_colors[d] = vc = d.vertex_colors
                tri = d.triangles
                vc[tr.vstart:tr.vend,:] = color

        for d, vc in vertex_colors.items():
            d.vertex_colors = vc

        if timing:
            t1 = time()
            print ('update_ribbon_colors(): %.4g' % (t1-t0))
        
    def update_ribbon_highlight(self, structure):

        # Find residues to highlight by RibbonDrawing
        res = structure.residues
        rsel = res[res.selected]
        res_triangle_ranges = self._residue_triangle_ranges
        dres = {}	# Map ribbon drawing to list of selected residues
        for r in rsel:
            tranges = res_triangle_ranges.get(r)
            if tranges:
                for tr in tranges:
                    d = tr.drawing
                    if d in dres:
                        dres[d].append(r)
                    else:
                        dres[d] = [r]

        # Set highlights for all RibbonDrawing
        for d in self.child_drawings():
            if isinstance(d, RibbonDrawing):
                if d in dres:
                    tmask = d.highlighted_triangles_mask
                    if tmask is None:
                        from numpy import zeros, bool
                        tmask = zeros((d.number_of_triangles(),), bool)
                    else:
                        tmask[:] = False
                    rlist = dres[d]
                    if len(rlist) == d.num_residues:
                        if tmask.all():
                            continue	# Already all highlighted
                        tmask[:] = True
                    else:
                        for r in rlist:
                            for trange in res_triangle_ranges[r]:
                                if trange.drawing is d:
                                    tmask[trange.start:trange.end] = True
                    d.highlighted_triangles_mask = tmask
                elif d.highlighted_triangles_mask is not None:
                    d.highlighted_triangles_mask = None

    def add_residue_triangle_ranges(self, tranges, ribbon_drawing):
        rtr = self._residue_triangle_ranges
        for tr in tranges:
            r = tr.residue
            if r in rtr:
                # Tube helices result in two triangle ranges for one residue
                # since a helix end residue is depicted as half cylinder, half strand.
                rtr[r].append(tr)
            else:
                rtr[r] = [tr]
            tr.drawing = ribbon_drawing

    def ribbon_spline_position(self, atom):
        return self._atom_tether_base[atom]
    
    def add_tethers(self, tether_atoms, tether_positions, tether_drawing):
        self._tether_atoms.append(tether_atoms)
        self._atom_tether_base.update(tether_positions)
        if tether_drawing is not None:
            self._tether_drawings.append(tether_drawing)
        
    def update_tethers(self, structure):
        if timing:
            t0 = time()

        for ta in self._tether_atoms:
            ta.update_ribbon_visibility()

        for td in self._tether_drawings:
            td.update_tethers(structure)

        if timing:
            t1 = time()
            print ('update_tethers(): %.4g' % (t1-t0))

class TethersDrawing(Drawing):
    def __init__(self, name, tethered_atoms, spline_coords,
                 tether_shape, tether_scale, tether_sides):

        self._tethered_atoms = tethered_atoms
        self._spline_coords = spline_coords     # Tether attachment points on ribbon
        self._tether_shape = tether_shape
        self._tether_scale = tether_scale
        
        Drawing.__init__(self, name)
        self.skip_bounds = True   # Don't include in bounds calculation. Optimization.
        self.no_cofr = True	# Don't use for finding center of rotation. Optimization.
        self.pickable = False	# Don't allow mouse picking.
        from chimerax import surface
        if tether_shape == TETHER_CYLINDER:
            va, na, ta = surface.cylinder_geometry(nc=tether_sides, nz=2, caps=False)
        else:
            # Assume it's either TETHER_CONE or TETHER_REVERSE_CONE
            va, na, ta = surface.cone_geometry(nc=tether_sides, caps=False, points_up=False)
        self.set_geometry(va, na, ta)

    def update_tethers(self, structure):
        tatoms = self._tethered_atoms
        xyz1 = self._spline_coords
        xyz2 = tatoms.coords
        radii = structure._atom_display_radii(tatoms) * self._tether_scale
        self.positions = _tether_placements(xyz1, xyz2, radii, self._tether_shape)
        self.display_positions = tatoms.visibles & tatoms.residues.ribbon_displays
        colors = tatoms.colors
        from numpy import around
        colors[:,3] = around(colors[:,3] * structure.ribbon_tether_opacity).astype(int)
        self.colors = colors

# -----------------------------------------------------------------------------
#
def _tether_placements(xyz0, xyz1, radius, shape):
    from .molobject import StructureData
    from .structure import _bond_cylinder_placements
    c0,c1 = (xyz1,xyz0) if shape == StructureData.TETHER_REVERSE_CONE else (xyz0,xyz1)
    return _bond_cylinder_placements(c0, c1, radius)
                        
class RibbonDrawing(Drawing):
    def __init__(self, name, num_residues):
        Drawing.__init__(self, name)
        self.num_residues = num_residues
        self.triangle_ranges = []         # List of RibbonTriangleRange, used for picking

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)):
            return None
        p = super().first_intercept(mxyz1, mxyz2)
        if p is None:
            return None
        tranges = self.triangle_ranges
        from bisect import bisect_right
        n = bisect_right(tranges, p.triangle_number)
        if n > 0:
            triangle_range = tranges[n - 1]
            from .structure import PickedResidue
            return PickedResidue(triangle_range.residue, p.distance)
        return None

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []
        tranges = self.triangle_ranges
        picks = []
        rp = super().planes_pick(planes)
        from chimerax.graphics import PickedTriangles
        from .structure import PickedResidues
        for p in rp:
            if isinstance(p, PickedTriangles) and p.drawing() is self:
                tmask = p._triangles_mask
                res = [rtr.residue for rtr in tranges if tmask[rtr.start:rtr.end].sum() > 0]
                if res:
                    from .molarray import Residues
                    rc = Residues(res)
                    picks.append(PickedResidues(rc))
        return picks

def _ribbon_update_spine(c, centers, normals, spine):
    from numpy import empty
    xyz1 = centers + normals
    xyz2 = centers - normals
    color = empty((len(xyz1), 4), int)
    color[:] = c
    if len(spine) == 0:
        spine.extend((color, xyz1, xyz2))
    else:
        # TODO: Fix O(N^2) accumulation
        from numpy import concatenate
        spine[0] = concatenate([spine[0], color])
        spine[1] = concatenate([spine[1], xyz1])
        spine[2] = concatenate([spine[2], xyz2])

def _ribbon_tethers(ribbon, residues, drawing,
                    min_tether_offset, tether_scale, tether_sides, tether_shape):
    # Cache position of backbone atoms on ribbon
    # and get list of tethered atoms
    spositions = {}
    positions = _ribbon_spline_position(ribbon, residues, _NonTetherPositions)
    spositions.update(positions)
    positions = _ribbon_spline_position(ribbon, residues, _TetherPositions)
    from .molarray import Atoms
    tether_atoms = Atoms(tuple(positions.keys()))
    from numpy import array
    spline_coords = array(tuple(positions.values()))
    spositions.update(positions)
    if len(spline_coords) == 0:
        spline_coords = spline_coords.reshape((0,3))
    atom_coords = tether_atoms.coords
    offsets = atom_coords - spline_coords
    from numpy.linalg import norm
    tethered = norm(offsets, axis=1) > min_tether_offset

    # Create tethers if necessary
    from numpy import any
    if tether_scale > 0 and any(tethered):
        name = drawing.structure_name + " ribbon tethers"
        tdrawing = TethersDrawing(name, tether_atoms.filter(tethered), spline_coords[tethered],
                                  tether_shape, tether_scale, tether_sides)
        drawing.add_drawing(tdrawing)
    else:
        tdrawing = None

    return tether_atoms, spositions, tdrawing

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

def _ribbon_spline_position(ribbon, residues, pos_map):
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

def _debug_show_normal_spline(ribbons_drawing, coords, ribbon):
    # Normal spline can be shown as spheres on either side (S)
    # or a cylinder across (C)
    num_coords = len(coords)
    try:
        spline = ribbon.normal_spline
        other_spline = ribbon.other_normal_spline
    except AttributeError:
        return
    sp = ribbons_drawing.new_drawing(ribbons_drawing.structure_name + " normal spline")
    from chimerax import surface
    from numpy import empty, array, float32, linspace
    from chimerax.geometry import Places
    num_pts = num_coords*ribbon_divisions
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
    sp.positions = _tether_placements(xyz1, xyz2, radii, TETHER_CYLINDER)
    sp_colors = empty((len(xyz1), 4), dtype=float32)
    sp_colors[:] = (255, 0, 0, 255)
    sp.colors = sp_colors
    #C

def _smooth_ribbon(rlist, coords, guides, helix_ranges, sheet_ranges, helix_mode, strand_mode):
    ribbon_adjusts = rlist.ribbon_adjusts
    from .molobject import StructureData
    if helix_mode == StructureData.RIBBON_MODE_DEFAULT:
        # Smooth helices
        # XXX: Skip helix smoothing for now since it does not work well for bent helices
        pass
        # for start, end in helix_ranges:
        #     _smooth_helix(coords, guides, ribbon_adjusts, start, end)
    elif helix_mode == StructureData.RIBBON_MODE_ARC:
        # No spline smoothing for tube helices
        pass
    elif helix_mode == StructureData.RIBBON_MODE_WRAP:
        for start, end in helix_ranges:
            _wrap_helix(rlist, coords, guides, start, end)
    if strand_mode == StructureData.RIBBON_MODE_DEFAULT:
        # Smooth strands
        for start, end in sheet_ranges:
            _smooth_strand(rlist, coords, guides, ribbon_adjusts, start, end)
    elif strand_mode == StructureData.RIBBON_MODE_ARC:
        for start, end in sheet_ranges:
            _arc_strand(rlist, coords, guides, start, end)

def _smooth_helix(rlist, coords, guides, ribbon_adjusts, start, end):
    # Try to fix up the ribbon orientation so that it is parallel to the helical axis
    from numpy import dot, newaxis, mean
    from numpy.linalg import norm
    # We only "optimize" longer helices because short
    # ones do not contain enough information to do
    # things intelligently
    ss_coords = coords[start:end]
    adjusts = ribbon_adjusts[start:end][:, newaxis]
    axis, centroid, rel_coords = _ss_axes(ss_coords)
    # Compute position of cylinder center corresponding to
    # helix control point atoms
    axis_pos = dot(rel_coords, axis)[:, newaxis]
    cyl_centers = centroid + axis * axis_pos
    if False:
        # Debugging code to display center of secondary structure
        name = ribbons_drawing.structure_name + " helix " + str(start)
        _ss_display(ribbons_drawing, name, cyl_centers)
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

def _make_arc_helix_drawing(rlist, coords, guides, ssids, xs_front, xs_back, xs_mgr,
                            ribbon_adjusts, start, end, ribbons_drawing):
    '''Creates a new RibbonDrawing for one tube helix.'''
    # Only bother if at least one residue is displayed
    displays = rlist.ribbon_displays
    from numpy import any
    if not any(displays[start:end]):
        return

    va, na, ca, ta, t_range = _arc_helix_geometry(rlist, coords, guides, ssids,
                                                  xs_front, xs_back, xs_mgr,
                                                  ribbon_adjusts, start, end, displays)
    
    # Create triangle range selection data structures
    tranges = []
    for i, r in t_range.items():
        res = rlist[i]
        t_start, t_end, v_start, v_end = r
        triangle_range = RibbonTriangleRange(t_start, t_end, v_start, v_end, res)
        tranges.append(triangle_range)

    # Fourth, create graphics drawing with vertices, normals, colors and triangles
    name = "helix-%d" % ssids[start]
    ssp = RibbonDrawing(name, len(rlist))
    ribbons_drawing.add_drawing(ssp)
    ssp.set_geometry(va, na, ta)
    ssp.vertex_colors = ca

    ssp.triangle_ranges = tranges
    ribbons_drawing.add_residue_triangle_ranges(tranges, ssp)

def _arc_helix_geometry(rlist, coords, guides, ssids, xs_front, xs_back, xs_mgr,
                        ribbon_adjusts, start, end, displays):
    '''Compute triangulation for one tube helix.'''

    from .sse import HelixCylinder
    from numpy import linspace, cos, sin
    from math import pi
    from numpy import empty, tile

    hc = HelixCylinder(coords[start:end], radius=xs_mgr.tube_radius)
    centers = hc.cylinder_centers()
    radius = hc.cylinder_radius()
    normals, binormals = hc.cylinder_normals()
    icenters, inormals, ibinormals = hc.cylinder_intermediates()
    coords[start:end] = centers

    # Compute unit circle in 2D
    num_pts = xs_mgr.params[xs_mgr.STYLE_ROUND]["sides"]
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
        t_range[start] = [0, ti, 0, vi]
    was_displayed = displays[start]
    for i in range(start+1, end-1):
        if displays[i]:
            t_start = ti
            v_start = vi
            if not was_displayed:
                add_front_cap(i)
            add_both_bands(i)
            t_range[i] = [t_start, ti, v_start, vi]
        else:
            if was_displayed:
                add_back_cap(i-1)
                t_range[i-1][1] = ti
                t_range[i-1][3] = vi
        was_displayed = displays[i]
    # last residue
    if displays[end-1]:
        t_start = ti
        v_start = vi
        if was_displayed:
            add_front_band(end-1)
            add_back_cap(end-1)
        else:
            add_front_cap(end-1)
            add_front_band(end-1)
            add_back_cap(end-1)
        t_range[end-1] = [t_start, ti, v_start, vi]
    elif was_displayed:
        add_back_cap(end-2)
        t_range[end-2][1] = ti
        t_range[end-2][3] = vi

    return va, na, ca, ta, t_range
        
def _wrap_helix(rlist, coords, guides, start, end):
    # Only bother if at least one residue is displayed
    displays = rlist.ribbon_displays
    from numpy import any
    if not any(displays[start:end]):
        return

    from .sse import HelixCylinder
    hc = HelixCylinder(coords[start:end])
    directions = hc.cylinder_directions()
    coords[start:end] = hc.cylinder_surface()
    guides[start:end] = coords[start:end] + directions
    if False:
        # Debugging code to display guides of secondary structure
        name = ribbons_drawing.structure_name + " helix guide " + str(start)
        _ss_guide_display(ribbons_drawing, name, coords[start:end], guides[start:end])


def _smooth_strand(rlist, coords, guides, ribbon_adjusts, start, end):
    if (end - start + 1) <= 2:
        # Short strands do not need smoothing
        return
    from numpy import zeros, empty, dot, newaxis
    from numpy.linalg import norm
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
        sname = ribbons_drawing.structure_name
        _ss_display(ribbons_drawing, sname + " strand " + str(start), ideal)
        _ss_guide_display(ribbons_drawing, sname + " strand guide " + str(start),
                          coords[start:end], guides[start:end])

def _arc_strand(rlist, coords, guides, start, end):
    if (end - start + 1) <= 2:
        # Short strands do not need to be shown as planks
        return
    # Only bother if at least one residue is displayed
    displays = rlist.ribbon_displays
    from numpy import any
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
        name = ribbons_drawing.structure_name + " strand guide " + str(start)
        _ss_guide_display(ribbons_drawing, name, sp.tilt_centers, g)
    coords[start:end] = centers
    #delta = guides[start:end] - coords[start:end]
    #guides[start:end] = coords[start:end] + delta
    guides[start:end] = coords[start:end] + binormals
    if True:
        # Debugging code to display center of secondary structure
        name = ribbons_drawing.structure_name + " strand " + str(start)
        _ss_display(ribbons_drawing, name, centers)

def _ss_axes(ss_coords):
    from numpy import mean, argmax
    from numpy.linalg import svd
    centroid = mean(ss_coords, axis=0)
    rel_coords = ss_coords - centroid
    ignore, vals, vecs = svd(rel_coords)
    axes = vecs[argmax(vals)]
    return axes, centroid, rel_coords

def _ss_display(ribbons_drawing, name, centers):
    ssp = ribbons_drawing.new_drawing(name)
    from chimerax import surface
    va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
    ssp.set_geometry(va, na, ta)
    from numpy import empty, float32
    ss_radii = empty(len(centers) - 1, float32)
    ss_radii.fill(0.2)
    ssp.positions = _tether_placements(centers[:-1], centers[1:], ss_radii, TETHER_CYLINDER)
    ss_colors = empty((len(ss_radii), 4), float32)
    ss_colors[:] = (0,255,0,255)
    ssp.colors = ss_colors

def _ss_guide_display(ribbons_drawing, name, centers, guides):
    ssp = ribbons_drawing.new_drawing(name)
    from chimerax import surface
    va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
    ssp.set_geometry(va, na, ta)
    from numpy import empty, float32
    ss_radii = empty(len(centers), float32)
    ss_radii.fill(0.2)
    ssp.positions = _tether_placements(centers, guides, ss_radii, TETHER_CYLINDER)
    ss_colors = empty((len(ss_radii), 4), float32)
    ss_colors[:] = (255,255,0,255)
    ssp.colors = ss_colors

def _ss_control_point_display(ribbons_drawing, coords, guides):
    # Debugging code to display line from control point to guide
    cp = ribbons_drawing.new_drawing(ribbons_drawing.structure_name + " control points")
    from chimerax import surface
    va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=False)
    cp.set_geometry(va, na, ta)
    from numpy import empty, float32
    cp_radii = empty(len(coords), float)
    cp_radii.fill(0.1)
    cp.positions = _tether_placements(coords, guides, cp_radii, TETHER_CYLINDER)
    cp_colors = empty((len(coords), 4), float)
    cp_colors[:] = (255,0,0,255)
    cp.colors = cp_colors

def _ss_spine_display(ribbons_drawing, spine_colors, spine_xyz1, spine_xyz2):
    sp = ribbons_drawing.new_drawing(ribbons_drawing.structure_name + " spine")
    from chimerax import surface
    va, na, ta = surface.cylinder_geometry(nc=3, nz=2, caps=True)
    sp.set_geometry(va, na, ta)
    from numpy import empty, float32
    spine_radii = empty(len(spine_colors), float32)
    spine_radii.fill(0.3)
    sp.positions = _tether_placements(spine_xyz1, spine_xyz2, spine_radii, TETHER_CYLINDER)
    sp.colors = spine_colors


class Ribbon:

    FIRST_HALF = 1
    SECOND_HALF = 2

    def __init__(self, coords, guides, orients, flip_modes, smooth_twist, segment_divisions,
                 use_spline_normals):
        # Extend the coordinates at start and end to make sure the
        # ribbon is straight on either end.  Compute the spline
        # coefficients for each axis.  Then throw away the
        # coefficients for the fake ends.

        self._smooth_twist = smooth_twist
        self._segment_divisions = segment_divisions
        self._use_spline_normals = use_spline_normals
        
        from numpy import empty, zeros, ones
        c = empty((len(coords) + 2, 3), float)
        c[0] = coords[0] + (coords[0] - coords[1])
        c[1:-1] = coords
        c[-1] = coords[-1] + (coords[-1] - coords[-2])
        coeff = [self._compute_coefficients(c, i) for i in range(3)]
        from numpy import transpose, float64
        self._coeff = transpose(coeff, axes = (1,0,2)).astype(float64)
        self.flipped = zeros(len(coords), bool)
        # Currently Structure::ribbon_orient() defines the orientation method as
        # ATOMS for helices, PEPTIDE for strands, and GUIDES for nucleic acids.
        atom_normals = None
        from .structure import Structure
        atom_mask = (orients == Structure.RIBBON_ORIENT_ATOMS)
        curvature_normals = None
        curvature_mask = (orients == Structure.RIBBON_ORIENT_CURVATURE)
        guide_normals = None
        guide_mask = ((orients == Structure.RIBBON_ORIENT_GUIDES) |
                       (orients == Structure.RIBBON_ORIENT_PEPTIDE))
        if atom_mask.any():
            atom_normals = self._compute_normals_from_control_points(coords)
        if curvature_mask.any():
            curvature_normals = self._compute_normals_from_curvature(coords)
        if guide_mask.any():
            if guides is None or len(coords) != len(guides):
                if atom_normals is None:
                    guide_normals = self._compute_normals_from_control_points(coords)
                else:
                    guide_normals = atom_normals
                guide_flip = False
            else:
                guide_normals = self._compute_normals_from_guides(coords, guides)
                guide_flip = True
        self.normals = None
        if atom_normals is not None:
            self.normals = atom_normals
        if curvature_normals is not None:
            if self.normals is None:
                self.normals = curvature_normals
            else:
                self.normals[curvature_mask] = curvature_normals[curvature_mask]
        if guide_normals is not None:
            if self.normals is None:
                self.normals = guide_normals
            else:
                self.normals[guide_mask] = guide_normals[guide_mask]
        # Currently Structure._use_spline_normals = False.
        if use_spline_normals:
            self._flip_normals(coords)
            num_coords = len(coords)
            from numpy import linspace, array, empty, float32, double
            x = linspace(0.0, num_coords, num=num_coords, endpoint=False, dtype=double)
            y = array(coords + self.normals, dtype=double)
            y2 = array(coords - self.normals, dtype=double)
            # Interpolation can be done using interpolating (1),
            # least-squares (2) or plain cubic splines (3):
            #
            #1
            from scipy.interpolate import make_interp_spline
            self.normal_spline = make_interp_spline(x, y)
            self.other_normal_spline = make_interp_spline(x, y2)
            #1
            #2
            #2 from scipy.interpolate import make_lsq_spline
            #2 t = empty(len(x) + 4, double)
            #2 t[:4] = x[0]
            #2 t[4:-4] = x[2:-2]
            #2 t[-4:] = x[-1]
            #2 self.normal_spline = make_lsq_spline(x, y, t)
            #2 self.other_normal_spline = make_lsq_spline(x, y2, t)
            #2
            #3
            #3 from scipy.interpolate import CubicSpline
            #3 self.normal_spline = CubicSpline(x, y)
            #3 self.other_normal_spline = CubicSpline(x, y2)
            #3
        else:
            # Only use flip_modes where guides are missing.
            # Currently flip modes other than FLIP_MINIMIZE is
            # used only for nucleic acids missing guide atoms.
            if guide_normals is not None and not guide_flip:
                fmodes = [(fmode if g else FLIP_MINIMIZE)
                          for fmode,g in zip(flip_modes, guide_mask)]
            else:
                fmodes = [FLIP_MINIMIZE] * len(flip_modes)
            self._flip_modes = fmodes

        # Initialize segment cache
        self._seg_cache = {}

    def _compute_normals_from_guides(self, coords, guides):
        from numpy import zeros, array
        from sys import __stderr__ as stderr
        t = self.get_tangents()
        n = guides - coords
        normals = zeros((len(coords), 3), float)
        for i in range(len(coords)):
            normals[i,:] = get_orthogonal_component(n[i], t[i])
        return normalize_vector_array(normals)

    def _compute_normals_from_curvature(self, coords):
        from numpy import empty, array, cross, inner
        from numpy.linalg import norm
        tangents = self.get_tangents()
        raw_normals = empty(coords.shape, float)
        c = self._coeff
        xcv,ycv,zcv = c[:,0,:], c[:,1,:], c[:,2,:] 
        # Curvature for the N-1 points are at the start of a segment
        # but the last one is at the end of the last segment and has
        # to be treated specially
        for seg in range(len(raw_normals) - 1):
            curvature = array((xcv[seg,2], ycv[seg,2], zcv[seg,2]))
            raw_normals[seg] = cross(curvature, tangents[seg])
        xc = xcv[-1]
        yc = ycv[-1]
        zc = zcv[-1]
        curvature = array((2 * xc[2] + 6 * xc[3],
                           2 * yc[2] + 6 * yc[3],
                           2 * zc[2] + 6 * zc[3]))
        raw_normals[-1] = cross(curvature, tangents[-1])
        #
        # The normals are NOT normalized.  We check for any near-zero
        # normals and replace them with vectors computed from well-defined
        # normals of adjacent control points
        normals = empty(raw_normals.shape, float)
        lengths = norm(raw_normals, axis=1)
        last_valid = None
        last_normal = None
        for i in range(len(lengths)):
            if lengths[i] > EPSILON:
                n = raw_normals[i] / lengths[i]
                normals[i] = n
                if last_valid is None:
                    # At start of spline, so we propagate this normal
                    # backwards (if needed)
                    for j in range(i):
                        normals[j] = n
                else:
                    # There was a previous normal defined so we interpolate
                    # for intermediates (if needed)
                    dv = n - last_normal
                    ncp = i - last_valid
                    for j in range(1, ncp):
                        f = j / ncp
                        rn = dv * f + last_normal
                        d = norm(rn)
                        int_n = rn / d
                        normals[last_valid+j] = int_n
                last_valid = i
                last_normal = n
        # Check whether there are control points at the end that have
        # no normals.  It could actually be all control points if the
        # spline is really a straight line.
        if last_valid is None:
            # No valid control point normals assigned.
            # Just pick a random one.
            for i in range(3):
                v = [0.0, 0.0, 0.0]
                v[i] = 1.0
                rn = cross(array(v), tangents[0])
                d = norm(rn)
                if d > EPSILON:
                    n = rn / d
                    normals[:] = n
                    return normals
            # Really, we only need range(2) for i since the spline line can only be
            # collinear with one of the axis, but we put 3 for esthetics.
            # If we try all three and fail, something is seriously wrong.
            raise RuntimeError("spline normal computation for straight line")
        else:
            for j in range(last_valid + 1, len(lengths)):
                normals[j] = last_normal
        return normals

    def _compute_normals_from_control_points(self, coords):
        # This version ignores the guide atom positions and computes normals
        # at each control point by making it perpendicular to the vectors pointing
        # to control points on either side.  The two ends and any collinear control
        # points are handled afterwards.  We also assume that ribbon cross sections
        # are symmetric in both x- and y-axis so that a twist by more than 90 degrees
        # is equvalent to an opposite twist of (180 - original_twist) degrees.
        from numpy import cross, empty, array, dot
        from numpy.linalg import norm
        #
        # Compute normals by cross-product of vectors to prev and next control points.
        # Normals are for range [1:-1] since 0 and -1 are missing prev and next
        # control points.  Normal index is therefore off by one.
        tangents = self.get_tangents()
        dv = coords[:-1] - coords[1:]
        raw_normals = cross(dv[:-1], -dv[1:])
        for i in range(len(raw_normals)):
            raw_normals[i] = get_orthogonal_component(raw_normals[i], tangents[i+1])
        #
        # Assign normal for first control point.  If there are collinear control points
        # at the beginning, assign same normal for all of them.  If there is no usable
        # normal across the entire "spline" (all control points collinear), just pick
        # a random vector normal to the line.
        normals = empty(coords.shape, float)
        lengths = norm(raw_normals, axis=1)
        for i in range(len(raw_normals)):
            if lengths[i] > EPSILON:
                # Normal for this control point is propagated to all previous ones
                prev_normal = raw_normals[i] / lengths[i]
                prev_index = i
                # Use i+2 because we need to assign one normal for the first control point
                # which has no corresponding raw_normal and (i + 1) for all the control
                # points up to and including this one.
                normals[:i+2] = prev_normal
                break
        else:
            # All control points collinear
            for i in range(3):
                v = [0.0, 0.0, 0.0]
                v[i] = 1.0
                rn = cross(array(v), tangents[0])
                d = norm(rn)
                if d > EPSILON:
                    n = rn / d
                    normals[:] = n
                    return normals
            # Really, we only need range(2) for i since the spline line can only be
            # collinear with one of the axis, but we put 3 for esthetics.
            # If we try all three and fail, something is seriously wrong.
            raise RuntimeError("spline normal computation for straight line")
        #
        # Now we have at least one normal assigned.  This is the anchor and we
        # look for the next control point that has a non-zero raw normal.
        # If we do not find one, then this is the last anchor and we just assign
        # the normal to the remainder of the control points.  Otherwise, we
        # have two normals (perpendicular to the same straight line) for 2 or
        # more control points.  The first normal is from the previous control
        # point whose normal we already set, and the second normal is for the
        # last control point in our range of 2 or more.  If there are more than
        # 2 control points, we interpolate the two normals to get the intermediate
        # normals.
        while i < len(raw_normals):
            if lengths[i] > EPSILON:
                # Our control points run from prev_index to i
                n = raw_normals[i] / lengths[i]
                # First we check whether we should flip it due to too much twist
                if dot(n, prev_normal) < 0:
                    n = -n
                # Now we compute normals for intermediate control points (if any)
                # Instead of slerp, we just use linear interpolation for simplicity
                ncp = i - prev_index
                dv = n - prev_normal
                for j in range(1, ncp):
                    f = j / ncp
                    rn = dv * f + prev_normal
                    d = norm(rn)
                    int_n = rn / d
                    normals[prev_index+j] = int_n
                # Finally, we assign normal for this control point
                normals[i+1] = n
                prev_normal = n
                prev_index = i
            i += 1
        # This is the last part of the spline, so assign the remainder of
        # the normals.
        normals[prev_index+1:] = prev_normal
        return normals

    def _compute_coefficients(self, coords, n):
        # Matrix from http://mathworld.wolfram.com/CubicSpline.html
        # Set b[0] and b[-1] to 1 to match TomG code in VolumePath
        import numpy
        size = len(coords)
        a = numpy.ones((size,), float)
        b = numpy.ones((size,), float) * 4
        b[0] = b[-1] = 2
        #b[0] = b[-1] = 1
        c = numpy.ones((size,), float)
        d = numpy.zeros((size,), float)
        d[0] = coords[1][n] - coords[0][n]
        d[1:-1] = 3 * (coords[2:,n] - coords[:-2,n])
        d[-1] = 3 * (coords[-1][n] - coords[-2][n])
        D = tridiagonal(a, b, c, d)
        from numpy import array
        c_a = coords[:-1,n]
        c_b = D[:-1]
        delta = coords[1:,n] - coords[:-1,n]
        c_c = 3 * delta - 2 * D[:-1] - D[1:]
        c_d = 2 * -delta + D[:-1] + D[1:]
        tcoeffs = array([c_a, c_b, c_c, c_d]).transpose()
        coef = tcoeffs[1:-1]
        return coef

    def _segment_coefficients(self, seg):
        return self._coeff[seg]

    def _flip_normals(self, coords):
        from numpy import cross, sqrt, dot
        from numpy.linalg import norm
        num_coords = len(coords)
        tangents = self.get_tangents()
        axes = cross(tangents[:-1], tangents[1:])
        s = norm(axes, axis=1)
        c = sqrt(1 - s * s)
        for i in range(1, num_coords):
            ne = self._rotate_around(axes[i-1], c[i-1], s[i-1], self.normals[i-1])
            n = self.normals[i]
            # Allow for a little extra twist before flipping
            if dot(ne, n) < 0:
            # if dot(ne, n) < 0.2:
                self.normals[i] = -n

    def _rotate_around(self, n, c, s, v):
        c1 = 1 - c
        m00 = c + n[0] * n[0] * c1
        m01 = n[0] * n[1] * c1 - s * n[2]
        m02 = n[2] * n[0] * c1 + s * n[1]
        m10 = n[0] * n[1] * c1 + s * n[2]
        m11 = c + n[1] * n[1] * c1
        m12 = n[2] * n[1] * c1 - s * n[0]
        m20 = n[0] * n[2] * c1 - s * n[1]
        m21 = n[1] * n[2] * c1 + s * n[0]
        m22 = c + n[2] * n[2] * c1
        # Use temporary so that v[0] does not get set too soon
        x = m00 * v[0] + m01 * v[1] + m02 * v[2]
        y = m10 * v[0] + m11 * v[1] + m12 * v[2]
        z = m20 * v[0] + m21 * v[1] + m22 * v[2]
        #v[0] = x
        #v[1] = y
        #v[2] = z
        import numpy
        return numpy.array((x, y, z))

    @property
    def num_segments(self):
        return len(self._coeff)

    def get_tangents(self):
        c = self._coeff
        xcv,ycv,zcv = c[:,0,:], c[:,1,:], c[:,2,:] 
        nc = len(xcv)
        from numpy import array
        t = [[xcv[n,1], ycv[n,1], zcv[n,1]] for n in range(nc)]
        xc = xcv[-1]
        yc = ycv[-1]
        zc = zcv[-1]
        t.append((xc[1] + 2 * xc[2] + 3 * xc[3],
                   yc[1] + 2 * yc[2] + 3 * yc[3],
                   zc[1] + 2 * zc[2] + 3 * zc[3]))
        return normalize_vector_array(array(t))

    def segment(self, seg, side, include_end):
        if timing:
            t0 = time()

        if seg == -1 and side == Ribbon.SECOND_HALF:
            return self._lead_segment()
        elif seg == self.num_segments and side == Ribbon.FIRST_HALF:
            return self._trail_segment()

        divisions = self._segment_divisions
        if seg in self._seg_cache:
            coords, tangents, normals = self._seg_cache[seg]
        else:
            coeffs = self._segment_coefficients(seg)
            coords, tangents = _spline_segment_path(coeffs, 0, 1, divisions+1)
            if self._use_spline_normals:
                from numpy import array, linspace, sum
                # We _should_ return normals that are orthogonal (O) to the
                # tangents, but it does not look as good as if we use
                # the interpolated non-orthogonal (NO) normals.
                #
                #O
                #O xyz = array([get_orthogonal_component(self.normal_spline(t) - coords[i],
                #O                                       tangents[i])
                #O              for i, t in enumerate(linspace(seg, seg+1.0, num=divisions+1,
                #O                                             endpoint=True))])
                #O normals = normalize_vector_array(xyz)
                #O
                #NO
                xyz = array([self.normal_spline(t)
                             for t in linspace(seg, seg+1.0, num=divisions+1, endpoint=True)])
                normals = normalize_vector_array(xyz - coords)
                #NO
            else:
                from ._ribbons import parallel_transport
                normals = parallel_transport(tangents, self.normals[seg])
                if self._smooth_twist[seg]:
                    end_normal = self.normals[seg + 1]
                    flip = _flip_end_normal(normals[-1], end_normal, tangents[-1],
                                            self._flip_modes[seg], self.flipped[seg], self.flipped[seg + 1])
                    if flip:
                        self.normals[seg + 1] = end_normal = -self.normals[seg + 1]
                        self.flipped[seg + 1] = not self.flipped[seg + 1]
                    from ._ribbons import smooth_twist as twist_normals
                    twist_normals(tangents, normals, end_normal)

            #normals = curvature_to_normals(curvature, tangents, prev_normal)
            self._seg_cache[seg] = (coords, tangents, normals)

        # divisions = number of segments = number of vertices + 1
        middle = divisions // 2
        if side is self.FIRST_HALF:
            start = 0
            end = middle + 1 if include_end else middle
        else:
            start = middle
            end = divisions + 2 if include_end else divisions + 1
            
        seg_coords, seg_tangents, seg_normals = coords[start:end], tangents[start:end], normals[start:end]

        if timing:
            global rsegtime
            rsegtime += time()-t0
            
        return seg_coords, seg_tangents, seg_normals

    def _lead_segment(self):
        coeffs = self._segment_coefficients(0)
        # We do not want to go from -0.5 to 0 because the
        # first residue will already have the "0" coordinates
        # as part of its ribbon.  We want to connect to that
        # coordinate smoothly.
        divisions = self._segment_divisions // 2
        n = divisions + 1
        step = 0.5 / n
        coords, tangents = _spline_segment_path(coeffs, -0.3, -step, n)
        n_start = self.normals[0]
        from ._ribbons import parallel_transport
        normals = parallel_transport(tangents, n_start)
        #normals = curvature_to_normals(curvature, tangents, None)
        return coords, tangents, normals

    def _trail_segment(self):
        coeffs = self._segment_coefficients(-1)
        # We do not want to go from 1 to 1.5 because the
        # last residue will already have the "1" coordinates
        # as part of its ribbon.  We want to connect to that
        # coordinate smoothly.
        divisions = self._segment_divisions // 2
        n = divisions + 1
        step = 0.5 / n
        coords, tangents = _spline_segment_path(coeffs, 1 + step, 1.3, n)
        n_end = self.normals[-1]
        from ._ribbons import parallel_transport
        normals = parallel_transport(tangents, n_end)
        #normals = curvature_to_normals(curvature, tangents, prev_normal)
        return coords, tangents, normals

    def position(self, seg, t):
        # Compute coordinates for segment seg with parameter t
        from numpy import array, dot
        coeffs = self._segment_coefficients(seg)
        st = array([1.0, t, t*t, t*t*t])
        return array([dot(st, coeffs[0]), dot(st, coeffs[1]), dot(st, coeffs[2])])

# Decide whether to flip the spline segment end normal so that it aligns better with
# the parallel transported normal.
def _flip_end_normal(transported_normal, end_normal, tangent,
                     flip_mode, start_flipped, end_flipped):

    if flip_mode == FLIP_MINIMIZE:
        # If twist is greater than 90 degrees, turn the opposite
        # direction.  (Assumes that ribbons are symmetric.)
        from ._ribbons import dihedral_angle
        a = dihedral_angle(transported_normal, end_normal, tangent)
        from math import pi
        # flip = (abs(a) > 0.5 * pi)
        flip = (abs(a) > 0.6 * pi)	# Not sure why this is not pi / 2.
    elif flip_mode == FLIP_PREVENT:
        # Make end_flip the same as start_flip
        flip = (end_flipped != start_flipped)
    elif flip_mode == FLIP_FORCE:
        # Make end_flip the opposite of start_flip
        flip = (end_flipped == start_flipped)
    else:
        flip = False

    return flip

from chimerax.core.state import State

class XSectionManager(State):
    """XSectionManager keeps track of ribbon cross sections used in an AtomicStructure instance.

    Constants:
      Residue classes:
        RC_NUCLEIC - all nucleotides
        RC_COIL - coil
        RC_SHEET_START - first residue in sheet
        RC_SHEET_MIDDLE - middle residue in sheet
        RC_SHEET_END - last residue in sheet
        RC_HELIX_START - first residue in helix
        RC_HELIX_MIDDLE - middle residue in helix
        RC_HELIX_END - last residue in helix
      Cross section styles:
        STYLE_SQUARE - four-sided cross section with right angle corners and faceted lighting
        STYLE_ROUND - round cross section with no corners
        STYLE_PIPING - flat cross section with piping on either end
      Ribbon cross sections:
        RIBBON_NUCLEIC - Use style for nucleotides
        RIBBON_COIL - Use style for coil
        RIBBON_SHEET - Use style for sheet
        RIBBON_SHEET_ARROW - Use style for sheet scaled into an arrow
        RIBBON_HELIX - Use style for helix
        RIBBON_HELIX_ARROW - Use style for helix scaled into an arrow
    """

    # Class constants
    (RC_NUCLEIC, RC_COIL,
     RC_SHEET_START, RC_SHEET_MIDDLE, RC_SHEET_END,
     RC_HELIX_START, RC_HELIX_MIDDLE, RC_HELIX_END) = range(8)
    RC_ANY_SHEET = set([RC_SHEET_START, RC_SHEET_MIDDLE, RC_SHEET_END])
    RC_ANY_HELIX = set([RC_HELIX_START, RC_HELIX_MIDDLE, RC_HELIX_END])
    (STYLE_SQUARE, STYLE_ROUND, STYLE_PIPING) = range(3)
    (RIBBON_NUCLEIC, RIBBON_SHEET, RIBBON_SHEET_ARROW,
     RIBBON_HELIX, RIBBON_HELIX_ARROW, RIBBON_COIL) = range(6)

    def __init__(self):
        self.structure = None
        self.scale_helix = (1.0, 0.2)
        self.scale_helix_arrow = ((2.0, 0.2), (0.2, 0.2))
        self.scale_sheet = (1.0, 0.2)
        self.scale_sheet_arrow = ((2.0, 0.2), (0.2, 0.2))
        self.scale_coil = (0.2, 0.2)
        self.scale_nucleic = (0.2, 1.0)
        self.style_helix = self.STYLE_ROUND
        self.style_sheet = self.STYLE_SQUARE
        self.style_coil = self.STYLE_ROUND
        self.style_nucleic = self.STYLE_SQUARE
        self.arrow_helix = False
        self.arrow_sheet = True
        self.params = {
            self.STYLE_ROUND: {
                "sides": 12,
                "faceted": False,
            },
            self.STYLE_SQUARE: {
                # No parameters yet for square style
            },
            self.STYLE_PIPING: {
                "sides": 18,
                "ratio": 0.5,
                "faceted": False,
            },
        }
        self.transitions = {
            # SHEET_START in the middle
            (self.RC_COIL, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_COIL, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_HELIX_END, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_HELIX_END, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_SHEET_END, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_SHEET_END, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            # SHEET_END in the middle
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_COIL):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_COIL):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_HELIX_START):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_HELIX_START):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_SHEET_START):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_SHEET_START):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            # HELIX_START in the middle
            (self.RC_COIL, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_COIL, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_HELIX_END, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_HELIX_END, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_SHEET_END, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_SHEET_END, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            # HELIX_END in the middle
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_COIL):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_COIL):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_HELIX_START):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_HELIX_START):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_SHEET_START):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_SHEET_START):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
        }
        self.tube_radius = None

        self._xs_helix = None
        self._xs_helix_arrow = None
        self._xs_sheet = None
        self._xs_sheet_arrow = None
        self._xs_coil = None
        self._xs_nucleic = None

    def set_structure(self, structure):
        import weakref
	# 0.21 is slightly bigger than the default stick radius
	# so ends of stick will be completely hidden by ribbon
	# instead of sticking out partially on the other side
        self.structure = weakref.ref(structure)

    def assign(self, rc0, rc1, rc2):
        """Return front and back cross sections for the middle residue.

        rc0, rc1 and rc2 are residue classes of three consecutive residues.
        The return value is a 2-tuple of XSection instances.
        The first is for the half-segment between rc0 and rc1, ending at rc1.
        The second is for the half-segment between rc1 and rc2, starting at rc1."""
        if rc1 is self.RC_NUCLEIC:
            return self.xs_nucleic, self.xs_nucleic
        if rc1 is self.RC_SHEET_MIDDLE:
            return self.xs_sheet, self.xs_sheet
        if rc1 is self.RC_HELIX_MIDDLE:
            return self.xs_helix, self.xs_helix
        if rc1 is self.RC_COIL:
            return self.xs_coil, self.xs_coil
        try:
            r_front, r_back = self.transitions[(rc0, rc1, rc2)]
            return self._xs_ribbon(r_front), self._xs_ribbon(r_back)
        except KeyError:
            print("unsupported transition %d-%d-%d" % (rc0, rc1, rc2))
            return self.xs_coil, self.xs_coil

    def is_compatible(self, xs0, xs1):
        """Return if the two cross sections can be blended."""
        return xs0 is xs1

    def set_helix_scale(self, x, y):
        """Set scale factors for helix ribbon cross section."""
        v = (x, y)
        if self.scale_helix != v:
            self.scale_helix = v
            self._xs_helix = None
            self._set_gc_ribbon()

    def set_helix_arrow_scale(self, x1, y1, x2, y2):
        """Set scale factors for helix arrow ribbon cross section."""
        v = ((x1, y1), (x2, y2))
        if self.scale_helix_arrow != v:
            self.scale_helix_arrow = v
            self._xs_helix_arrow = None
            self._set_gc_ribbon()

    def set_sheet_scale(self, x, y):
        """Set scale factors for sheet ribbon cross section."""
        v = (x, y)
        if self.scale_sheet != v:
            self.scale_sheet = v
            self._xs_sheet = None
            self._set_gc_ribbon()

    def set_sheet_arrow_scale(self, x1, y1, x2, y2):
        """Set scale factors for sheet arrow ribbon cross section."""
        v = ((x1, y1), (x2, y2))
        if self.scale_sheet_arrow != v:
            self.scale_sheet_arrow = v
            self._xs_sheet_arrow = None
            self._set_gc_ribbon()

    def set_coil_scale(self, x, y):
        """Set scale factors for coil ribbon cross section."""
        v = (x, y)
        if self.scale_coil != v:
            self.scale_coil = v
            self._xs_coil = None
            self._set_gc_ribbon()

    def set_nucleic_scale(self, x, y):
        """Set scale factors for nucleic ribbon cross section."""
        v = (x, y)
        if self.scale_nucleic != v:
            self.scale_nucleic = v
            self._xs_nucleic = None
            self._set_gc_ribbon()

    def set_helix_style(self, s):
        """Set style for helix ribbon cross section."""
        if self.style_helix != s:
            self.style_helix = s
            self._xs_helix = None
            self._xs_helix_arrow = None
            self._set_gc_ribbon()

    def set_sheet_style(self, s):
        """Set style for sheet ribbon cross section."""
        if self.style_sheet != s:
            self.style_sheet = s
            self._xs_sheet = None
            self._xs_sheet_arrow = None
            self._set_gc_ribbon()

    def set_coil_style(self, s):
        """Set style for coil ribbon cross section."""
        if self.style_coil != s:
            self.style_coil = s
            self._xs_coil = None
            self._set_gc_ribbon()

    def set_nucleic_style(self, s):
        """Set style for helix ribbon cross section."""
        if self.style_nucleic != s:
            self.style_nucleic = s
            self._xs_nucleic = None
            self._set_gc_ribbon()

    def set_transition(self, rc0, rc1, rc2, rf, rb):
        """Set transition for ribbon cross section across residue classes.

        The "assign" method converts residue classes for three consecutive
        residues into two cross sections.  This method defines the values
        that are used by "assign" when the residue classes are different.
        If rc1 is RC_NUCLEIC, RC_COIL, RC_SHEET_MIDDLE, RC_HELIX_MIDDLE,
        the two returned cross sections are the same and match the residue
        class.  In all other cases, a lookup dictionary is used with the
        3-tuple key of (rc0, rc1, rc2).  For this method, the values to
        be inserted into the lookup dictionary are the RIBBON_* constants."""
        key = (rc0, rc1, rc2)
        if key not in self.transitions:
            raise ValueError("transition %d-%d-%d is never used" % key)
        v = (rf, rb)
        if self.transitions[key] != v:
            self.transitions[key] = v
            self._set_gc_ribbon()

    def set_helix_end_arrow(self, b):
        if self.arrow_helix != b:
            self.arrow_helix = b
            self._set_gc_ribbon()

    def set_sheet_end_arrow(self, b):
        if self.arrow_sheet != b:
            self.arrow_sheet = b
            self._set_gc_ribbon()

    def set_tube_radius(self, r):
        if self.tube_radius != r:
            self.tube_radius = r
            self._set_gc_ribbon()

    def set_params(self, style, **kw):
        param = self.params[style]
        for k in kw.keys():
            if k not in param:
                raise ValueError("unknown parameter %s" % k)
        any_changed = False
        for k, v in kw.items():
            if param[k] != v:
                param[k] = v
                any_changed = True
                self._set_gc_ribbon()
        if any_changed:
            if self.style_helix == style:
                self._xs_helix = None
                self._xs_helix_arrow = None
            if self.style_sheet == style:
                self._xs_sheet = None
                self._xs_sheet_arrow = None
            if self.style_coil == style:
                self._xs_coil = None
            if self.style_nucleic == style:
                self._xs_nucleic = None

    @property
    def xs_helix(self):
        if self._xs_helix is None:
            self._xs_helix = self._make_xs(self.style_helix, self.scale_helix)
        return self._xs_helix

    @property
    def xs_helix_arrow(self):
        if self._xs_helix_arrow is None:
            if self.style_helix == self.STYLE_PIPING:
                style = self.STYLE_ROUND
            else:
                style = self.style_helix
            base = self._make_xs(style, (1.0, 1.0))
            self._xs_helix_arrow = base.arrow(self.scale_helix_arrow)
        return self._xs_helix_arrow

    @property
    def xs_sheet(self):
        if self._xs_sheet is None:
            self._xs_sheet = self._make_xs(self.style_sheet, self.scale_sheet)
        return self._xs_sheet

    @property
    def xs_sheet_arrow(self):
        if self._xs_sheet_arrow is None:
            if self.style_sheet == self.STYLE_PIPING:
                style = self.STYLE_ROUND
            else:
                style = self.style_sheet
            base = self._make_xs(style, (1.0, 1.0))
            self._xs_sheet_arrow = base.arrow(self.scale_sheet_arrow)
        return self._xs_sheet_arrow

    @property
    def xs_coil(self):
        if self._xs_coil is None:
            self._xs_coil = self._make_xs(self.style_coil, self.scale_coil)
        return self._xs_coil

    @property
    def xs_nucleic(self):
        if self._xs_nucleic is None:
            self._xs_nucleic = self._make_xs(self.style_nucleic, self.scale_nucleic)
        return self._xs_nucleic

    def _make_xs(self, style, scale):
        if style is self.STYLE_ROUND:
            return self._make_xs_round(scale)
        elif style is self.STYLE_SQUARE:
            return self._make_xs_square(scale)
        elif style is self.STYLE_PIPING:
            return self._make_xs_piping(scale)
        else:
            raise ValueError("unknown style %s" % style)

    def _make_xs_round(self, scale):
        from numpy import array
        from .molobject import RibbonXSection as XSection
        coords = []
        normals = []
        param = self.params[self.STYLE_ROUND]
        sides = param["sides"]
        from numpy import linspace, cos, sin, stack
        from math import pi
        angles = linspace(0, 2 * pi, sides, endpoint=False)
        ca = cos(angles)
        sa = sin(angles)
        circle = stack((ca, sa), axis=1)
        coords = circle * array(scale)
        if param["faceted"]:
            return XSection(coords, faceted=True)
        else:
            normals = circle * array((scale[1], scale[0]))
            return XSection(coords, normals=normals, faceted=False)

    def _make_xs_square(self, scale):
        from numpy import array
        from .molobject import RibbonXSection as XSection
        coords = array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))) * array(scale)
        return XSection(coords, faceted=True)

    def _make_xs_piping(self, scale):
        from numpy import array
        from math import pi, cos, sin, asin
        from .molobject import RibbonXSection as XSection
        if scale[0] > scale[1]:
            flipped = False
            delta = scale[0] - scale[1]
            radius = scale[1]
        else:
            flipped = True
            delta = scale[1] - scale[0]
            radius = scale[0]
        coords = []
        normals = []
        param = self.params[self.STYLE_PIPING]
        # The total number of sides is param["sides"]
        # We subtract the two connectors and then divide
        # by two while rounding up
        sides = (param["sides"] - 1) // 2
        ratio = param["ratio"]
        theta = asin(ratio)
        side_angle = 2.0 * (pi - theta) / sides
        # Generate the vertices for the two piping on either side.
        # The first and last points are used to connect the two.
        for start_angle, offset in ((theta - pi, delta), (theta, -delta)):
            for i in range(sides + 1):
                angle = start_angle + i * side_angle
                x = cos(angle) * radius
                y = sin(angle) * radius
                if flipped:
                    normals.append((-y, x))
                    coords.append((-y, x + offset))
                else:
                    normals.append((x, y))
                    coords.append((x + offset, y))
        coords = array(coords)
        tess = ([(0, sides, sides+1),(0, sides+1, 2*sides+1)] +         # connecting rectangle
                [(0, i, i+1) for i in range(1, sides)] +                # first piping
                [(sides+1, i, i+1) for i in range(sides+2, 2*sides+1)]) # second piping
        if param["faceted"]:
            return XSection(coords, faceted=True, tess=tess)
        else:
            if flipped:
                normals[0] = normals[-1] = (1, 0)
                normals[sides] = normals[sides + 1] = (-1, 0)
            else:
                normals[0] = normals[-1] = (0, -1)
                normals[sides] = normals[sides + 1] = (0, 1)
            normals = array(normals)
            return XSection(coords, normals=normals, faceted=False, tess=tess)

    def _xs_ribbon(self, r):
        if r is self.RIBBON_HELIX:
            return self.xs_helix
        elif r is self.RIBBON_SHEET:
            return self.xs_sheet
        elif r is self.RIBBON_COIL:
            return self.xs_coil
        elif r is self.RIBBON_NUCLEIC:
            return self.xs_nucleic
        elif r is self.RIBBON_HELIX_ARROW:
            if self.arrow_helix:
                return self.xs_helix_arrow
            else:
                return self.xs_helix
        elif r is self.RIBBON_SHEET_ARROW:
            if self.arrow_sheet:
                return self.xs_sheet_arrow
            else:
                return self.xs_sheet
        else:
            raise ValueError("unknown ribbon ref %d" % r)

    def _set_gc_ribbon(self):
        # Mark ribbon for rebuild
        s = self.structure()
        if s is not None:
            s._graphics_changed |= s._RIBBON_CHANGE

    # Session methods

    _SessionAttrs = [
        "scale_helix",
        "scale_helix_arrow",
        "scale_sheet",
        "scale_sheet_arrow",
        "scale_coil",
        "scale_nucleic",
        "style_helix",
        "style_sheet",
        "style_coil",
        "style_nucleic",
        "arrow_helix",
        "arrow_sheet",
        "params",
        "transitions",
        "tube_radius",
    ]

    def take_snapshot(self, session, flags):
        data = dict([(attr, getattr(self, attr))
                     for attr in self._SessionAttrs])
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        xs_mgr = cls()
        xs_mgr.set_state_from_snapshot(session, data)
        return xs_mgr

    def set_state_from_snapshot(self, session, data):
        for attr in self._SessionAttrs:
            try:
                setattr(self, attr, data[attr])
            except KeyError:
                # Older sessions may not have all the current parameters
                pass


def normalize(v):
    # normalize a single vector
    from numpy import isnan
    from numpy.linalg import norm
    d = norm(v)
    if isnan(d) or d < EPSILON:
        return v
    return v / d


def normalize_vector_array(a):
    from numpy.linalg import norm
    import numpy
    d = norm(a, axis=1)
    d[numpy.isnan(d)] = 1
    d[d < EPSILON] = 1
    n = a / d[:, numpy.newaxis]
    return n


def tridiagonal(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    Hacked source from
    http://ofan666.blogspot.com/2012/02/tridiagonal-matrix-algorithm-solver-in.html
    '''
    nf = len(a)     # number of equations
    for i in range(1, nf):
        mc = a[i] / b[i - 1]
        b[i] = b[i] - mc * c[i - 1]
        d[i] = d[i] - mc * d[i - 1]
    xc = a
    xc[-1] = d[-1] / b[-1]
    for i in range(nf - 2, -1, -1):
        xc[i] = (d[i] - c[i] * xc[i + 1]) / b[i]
    return xc


def get_orthogonal_component(v, ref):
    from numpy import inner
    from numpy.linalg import norm
    d = inner(v, ref)
    ref_len = norm(ref)
    return v + ref * (-d / ref_len)

from chimerax.geometry import cubic_path as _spline_segment_path

def _spline_segment_path_unused(coeffs, tmin, tmax, num_points):
    if timing:
        t0 = time()
    # coeffs is a 3x4 array of float.
    # Compute coordinates by multiplying spline parameter vector
    # (1, t, t**2, t**3) by the spline coefficients, and
    # compute tangents by multiplying spline parameter vector
    # (0, 1, 2*t, 3*t**2) by the same spline coefficients
    from numpy import array, ones, linspace, dot
    spline = array(coeffs).transpose()
    t = linspace(tmin, tmax, num_points)
    t2 = t * t
    st = ones((num_points, 4), float)    # spline multiplier matrix
    # st[:,0] is 1.0        # 1
    st[:,1] = t             # t
    st[:,2] = t2            # t^2
    st[:,3] = t * t2        # t^3
    coords = dot(st, spline)
    # st[:,0] stays at 1.0  # 1
    st[:,1] *= 2.0          # 2t
    st[:,2] *= 3.0          # 3t^2
    tangents = dot(st[:,:-1], spline[1:])
    tangents = normalize_vector_array(tangents)
    #st[:,0] = 2.0           # 2
    #st[:,1] *= 3.0;         # 6t
    #curvature = dot(st[:,:-2], spline[2:])
    #return coords, tangents, curvature

    if timing:
        global spathtime
        spathtime += time()-t0
    return coords, tangents

def curvature_to_normals(curvature, tangents, prev_normal):
    from numpy.linalg import norm
    from numpy import empty, cross, dot
    normals = empty(curvature.shape, dtype=float)
    for i in range(len(curvature)):
        c_len = norm(curvature[i])
        if c_len < EPSILON:
            # No curvature, must use previous normal
            # TODO: more here
            pass
        else:
            # Normal case
            n = cross(curvature[i], tangents[i])
            if prev_normal is not None and dot(prev_normal, n) < 0:
                n = -n
            prev_normal = normals[i] = n
    #normals = normalize_vector_array(cross(curvature, tangents))
    normals = normalize_vector_array(normals)
    return normals



# Code for debugging moving code from Python to C++
#
# DebugCVersion = False
#
def _debug_compare(label, test, ref, verbose=False):
    from sys import __stderr__ as stderr
    if isinstance(ref, list):
        from numpy import array
        ref = array(ref)
    from numpy import allclose
    try:
        if not allclose(test, ref, atol=1e-4):
            raise ValueError("not same")
    except ValueError:
        print(label, "--- not same!", test.shape, ref.shape, file=stderr)
        print(test, file=stderr)
        print(ref, file=stderr)
        print(test - ref, file=stderr)
    else:
        if verbose:
            print(label, "-- okay", file=stderr)
