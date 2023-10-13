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

timing = False
#timing = True
if timing:
    from time import time
    coeftime = normaltime = 0

EPSILON = 1e-6

from .molobject import StructureData
TETHER_CYLINDER = StructureData.TETHER_CYLINDER

from numpy import array, zeros, ones, empty, float32, float64, uint8
from numpy import dot, concatenate, any, linspace, newaxis, inner, cross, mean
from numpy.linalg import norm

from . import _ribbons

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

    rangestime = xstime = smootime = tubetime = pathtime = spltime = interptime = geotime = tethertime = 0
    global coeftime, normaltime
    coeftime = normaltime = 0

    # Ribbon quality. Number of band per residue.
    lod = structure._level_of_detail
    segment_divisions = lod.ribbon_fixed_divisions
    if segment_divisions is None:
        nres = structure.num_ribbon_residues
        segment_divisions = lod.ribbon_divisions(nres)

    # Accumulate ribbon information for all polymer chains.
    geometry = TriangleAccumulator()
    polyres = []
    roffset = 0
    tethered_atoms = []
    backbone_atoms = []

    for rlist, ptype in polymers:
        # Always call get_polymer_spline to make sure hide bits are
        # properly unset when ribbons are completely undisplayed.
        # Guides are O amino acid positions or C1' nucleic acid positions
        # but will be None if any of those atoms are missing.
        any_display, atoms, coords, guides = _get_polymer_spline(rlist)
        if not any_display:
            continue

        # Use residues instead of rlist below because rlist may contain
        # residues that do not participate in ribbon (e.g., because
        # it does not have a CA)
        residues = atoms.residues

        # Always update all atom visibility so that undisplaying ribbon
        # will bring back previously hidden backbone atoms
        residues.atoms.update_ribbon_backbone_atom_visibility()

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
        res_class, helix_ranges, sheet_ranges, display_ranges = \
            _ribbon_ranges(is_helix, residues.is_strand, ssids, displays,
                           residues.polymer_types, arc_helix)

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

        # Create tube helices.
        if arc_helix:
            ribbon_adjusts = residues.ribbon_adjusts
            for start, end in helix_ranges:
                if displays[start:end].any():
                    centers = _arc_helix_geometry(coords, xs_mgr, displays, start, end, geometry)
                    # Adjust coords so non-tube half of helix ends joins center of cylinder
                    coords[start:end] = centers

        if timing:
            tubetime += time()-t0
            t0 = time()

        # _ss_control_point_display(ribbons_drawing, coords, guides)

        # Create spline path
        if timing:
            t1 = time()
        orients = structure.ribbon_orients(residues)
        flip_normals = _ribbon_flip_normals(structure, is_helix)
        ribbon = Ribbon(coords, guides, orients, flip_normals, smooth_twist, segment_divisions,
                        structure.spline_normals)
        if timing:
            spltime += time()-t1

        if timing:
            t1 = time()
        path = ribbon.path()
        # _debug_show_normal_spline(ribbons_drawing, coords, ribbon, num_divisions)
        if timing:
            interptime += time()-t1

        if timing:
            pathtime += time() - t0
            t0 = time()
            
        # Compute ribbon triangles
        _ribbon_geometry(path, display_ranges, len(residues), xs_front, xs_back, geometry)

        if timing:
            geotime += time() - t0
            t0 = time()

        # Get list of tethered atoms and attachment position to ribbon.
        if structure.ribbon_tether_scale > 0:
            min_tether_offset = structure.bond_radius
            t_atoms, b_atoms = _ribbon_tethers(ribbon, residues, min_tether_offset)
            if t_atoms:
                tethered_atoms.append(t_atoms)
            if b_atoms:
                backbone_atoms.append(b_atoms)
                
        if timing:
            tethertime += time()-t0
        
        polyres.append(residues)

        roffset += len(residues)
        geometry.set_range_offset(roffset)

    if timing:
        t0 = time()
        
    # Set ribbon drawing geometry, colors, residue triangle ranges, and tethers
    if not geometry.empty():
        # Set drawing geometry
        va, na, ta = geometry.vertex_normal_triangle_arrays()
        ribbons_drawing.set_geometry(va, na, ta)
        # ribbons_drawing.display_style = rp.Mesh

        # Remember triangle ranges for each residue.
        from . import concatenate, Residues
        residues = concatenate(polyres, Residues)
        ribbons_drawing.set_triangle_ranges(residues, geometry.triangle_ranges)

        # Set colors
        ribbons_drawing.update_ribbon_colors()

        # Make tethers
        ribbons_drawing.set_tethers(tethered_atoms, backbone_atoms,
                                    structure.ribbon_tether_shape,
                                    structure.ribbon_tether_scale,
                                    structure.ribbon_tether_sides)

    if timing:
        drtime = time() - t0
        t0 = time()

    if timing:
        nres = sum(structure.residues.ribbon_displays)
        print('ribbon times %d polymers, %d residues, polymers %.4g, ranges %.4g, xsect %.4g, smooth %.4g, tube %.4g, path %.4g (spline %.4g (coef %.4g, normals %.4g), interpolate %.4g) , triangles %.4g, makedrawing %.4g, tethers %.4g'
              % (len(polymers), nres,
                 poltime, rangestime, xstime, smootime, tubetime,
                 pathtime, spltime, coeftime, normaltime, interptime,
                 geotime, drtime, tethertime))


def _get_polymer_spline(residues):
    '''Return a tuple of spline center and guide coordinates for a
    polymer chain.  Residues in the chain that do not have a center
    atom will have their display bit turned off.  Center coordinates
    are returned as a numpy array.  Guide coordinates are only returned
    if all spline atoms have matching guide atoms; otherwise, None is
    returned for guide coordinates.'''
    any_display, atom_pointers, centers, guides = _ribbons.get_polymer_spline(residues.pointers)
    from chimerax.atomic import Atoms
    atoms = None if atom_pointers is None else Atoms(atom_pointers)
    return any_display, atoms, centers, guides
    
def _ribbon_flip_normals(structure, is_helix):
    nres = len(is_helix)
    if structure.ribbon_mode_helix == structure.RIBBON_MODE_DEFAULT:
        last = nres - 1
        flip_normals = [(False if is_helix[i] and (i == last or is_helix[i + 1]) else True)
                        for i in range(nres)]
    else:
        flip_normals = [True] * nres
    return flip_normals

#
# Assign a residue class to each residue and compute the
# ranges of secondary structures.
# Returned helix and strand ranges (r0,r1) start with residue index r0 and end with index r1-1.
# Returned display_ranges (r0,r1) start with residue r0 and end with residue r1.
#
def _ribbon_ranges(is_helix, is_strand, ssids, displays, polymer_type, arc_helix):
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
        ranges = []
        s = 0
        for r0,r1 in helix_ranges:
            if r0 > 0:
                ranges.append((s,r0))
            s = r1-1
        if s < nr-1:
            ranges.append((s,nr-1))
    else:
        ranges = [(0,nr-1)]

    display_ranges = []
    for r0,r1 in ranges:
        display_ranges.extend(_displayed_subranges(r0, r1, displays))

    return res_class, helix_ranges, sheet_ranges, display_ranges

def _displayed_subranges(r0, r1, displays):
    sranges = []
    s = e = None
    for r in range(r0,r1+1):
        if displays[r]:
            if s is None:
                s = r
            e = r
        elif s is not None:
            sranges.append((s,e))
            s = e = None
    if s is not None:
        sranges.append((s,e))
    return sranges
                
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
            if rc == XSectionManager.RC_HELIX_START or rc == XSectionManager.RC_HELIX_END:
                xs_front[i] = xs_back[i] = ribbon_xs_mgr.xs_coil
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
    if rc1 is XSectionManager.RC_HELIX_START or rc1 is XSectionManager.RC_SHEET_START:
        return True
    return False

def _ribbon_geometry(path, ranges, num_res, xs_front, xs_back, geometry):        
    centers, tangents, normals = path
    xsf = [xs._xs_pointer for xs in xs_front]
    xsb = [xs._xs_pointer for xs in xs_back]
    _ribbons.ribbon_extrusions(centers, tangents, normals, ranges,
                               num_res, xsf, xsb, geometry._geom_cpp)
    
# Compute triangle geometry for ribbon.
# Only certain ranges of residues are considered, since not all
# residues need be displayed and also tube helix geometry is created by other code.
# TODO: This routine is taking half the ribbon compute time.  Probably a
#  big contributor is that 17 numpy arrays are being made per residue.
#  Might want to put TriangleAccumulator into C++ to get rid of half of those
#  and have extrude() put results directly into it.
#  Maybe Ribbon spline coords, tangents, normals could use recycled numpy arrays.
def _ribbon_geometry_unused(path, ranges, num_res, xs_front, xs_back, geometry):

    coords, tangents, normals = path
    nsp = len(coords) // num_res  # Path points per residue
    nlp, nrp = nsp // 2, (nsp + 1) // 2
    
    # Each residue has left and right half (also called front and back)
    # with the residue centered in the middle.
    # The two halfs can have different crosssections, e.g. turn and helix.
    # At the ends of the polymer the spline is extended to make the first residue
    # have a left half and the last residue have a right half.
    # If an interior range is shown only half segments are shown at the ends
    # since other code (e.g. tube cylinders) will render the other halfs.
    for r0,r1 in ranges:

        capped = True
        
        for i in range(r0, r1+1):
            # Left half
            mid_cap = (xs_front[i] != xs_back[i])
            s = i * nsp
            e = s + nlp + 1
            front_c, front_t, front_n = coords[s:e], tangents[s:e], normals[s:e]
            xs_front[i].extrude(front_c, front_t, front_n, capped, mid_cap, geometry)

            # Right half
            next_cap = True if i == r1 else (xs_back[i] != xs_front[i + 1])
            s = i * nsp + nlp
            e = s + nrp + 1
            back_c, back_t, back_n = coords[s:e], tangents[s:e], normals[s:e]
            xs_back[i].extrude(back_c, back_t, back_n, mid_cap, next_cap, geometry)

            capped = next_cap
            geometry.add_range(i)

class TriangleAccumulator:
    '''Accumulate triangles from segments of a ribbon.'''
    def __init__(self):
        self._geom_cpp = _ribbons.geometry_new()

    def __del__(self):
        _ribbons.geometry_delete(self._geom_cpp)
        self._geom_cpp = None
            
    def empty(self):
        return _ribbons.geometry_empty(self._geom_cpp)

    def add_range(self, residue_index):
        _ribbons.geometry_add_range(self._geom_cpp, residue_index)

    def set_range_offset(self, roffset):
        _ribbons.geometry_set_range_offset(self._geom_cpp, roffset)

    def vertex_normal_triangle_arrays(self):
        return _ribbons.geometry_arrays(self._geom_cpp)

    @property
    def triangle_ranges(self):
        return _ribbons.geometry_ranges(self._geom_cpp)
        
class TriangleAccumulator1:
    '''Accumulate triangles from segments of a ribbon.'''
    def __init__(self):
        self._v_start = 0         # for tracking starting vertex index for each residue
        self._v_end = 0
        self._t_start = 0         # for tracking starting triangle index for each residue
        self._t_end = 0
        self._vertex_list = []
        self._normal_list = []
        self._triangle_list = []
        self._triangle_ranges = []	# List of 5-tuples (residue_index, tstart, tend, vstart, vend)
        self._residue_base = 0

    def empty(self):
        return len(self._triangle_list) == 0
    
    @property
    def v_offset(self):
        return self._v_end
    
    def add_extrusion(self, extrusion, offset = False):
        e = extrusion
        self._add_geometry(e.vertices, e.normals, e.triangles, offset=offset)
    
    def _add_geometry(self, vertices, normals, triangles, offset = True):
        if offset:
            triangles += self._v_end
        self._v_end += len(vertices)
        self._t_end += len(triangles)
        self._vertex_list.append(vertices)
        self._normal_list.append(normals)
        self._triangle_list.append(triangles)

    def add_range(self, residue_index):
        ts,te,vs,ve = self._t_start, self._t_end, self._v_start, self._v_end
        if te == ts and ve == vs:
            return
        self._triangle_ranges.append((self._residue_base + residue_index, ts, te, vs, ve))
        self._t_start = te
        self._v_start = ve

    def set_range_offset(self, residue_base):
        self._residue_base = residue_base
        
    def vertex_normal_triangle_arrays(self):
        if self._vertex_list:
            va = concatenate(self._vertex_list)
            na = concatenate(self._normal_list)
            from chimerax.geometry import normalize_vectors
            normalize_vectors(na)
            ta = concatenate(self._triangle_list)
        else:
            va = na = ta = None
        return va, na, ta

    @property
    def triangle_ranges(self):
        return self._triangle_ranges

# -----------------------------------------------------------------------------
#
from chimerax.graphics import Drawing
class RibbonsDrawing(Drawing):
    def __init__(self, name, structure_name):
        Drawing.__init__(self, name)
        self.structure_name = structure_name
        self._tethers_drawing = None		# TethersDrawing
        self._triangle_ranges = None
        self._triangle_ranges_sorted = None	# Sorted ranges for first_intercept() calc
        self._residues = None			# Residues used with _triangle_ranges
        self._residues_count = 0		# For detecting deleted residues
        
    def clear(self):
        self.set_geometry(None, None, None)
        self.remove_all_drawings()
        self._tethers_drawing = None
        self._triangle_ranges = None
        self._residues = None

    def compute_ribbons(self, structure):
        if timing:
            t0 = time()
        _make_ribbon_graphics(structure, self)
        if timing:
            t1 = time()
            print ('compute_ribbons(): %.4g' % (t1-t0))

    def set_triangle_ranges(self, residues, triangle_ranges):
        self._residues = residues
        self._residues_count = len(residues)	# For detecting deleted residues
        self._triangle_ranges = triangle_ranges
        self._triangle_ranges_sorted = None
        
    def update_ribbon_colors(self):
        res = self._residues
        if res is None:
            return

        if timing:
            t0 = time()
            
        vc = self.vertex_colors
        if vc is None:
            vc = empty((len(self.vertices),4), uint8)

        _ribbons.ribbon_vertex_colors(res.pointers, self._triangle_ranges, vc)
#        rcolor = res.ribbon_colors
#        for i,ts,te,vs,ve in self._triangle_ranges:
#            vc[vs:ve,:] = rcolor[i]

        self.vertex_colors = vc

        if timing:
            t1 = time()
            print ('update_ribbon_colors(): %.4g' % (t1-t0))
        
    def update_ribbon_highlight(self):
        res = self._residues
        if res is None:
            return
        rsel = res.selected
        if rsel.any():
            sel_tranges = [(ts,te) for i,ts,te,vs,ve in self._triangle_ranges if rsel[i]]
        else:
            sel_tranges = []
        if sel_tranges:
            tmask = self.highlighted_triangles_mask
            if tmask is None:
                tmask = zeros((len(self.triangles),), bool)
            else:
                tmask[:] = False
            for s,e in sel_tranges:
                tmask[s:e] = True
        else:
            tmask = None
        self.highlighted_triangles_mask = tmask
        
    def set_tethers(self, tethered_atoms, backbone_atoms,
                    tether_shape, tether_scale, tether_sides):
        if len(tethered_atoms) == 0:
            return
        
        name = self.structure_name + " ribbon tethers"
        from . import concatenate, Atoms
        t_atoms = concatenate(tethered_atoms, Atoms)
        b_atoms = concatenate(backbone_atoms, Atoms)
            
        tether_drawing = TethersDrawing(name, t_atoms, b_atoms,
                                        tether_shape, tether_scale, tether_sides)
        self.add_drawing(tether_drawing)
        self._tethers_drawing = tether_drawing

    def update_tethers(self, structure):
        if timing:
            t0 = time()

        td = self._tethers_drawing
        if td:
            td.update_tethers(structure)

        if timing:
            t1 = time()
            print ('update_tethers(): %.4g' % (t1-t0))

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        if not self.display or (exclude and exclude(self)) or self._residues is None:
            return None
        p = super().first_intercept(mxyz1, mxyz2)
        if p is None or (hasattr(p, 'drawing') and p.drawing() is not self):
            return None
        if len(self._residues) < self._residues_count:
            return None		# Some residues have been deleted.
        tranges = self._triangle_ranges
        if self._triangle_ranges_sorted is None:
            # Sort by triangle start for bisect_right() search
            from numpy import argsort
            order = argsort(tranges[:,1])
            tranges[:] = tranges[order]
            self._triangle_ranges_sorted = tranges[:,1]
        from bisect import bisect_right
        n = bisect_right(self._triangle_ranges_sorted, p.triangle_number)
        if n > 0:
            ri = tranges[n-1,0]
            r = self._residues[ri]
            from .structure import PickedResidue
            return PickedResidue(r, p.distance)
        return None

    def planes_pick(self, planes, exclude=None):
        if not self.display or self._residues is None:
            return []
        if exclude is not None and exclude(self):
            return []
        tranges = self._triangle_ranges
        picks = []
        rp = super().planes_pick(planes)
        from chimerax.graphics import PickedTriangles
        from .structure import PickedResidues
        for p in rp:
            if isinstance(p, PickedTriangles) and p.drawing() is self:
                tmask = p._triangles_mask
                ires = [i for i,ts,te,vs,ve in tranges if tmask[ts:te].sum() > 0]
                if ires:
                    rc = self._residues.filter(ires)
                    picks.append(PickedResidues(rc))
        return picks

class TethersDrawing(Drawing):
    def __init__(self, name, tethered_atoms, backbone_atoms,
                 tether_shape, tether_scale, tether_sides):

        self._tethered_atoms = tethered_atoms
        self._backbone_atoms = backbone_atoms

        self._tether_shape = tether_shape
        self._tether_scale = tether_scale
        self._tether_sides = tether_sides
        
        Drawing.__init__(self, name)
        self.skip_bounds = True   # Don't include in bounds calculation. Optimization.
        self.pickable = False	# Don't allow mouse picking.

        if tether_shape == TETHER_CYLINDER:
            from chimerax.surface import cylinder_geometry
            va, na, ta = cylinder_geometry(nc=tether_sides, nz=2, caps=False)
        else:
            # Assume it's either TETHER_CONE or TETHER_REVERSE_CONE
            from chimerax.surface import cone_geometry
            va, na, ta = cone_geometry(nc=tether_sides, caps=False, points_up=False)
            # Instancing stretches the cone along its axis and distorts the normal
            # vectors which makes the cone too opaque.  So use cylinder normals
            # to mitigate the problem.  Bug #4797.
            na[:,2] = 0
            from chimerax.geometry import normalize_vectors
            normalize_vectors(na)
        self.set_geometry(va, na, ta)

    def update_tethers(self, structure):

        # Make backbone atoms as hidden unless it has a visible connecting atom.
        # For instance, hide CA atom unless CB atom is shown.
        self._backbone_atoms.update_ribbon_backbone_atom_visibility()

        # Set tether graphics for currently shown tether atoms.
        tatoms = self._tethered_atoms
        xyz1 = tatoms.ribbon_coords
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

def _ribbon_tethers(ribbon, residues, min_tether_offset):
    # Find position of backbone atoms on ribbon for drawing tethers
    t_atoms = _set_tether_positions(residues, ribbon.segment_coefficients)
    offsets = t_atoms.coords - t_atoms.ribbon_coords
    tethered = norm(offsets, axis=1) > min_tether_offset
    tethered_atoms = t_atoms.filter(tethered) if any(tethered) else None
    return tethered_atoms, t_atoms

def _set_tether_positions(residues, coef):
    # This _ribbons call sets atom.ribbon_coord to the position for each tethered atom.
    atom_pointers = _ribbons.set_atom_tether_positions(residues.pointers, coef)
    from . import Atoms
    atoms = Atoms(atom_pointers)
    return atoms

def _set_tether_positions_unused(residues, coef):
    nt_atoms, nt_positions = _atom_spline_positions(residues, _NonTetherPositions, coef)
    nt_atoms.ribbon_coords = nt_positions
    t_atoms, t_positions = _atom_spline_positions(residues, _TetherPositions, coef)
    t_atoms.ribbon_coords = t_positions
    return t_atoms

def _atom_spline_positions(residues, atom_offset_map, spline_coef):
    atom_pointers, positions = _ribbons.atom_spline_positions(residues.pointers, atom_offset_map, spline_coef)
    from . import Atoms
    atoms = Atoms(atom_pointers)
    return atoms, positions

def _atom_spline_positions_unused(residues, atom_offset_map, spline_coef):
    alist = []
    tlist = []
    for ri, r in enumerate(residues):
        for atom_name, offset in atom_offset_map.items():
            a = r.find_atom(atom_name)
            if a is not None and a.is_backbone():
                alist.append(a)
                tlist.append(ri + offset)
    positions = _spline_positions(tlist, spline_coef)
    from . import Atoms
    atoms = Atoms(alist)
    return atoms, positions

def _spline_positions(tlist, coef):
    xyz = empty((len(tlist),3), float64)
    n = len(coef)
    for i,s in enumerate(tlist):
        seg = int(s)
        t = s-seg
        if seg < 0:
            t += seg
            seg = 0
        elif seg > n-1:
            t += seg - (n-1)
            seg = n-1
        xyz[i,:] = dot(coef[seg], (1.0, t, t*t, t*t*t))
    return xyz

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
    "H":  -1/3.,
    "CA":  0.,
    "C":   1/3.,
    "OXT":  1/3.,
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

def _arc_helix_geometry(coords, xs_mgr, displays, start, end, geometry):
    '''Compute triangulation for one tube helix.'''

    from .sse import HelixCylinder
    hc = HelixCylinder(coords[start:end])
    centers = hc.cylinder_centers()
    radius = hc.cylinder_radius()
    normals, binormals = hc.cylinder_normals()
    from chimerax.geometry import cross_products
    tangents = cross_products(normals, binormals)
    icenters, inormals, ibinormals = hc.cylinder_intermediates()
    itangents = cross_products(inormals, ibinormals)

    c = _interleave_vectors(icenters, centers)
    t = _interleave_vectors(itangents, tangents)
    n = _interleave_vectors(inormals, normals)
    np = len(c)
    
    xsection = xs_mgr.xs_helix_tube(radius)
    for r in range(start, end):
        if displays[r]:
            i = 2*(r-start)
            s,e = i,i+3
            cap_front = (r == start or not displays[r-1])
            cap_back = (r == end-1 or not displays[r+1])
            xsection.extrude(c[s:e], t[s:e], n[s:e], cap_front, cap_back, geometry)
            geometry.add_range(r)

    return centers

def _interleave_vectors(u, v):
    uv = empty((len(u)+len(v),3), u.dtype)
    uv[::2] = u
    uv[1::2] = v
    return uv
        
def _wrap_helix(rlist, coords, guides, start, end):
    # Only bother if at least one residue is displayed
    displays = rlist.ribbon_displays
    if not any(displays[start:end]):
        return

    from .sse import HelixCylinder
    hc = HelixCylinder(coords[start:end])
    directions = hc.cylinder_directions()
    coords[start:end] = hc.cylinder_surface()
    if guides is not None:
        guides[start:end] = coords[start:end] + directions
    if False:
        # Debugging code to display guides of secondary structure
        name = ribbons_drawing.structure_name + " helix guide " + str(start)
        _ss_guide_display(ribbons_drawing, name, coords[start:end], guides[start:end])


def _smooth_strand(rlist, coords, guides, ribbon_adjusts, start, end):
    if (end - start + 1) <= 2:
        # Short strands do not need smoothing
        return
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
    if not any(displays[start:end]):
        return
    from .sse import StrandPlank
    atoms = rlist[start:end].atoms
    oxygens = atoms.filter(atoms.names == 'O')
    print(len(oxygens), "oxygens of", len(atoms), "atoms in", end - start, "residues")
    sp = StrandPlank(coords[start:end], oxygens.coords)
    centers = sp.plank_centers()
    normals, binormals = sp.plank_normals()
    if True:
        # Debugging code to display guides of secondary structure
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
    from numpy import argmax
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
    spine_radii = empty(len(spine_colors), float32)
    spine_radii.fill(0.3)
    sp.positions = _tether_placements(spine_xyz1, spine_xyz2, spine_radii, TETHER_CYLINDER)
    sp.colors = spine_colors


class Ribbon:

    def __init__(self, coords, guides, orients, flip_normals, smooth_twist, segment_divisions,
                 use_spline_normals):
        # Extend the coordinates at start and end to make sure the
        # ribbon is straight on either end.  Compute the spline
        # coefficients for each axis.  Then throw away the
        # coefficients for the fake ends.

        self._smooth_twist = smooth_twist
        self._segment_divisions = segment_divisions
        self._use_spline_normals = use_spline_normals
        
        if timing:
            t0 = time()
        self._coeff = _natural_cubic_spline_coefficients(coords)
        if timing:
            global coeftime
            coeftime += time() - t0

        if timing:
            t0 = time()
        # Currently Structure::ribbon_orient() defines the orientation method as
        # ATOMS for helices and coils, PEPTIDE for strands, and GUIDES for nucleic acids.
        atom_normals = None
        from .structure import Structure
        atom_mask = (orients == Structure.RIBBON_ORIENT_ATOMS)
        curvature_normals = None
        curvature_mask = (orients == Structure.RIBBON_ORIENT_CURVATURE)
        guide_normals = None
        guide_mask = ((orients == Structure.RIBBON_ORIENT_GUIDES) |
                       (orients == Structure.RIBBON_ORIENT_PEPTIDE))
        tangents = self.get_tangents()
        if atom_mask.any():
            atom_normals = _path_plane_normals(coords, tangents)
        if curvature_mask.any():
            curvature_normals = self._compute_normals_from_curvature(coords)
        if guide_mask.any():
            if guides is None or len(coords) != len(guides):
                if atom_normals is None:
                    guide_normals = _path_plane_normals(coords, tangents)
                else:
                    guide_normals = atom_normals
                guide_flip = False
            else:
                guide_normals = _compute_normals_from_guides(coords, guides, tangents)
                guide_flip = True
        if timing:
            global normaltime
            normaltime += time() - t0

        normals = None
        if atom_normals is not None:
            normals = atom_normals
        if curvature_normals is not None:
            if normals is None:
                normals = curvature_normals
            else:
                normals[curvature_mask] = curvature_normals[curvature_mask]
        if guide_normals is not None:
            if normals is None:
                normals = guide_normals
            else:
                normals[guide_mask] = guide_normals[guide_mask]
        self._normals = normals
        
        # Currently Structure._use_spline_normals = False.
        if use_spline_normals:
            self._flip_path_normals(coords)
            num_coords = len(coords)
            x = linspace(0.0, num_coords, num=num_coords, endpoint=False, dtype=float64)
            y = array(coords + self._normals, dtype=double)
            y2 = array(coords - self._normals, dtype=double)
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
            # Only prevent flipping normals where guides are missing.
            # Currently don't flip for nucleic acids missing guide atoms.
            if guide_normals is not None and not guide_flip:
                fnormals = [(flip if g else True)
                            for flip,g in zip(flip_normals, guide_mask)]
            else:
                fnormals = [True] * len(flip_normals)
            self._flip_normals = fnormals

    def _compute_normals_from_curvature(self, coords):
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

    @property
    def segment_coefficients(self):
        return self._coeff

    def _flip_path_normals(self, coords):
        from numpy import sqrt
        num_coords = len(coords)
        tangents = self.get_tangents()
        axes = cross(tangents[:-1], tangents[1:])
        s = norm(axes, axis=1)
        c = sqrt(1 - s * s)
        for i in range(1, num_coords):
            ne = self._rotate_around(axes[i-1], c[i-1], s[i-1], self._normals[i-1])
            n = self._normals[i]
            # Allow for a little extra twist before flipping
            if dot(ne, n) < 0:
            # if dot(ne, n) < 0.2:
                self._normals[i] = -n

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
        t = [[xcv[n,1], ycv[n,1], zcv[n,1]] for n in range(nc)]
        xc = xcv[-1]
        yc = ycv[-1]
        zc = zcv[-1]
        t.append((xc[1] + 2 * xc[2] + 3 * xc[3],
                   yc[1] + 2 * yc[2] + 3 * yc[3],
                   zc[1] + 2 * zc[2] + 3 * zc[3]))
        return normalize_vector_array(array(t))

    def path(self):
        coords, tangents, normals = _spline_path(self._coeff, self._normals, self._flip_normals,
                                                 self._smooth_twist, self._segment_divisions)
        return coords, tangents, normals

    def position(self, seg, t):
        # Compute coordinates for segment seg with parameter t
        return dot(self._coeff[seg], (1.0, t, t*t, t*t*t))

    def positions(self, tlist):
        # Compute coordinates on path for position t.
        # The integer part indicates the segment number,
        # and fractional part the position in the segment.
        return _spline_positions(tlist, self.coeff)
    
_natural_cubic_spline_coefficients = _ribbons.cubic_spline
def _natural_cubic_spline_coefficients_unused(coords):
    # Extend ends
    ne = len(coords) + 2
    ce = empty((ne, 3), float)
    ce[0] = coords[0] - (coords[1] - coords[0])
    ce[1:-1] = coords
    ce[-1] = coords[-1] + (coords[-1] - coords[-2])

    a = empty((ne,), float)
    b = empty((ne,), float)
    c = empty((ne,), float)
    d = empty((ne,), float)

    coef = empty((len(coords)-1,3,4), float64)
    for axis in range(3):
        values = ce[:,axis]
        # Cubic spline from http://mathworld.wolfram.com/CubicSpline.html
        # Set b[0] and b[-1] to 1 to match TomG code in VolumePath
        a[:] = 1
        b[:] = 4
        b[0] = b[-1] = 2
        #b[0] = b[-1] = 1
        c[:] = 1
        d[:] = 0
        d[0] = values[1] - values[0]
        d[1:-1] = 3 * (values[2:] - values[:-2])
        d[-1] = 3 * (values[-1] - values[-2])
        D = tridiagonal(a, b, c, d)

        delta = values[2:-1] - values[1:-2]
        coef[:,axis,0] = values[1:-2]
        coef[:,axis,1] = D[1:-2]
        coef[:,axis,2] = 3 * delta - 2 * D[1:-2] - D[2:-1]
        coef[:,axis,3] = 2 * -delta + D[1:-2] + D[2:-1]
    
    return coef

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

_path_plane_normals = _ribbons.path_plane_normals

def _path_plane_normals_unused(coords, tangents):
    '''
    Compute normal vectors to a path perpendicular to tangent vectors.
    The normal at a path point is obtained by taking the vector perpendicular
    to the segments to the preceeding and next points and taking its orthogonal
    component to the tangent vector.  If a normal at a point points opposite
    the normal of the preceding point (inner product < 0) then it is flipped.
    If a normal vector is zero for example due to 3 colinear path points
    then the normal is interpolated from the preceeding and following
    non-zero normals and orthogonal component to the tangent is taken.
    For leading or trailing zero normals the nearest non-zero normal is used
    and orthogonalized against the tangents.  If the whole path is straight
    or all normals are zero then arbitrary normals perpendicular to the
    tangents are used.
    '''
    #
    # Compute normals by cross-product of vectors to prev and next control points.
    # End points are same as second from end-points.
    #
    normals = empty(coords.shape, float)
    step = coords[:-1] - coords[1:]
    normals[1:-1] = cross(step[:-1], -step[1:])
    normals[0] = normals[1]
    normals[-1] = normals[-2]

    # Take component perpendicular to tangent vectors and make it unit length.
    num_zero = 0
    prev_normal = None
    num_pts = len(coords)
    for i in range(num_pts):
        n = _orthogonal_component(normals[i], tangents[i])
        d = norm(n)
        if d > 0:
            if prev_normal is not None and dot(n, prev_normal) < 0:
                d = -d   # Flip this normal to align better with previous one.
            n /= d
            normals[i] = n
            prev_normal = n
        else:
            normals[i] = (0,0,0)
            num_zero += 1

    if num_zero > 0:
        _replace_zero_normals(normals, tangents)

    return normals

def _replace_zero_normals(normals, tangents):
    # Some normal vector has zero length.  Replace it using nearby non-zero normals.
    s = 0
    num_pts = len(normals)
    while True:
        s = _next_zero_vector(s, normals)
        if s == num_pts:
            break
        e = _next_nonzero_vector(s+1, normals)
        if s == 0 and e < num_pts:
            # Set leading 0 normals to first non-zero normal.
            normals[:e] = normals[e]
        elif s > 0 and e < num_pts:
            # Linearly interpolate between non-zero normals.
            n0, n1 = normals[s-1], normals[e]
            for i in range(s,e):
                f = (i - (s-1)) / (e - (s-1))
                normals[i] = (1-f)*n0 + f*n1
        elif s > 0 and e == num_pts:
            # Set trailing 0 normals to last non-zero normal.
            normals[s:] = normals[s-1]
        elif s == 0 and e == num_pts:
            # All normals have zero length, possibly straight line path.
            for i in range(num_pts):
                normals[i] = _normal_vector(tangents[i])
        # Make new normals orthogonal to tangents and unit length.
        for i in range(s, e):
            n = _orthogonal_component(normals[i], tangents[i])
            d = norm(n)
            if d > 0:
                normals[i] = n / d
            else:
                n = _normal_vector(tangents[i])
                normals[i] = n / norm(n)
        s = e+1

def _next_zero_vector(start, normals):
    for i in range(start, len(normals)):
        if normals[i,0] == 0 and normals[i,1] == 0 and normals[i,2] == 0:
            return i
    return len(normals)

def _next_nonzero_vector(start, normals):
    for i in range(start, len(normals)):
        if not (normals[i,0] == 0 and normals[i,1] == 0 and normals[i,2] == 0):
            return i
    return len(normals)

_compute_normals_from_guides = _ribbons.path_guide_normals
def _compute_normals_from_guides_unused(coords, guides, tangents):
    n = guides - coords
    normals = zeros((len(coords), 3), float)
    for i in range(len(coords)):
        normals[i,:] = _orthogonal_component(n[i], tangents[i])
    return normalize_vector_array(normals)

def _orthogonal_component(v, ref):
    d = inner(v, ref)
    ref_len = norm(ref)
    return v + ref * (-d / ref_len)

def _normal_vector(v):
    if v[0] != 0 or v[1] != 0:
        n = (-v[1], v[0], v[2])
    else:
        n = (-v[2], v[1], v[0])
    return n

def _path_plane_tests():

    n = 7
    straight = zeros((n,3),float32)
    for i in range(n):
        straight[i,2] = i
    tangents = zeros((n,3),float32)
    tangents[:,2] = 1

    def _print_results(descrip):
        normals = _path_plane_normals(coords, tangents)
        print (descrip)
        print ('coords\n', coords)
        print ('tangents\n', tangents)
        print ('normals\n', normals)

    # All coords 0
    coords = zeros((n,3), float32)
    _print_results('All 0 path')

    # Straight path
    coords = straight
    _print_results('Straight path')

    # Straight leading and trailing segments
    coords = straight.copy()
    coords[3] = (1,.5,3)
    _print_results('Leading and trailing straight')

    # Straight trailing segment
    coords = straight.copy()
    coords[0] = (1,1,0)
    _print_results('Trailing straight')

    # Straight leading segment
    coords = straight.copy()
    coords[n-1] = (1,-1,n-1)
    _print_results('Leading straight')
    
    # Straight middle segment
    coords = straight.copy()
    coords[0] = (1,0,0)
    coords[n-1] = (0,1,n-1)
    _print_results('Straight middle')
    
    # All curved
    coords = straight.copy()
    from math import sin, cos
    for i in range(n):
        coords[i] = (cos(i), sin(i), i)
    _print_results('All curved')
    
_spline_path = _ribbons.spline_path
def _spline_path_unused(coeffs, start_normals, flip_normals, twist, ndiv):
    lead = _spline_path_lead_segment(coeffs[0], start_normals[0], ndiv//2)
    geom = [ lead ]

    nseg = len(coeffs)
    end_normal = None
    for seg in range(nseg):
        coords, tangents = _spline_segment_path(coeffs[seg], 0, 1, ndiv+1)
        start_normal = start_normals[seg] if end_normal is None else end_normal
        normals = _ribbons.parallel_transport(tangents, start_normal)
        if twist[seg]:
            end_normal = start_normals[seg + 1]
            if flip_normals[seg] and _need_normal_flip(normals[-1], end_normal, tangents[-1]):
                end_normal = -end_normal
            _ribbons.smooth_twist(tangents, normals, end_normal)
        spath = (coords[:-1], tangents[:-1], normals[:-1])
        geom.append(spath)

    trail = _spline_path_trail_segment(coeffs[-1], end_normal, (ndiv + 1)//2)
    geom.append(trail)

    npp = (nseg + 1) * ndiv
    coords, tangents, normals = _concatenate_paths(geom, npp)
    return coords, tangents, normals

def _spline_path_lead_segment(coeffs, normal, n):
    coords, tangents = _spline_segment_path(coeffs, -0.3, 0, n+1)
    # Parallel transport normal backwards
    normals = _ribbons.parallel_transport(tangents[::-1], normal)[::-1]
    # Don't include right end point.
    return coords[:-1], tangents[:-1], normals[:-1]

def _spline_path_trail_segment(coeffs, normal, n):
    coords, tangents = _spline_segment_path(coeffs, 1, 1.3, n)
    normals = _ribbons.parallel_transport(tangents, normal)
    return coords, tangents, normals

# Decide whether to flip the spline segment end normal so that it aligns better with
# the parallel transported normal.
def _need_normal_flip(transported_normal, end_normal, tangent):

    # If twist is greater than 90 degrees, turn the opposite
    # direction.  (Assumes that ribbons are symmetric.)
    a = _ribbons.dihedral_angle(transported_normal, end_normal, tangent)
    from math import pi
    # flip = (abs(a) > 0.5 * pi)
    flip = (abs(a) > 0.6 * pi)	# Not sure why this is not pi / 2.
    return flip

def _concatenate_paths(paths, num_pts):
    # Concatenate segment paths
    coords, tangents, normals = [empty((num_pts,3),float32) for i in range(3)]
    o = 0
    for c,t,n in paths:
        nsp = len(c)
        coords[o:o+nsp] = c
        tangents[o:o+nsp] = t
        normals[o:o+nsp] = n
        o += nsp
    return coords, tangents, normals

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
        self.tube_radius = None		# If None each helix calculates its own radius
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

        self._xs_helix = None
        self._xs_helix_arrow = None
        self._xs_helix_tube = None
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
            self._xs_helix_tube = None
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
                self._xs_helix_tube = None
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

    def xs_helix_tube(self, radius):
        r = self.tube_radius
        if r is None:
            r = radius
        xs = self._xs_helix_tube
        if xs is None or xs.radius != r:
            scale = (r,r)
            self._xs_helix_tube = xs = self._make_xs(self.STYLE_ROUND, scale)
            xs.radius = r
        return xs

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
            param = self.params[self.STYLE_ROUND]
            return _xsection_round(scale, param['sides'], param['faceted'])
        elif style is self.STYLE_SQUARE:
            return _xsection_square(scale)
        elif style is self.STYLE_PIPING:
            param = self.params[self.STYLE_PIPING]
            return _xsection_piping(scale, param['sides'], param['ratio'], param['faceted'])
        else:
            raise ValueError("unknown style %s" % style)

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

# -----------------------------------------------------------------------------
#
from collections import namedtuple
ExtrudeValue = namedtuple("ExtrudeValue", ["vertices", "normals", "triangles"])

# Cross section coordinates are 2D and counterclockwise
# Use C++ version of RibbonXSection instead of Python version
class RibbonXSection:
    '''
    A cross section that can extrude ribbons when given the
    required control points, tangents, normals and colors.
    '''
    def __init__(self, coords=None, coords2=None, normals=None, normals2=None,
                 faceted=False, tess=None, xs_pointer=None):
        if xs_pointer is None:
            kw = {name:value for name,value in (('coords',coords), ('coords2', coords2),
                                                ('normals',normals), ('normals2',normals2),
                                                ('faceted',faceted), ('tess',tess))
                  if value is not None}
            xs_pointer = _ribbons.rxsection_new(**kw)
        self._xs_pointer = xs_pointer

    def __del__(self):
        _ribbons.rxsection_delete(self._xs_pointer)
        self._xs_pointer = None
    
    def extrude(self, centers, tangents, normals,
                cap_front, cap_back, geometry):
        '''Return the points, normals and triangles for a ribbon.'''
        _ribbons.rxsection_extrude(self._xs_pointer, centers, tangents, normals,
                                   cap_front, cap_back, geometry._geom_cpp)

    def scale(self, scale):
        '''Return new cross section scaled by 2-tuple scale.'''
        p = _ribbons.rxsection_scale(self._xs_pointer, scale[0], scale[1])
        return RibbonXSection(xs_pointer=p)

    def arrow(self, scales):
        '''Return new arrow cross section scaled by 2x2-tuple scale.'''
        p = _ribbons.rxsection_arrow(self._xs_pointer, scales[0][0], scales[0][1],
                                     scales[1][0], scales[1][1])
        return RibbonXSection(xs_pointer=p)

def _xsection_round(scale, sides, faceted = False):
    coords = []
    normals = []
    from numpy import cos, sin, stack
    from math import pi
    angles = linspace(0, 2 * pi, sides, endpoint=False)
    ca = cos(angles)
    sa = sin(angles)
    circle = stack((ca, sa), axis=1)
    coords = circle * array(scale)
    if faceted:
        xs = RibbonXSection(coords, faceted=True)
    else:
        normals = circle * array((scale[1], scale[0]))
        xs = RibbonXSection(coords, normals=normals, faceted=False)
    return xs

def _xsection_square(scale):
    coords = array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))) * array(scale)
    return RibbonXSection(coords, faceted=True)

def _xsection_piping(scale, sides, ratio, faceted=False):
    from math import pi, cos, sin, asin
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
    # The total number of sides is sides.
    # We subtract the two connectors and then divide
    # by two while rounding up
    sides = (sides - 1) // 2
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
    if faceted:
        xs = RibbonXSection(coords, faceted=True, tess=tess)
    else:
        if flipped:
            normals[0] = normals[-1] = (1, 0)
            normals[sides] = normals[sides + 1] = (-1, 0)
        else:
            normals[0] = normals[-1] = (0, -1)
            normals[sides] = normals[sides + 1] = (0, 1)
        normals = array(normals)
        xs = RibbonXSection(coords, normals=normals, faceted=False, tess=tess)
    return xs

def normalize(v):
    # normalize a single vector
    from numpy import isnan
    d = norm(v)
    if isnan(d) or d < EPSILON:
        return v
    return v / d


def normalize_vector_array(a):
    import numpy
    d = norm(a, axis=1)
    d[numpy.isnan(d)] = 1
    d[d < EPSILON] = 1
    n = a / d[:, numpy.newaxis]
    return n

_spline_segment_path = _ribbons.cubic_path

def _spline_segment_path_unused(coeffs, tmin, tmax, num_points):
    # coeffs is a 3x4 array of float.
    # Compute coordinates by multiplying spline parameter vector
    # (1, t, t**2, t**3) by the spline coefficients, and
    # compute tangents by multiplying spline parameter vector
    # (0, 1, 2*t, 3*t**2) by the same spline coefficients
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
    return coords, tangents

def curvature_to_normals(curvature, tangents, prev_normal):
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
