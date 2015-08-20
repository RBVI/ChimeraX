# vi: set expandtab shiftwidth=4 softtabstop=4:
from .. import io
from ..models import Model
from ..session import RestoreError
from .molobject import AtomicStructureData

CATEGORY = io.STRUCTURE


class AtomicStructure(AtomicStructureData, Model):
    """
    Molecular model including atomic coordinates.
    The data is managed by the :class:`.AtomicStructureData` base class
    which provides access to the C++ structures.
    """

    STRUCTURE_STATE_VERSION = 0

    def __init__(self, name, atomic_structure_pointer = None,
                 initialize_graphical_attributes = True):

        AtomicStructureData.__init__(self, atomic_structure_pointer)
        from . import molobject
        molobject.add_to_object_map(self)

        Model.__init__(self, name)

        self._atoms = None              # Cached atoms array

        self.ball_scale = 0.3		# Scales sphere radius in ball and stick style
        self.bond_radius = 0.2
        self.pseudobond_radius = 0.05
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._pseudobond_group_drawings = {}    # Map name to drawing
        self.triangles_per_sphere = None
        self._cached_atom_bounds = None
        self._atom_bounds_needs_update = True

        self.ribbon_divisions = 10
        self._ribbon_drawing = None
        self._ribbon_t2r = {}         # ribbon triangles-to-residue map
        self._ribbon_r2t = {}         # ribbon residue-to-triangles map
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
        self._ribbon_xs_turn = XSection(xsc_turn, faceted=True)
        self._ribbon_xs_arrow = XSection(xsc_arrow_head, xsc_arrow_tail, faceted=True)
        self._ribbon_selected_residues = set()

        self._make_drawing(initialize_graphical_attributes)

    def delete(self):
        self._atoms = None
        AtomicStructureData.delete(self)
        Model.delete(self)

    def copy(self, name):
        m = AtomicStructure(name, AtomicStructureData._copy(self),
                            initialize_graphical_attributes = False)
        m.positions = self.positions
        return m

    def added_to_session(self, session):
        v = session.main_view
        v.add_callback('graphics update', self._update_graphics_if_needed)

    def removed_from_session(self, session):
        v = session.main_view
        v.remove_callback('graphics update', self._update_graphics_if_needed)

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
        return self.pbg_map

    def shown_atom_count(self):
        na = sum(self.atoms.displays) if self.display else 0
        return na

    def _initialize_graphical_attributes(self):
        a = self.atoms
        from .molobject import Atom
        a.draw_modes = Atom.SPHERE_STYLE
        from ..colors import element_colors
        a.colors = element_colors(a.element_numbers)
        b = self.bonds
        b.radii = self.bond_radius
        pb_colors = {'metal coordination bonds':(147,112,219,255)}
        for name, pbg in self.pseudobond_groups.items():
            pb = pbg.pseudobonds
            pb.radii = self.pseudobond_radius
            pb.halfbonds = False
            pb.colors = pb_colors.get(name, (255,255,0,255))

    def _make_drawing(self, initialize_graphical_attributes):

        if initialize_graphical_attributes:
            self._initialize_graphical_attributes()

        a = self.atoms
        coords = a.coords
        radii = self._atom_display_radii()
        colors = a.colors
        display = a.displays
        b = self.bonds
        pbgs = self.pseudobond_groups

        # Create graphics
        self._create_atom_spheres(coords, radii, colors, display)
        self._update_bond_graphics(b.atoms, a.draw_modes, b.radii, b.colors, b.halfbonds)
        for name, pbg in pbgs.items():
            pb = pbg.pseudobonds
            self._update_pseudobond_graphics(name, pb.atoms, pb.radii, pb.colors, pb.halfbonds)
        self._update_ribbon_graphics()

    def _create_atom_spheres(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            self._atoms_drawing = p = self.new_drawing('atoms')
            # Add atom picking method to the atom drawing
            from types import MethodType
            p.first_intercept = MethodType(_atom_first_intercept, p)

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

    def _update_graphics_if_needed(self):
        c, s, se = self._gc_color, self._gc_shape, self._gc_select
        if c or s or se:
            self._gc_color = self._gc_shape = self._gc_select = False
            self._update_graphics()
            self.redraw_needed(shape_changed = s, selection_changed = se)
            if s:
                self._atom_bounds_needs_update = True

    def _update_graphics(self):
        a = self.atoms
        b = self.bonds
        self._update_atom_graphics(a.coords, self._atom_display_radii(), a.colors, a.displays)
        self._update_bond_graphics(b.atoms, a.draw_modes, b.radii, b.colors, b.halfbonds)
        pbgs = self.pseudobond_groups
        for name, pbg in pbgs.items():
            pb = pbg.pseudobonds
            self._update_pseudobond_graphics(name, pb.atoms, pb.radii, pb.colors, pb.halfbonds)
        self._update_ribbon_graphics(rebuild = True)

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

    def _atom_display_radii(self):
        a = self.atoms
        r = a.radii.copy()
        dm = a.draw_modes
        from .molobject import Atom
        r[dm == Atom.BALL_STYLE] *= self.ball_scale
        r[dm == Atom.STICK_STYLE] = self.bond_radius
        return r

    def _update_bond_graphics(self, bond_atoms, draw_mode, radii,
                              bond_colors, half_bond_coloring):
        p = self._bonds_drawing
        if p is None:
            from .molobject import Atom
            if (draw_mode == Atom.SPHERE_STYLE).all():
                return
            self._bonds_drawing = p = self.new_drawing('bonds')
            # Suppress bond picking since bond selections are not supported.
            from types import MethodType
            p.first_intercept = MethodType(_bond_first_intercept, p)
            from .. import surface
            # Use 3 z-sections so cylinder ends match in half-bond mode.
            va, na, ta = surface.cylinder_geometry(nz = 3, caps = False)
            p.geometry = va, ta
            p.normals = na

        xyz1, xyz2 = bond_atoms[0].coords, bond_atoms[1].coords
        p.positions = _bond_cylinder_placements(xyz1, xyz2, radii, half_bond_coloring)
        p.display_positions = self._shown_bond_cylinders(bond_atoms, half_bond_coloring)
        self._set_bond_colors(p, bond_atoms, bond_colors, half_bond_coloring)

    def _set_bond_colors(self, drawing, bond_atoms, bond_colors, half_bond_coloring):
        p = drawing
        if p is None:
            return

        if half_bond_coloring.any():
            bc0,bc1 = bond_atoms[0].colors, bond_atoms[1].colors
            from numpy import concatenate
            p.colors = concatenate((bc0,bc1))
        else:
            p.colors = bond_colors

    def _shown_bond_cylinders(self, bond_atoms, half_bond_coloring):
        sb = bond_atoms[0].displays & bond_atoms[1].displays  # Show bond if both atoms shown
        from .molobject import Atom
        ns = ((bond_atoms[0].draw_modes != Atom.SPHERE_STYLE) |
              (bond_atoms[1].draw_modes != Atom.SPHERE_STYLE))       # Don't show if both atoms in sphere style
        import numpy
        numpy.logical_and(sb,ns,sb)
        if half_bond_coloring.any():
            sb2 = numpy.concatenate((sb,sb))
            return sb2
        return sb

    def _update_pseudobond_graphics(self, name, bond_atoms, radii,
                                    bond_colors, half_bond_coloring):
        pg = self._pseudobond_group_drawings
        if not name in pg:
            pg[name] = p = self.new_drawing(name)
            va, na, ta = _pseudobond_geometry()
            p.geometry = va, ta
            p.normals = na
        else:
            p = pg[name]

        xyz1, xyz2 = bond_atoms[0].coords, bond_atoms[1].coords
        p.positions = _bond_cylinder_placements(xyz1, xyz2, radii, half_bond_coloring)
        p.display_positions = self._shown_bond_cylinders(bond_atoms, half_bond_coloring)
        self._set_bond_colors(p, bond_atoms, bond_colors, half_bond_coloring)

    def _update_ribbon_graphics(self, rebuild=False):
        if rebuild:
            from .ribbon import Ribbon, XSection
            from numpy import concatenate, array, uint8
            polymers = self.polymers(False, False)
            if self._ribbon_drawing is None:
                self._ribbon_drawing = p = self.new_drawing('ribbon')
                p.display = True
            else:
                p = self._ribbon_drawing
                p.remove_all_drawings()
            self._ribbon_t2r = {}
            self._ribbon_r2t = {}
            for rlist in polymers:
                rp = p.new_drawing(rlist.strs[0])
                t2r = []
                displays = rlist.ribbon_displays
                if displays.sum() == 0:
                    continue
                coords, guides = self._get_polymer_spline(rlist)
                if len(coords) < 4:
                    continue
                any_ribbon = True
                ribbon = Ribbon(coords, guides)
                offset = 0
                vertex_list = []
                normal_list = []
                color_list = []
                triangle_list = []
                # Draw first and last residue differently because they
                # are each only a single half segment, where the middle
                # residues are each two half segments.
                if self.ribbon_divisions % 2 == 1:
                    seg_blend = self.ribbon_divisions
                    seg_cap = seg_blend + 1
                else:
                    seg_cap = self.ribbon_divisions
                    seg_blend = seg_cap + 1
                is_helix = rlist.is_helix
                is_sheet = rlist.is_sheet
                colors = rlist.ribbon_colors
                # Assign cross sections
                xss = []
                was_strand = False
                for i in range(len(rlist)):
                    if is_sheet[i]:
                        xss.append(self._ribbon_xs_strand)
                        was_strand = True
                    else:
                        if was_strand:
                            xss[-1] = self._ribbon_xs_arrow
                        if is_helix[i]:
                            xss.append(self._ribbon_xs_helix)
                        else:
                            xss.append(self._ribbon_xs_turn)
                        was_strand = False
                # Per-residue state variables
                t_start = 0
                # First residues
                if displays[0]:
                    capped = displays[0] != displays[1] or xss[0] != xss[1]
                    seg = capped and seg_cap or seg_blend
                    centers, tangents, normals = ribbon.segment(0, ribbon.FRONT, seg)
                    s = xss[0].extrude(centers, tangents, normals, colors[0],
                                       True, capped, offset)
                    offset += len(s.vertices)
                    t_end = t_start + len(s.triangles)
                    vertex_list.append(s.vertices)
                    normal_list.append(s.normals)
                    triangle_list.append(s.triangles)
                    color_list.append(s.colors)
                    prev_band = s.back_band
                    triangle_range = RibbonTriangleRange(t_start, t_end, rp, rlist[0])
                    t2r.append(triangle_range)
                    self._ribbon_r2t[rlist[0]] = triangle_range
                    t_start = t_end
                else:
                    capped = True
                    prev_band = None
                # Middle residues
                for i in range(1, len(rlist) - 1):
                    if not displays[i]:
                        continue
                    seg = capped and seg_cap or seg_blend
                    front_c, front_t, front_n = ribbon.segment(i - 1, ribbon.BACK, seg)
                    next_cap = displays[i] != displays[i + 1] or xss[i] != xss[i + 1]
                    seg = next_cap and seg_cap or seg_blend
                    back_c, back_t, back_n = ribbon.segment(i, ribbon.FRONT, seg)
                    centers = concatenate((front_c, back_c))
                    tangents = concatenate((front_t, back_t))
                    normals = concatenate((front_n, back_n))
                    s = xss[i].extrude(centers, tangents, normals, colors[i],
                                       capped, next_cap, offset)
                    offset += len(s.vertices)
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
                    triangle_range = RibbonTriangleRange(t_start, t_end, rp, rlist[i])
                    t2r.append(triangle_range)
                    self._ribbon_r2t[rlist[i]] = triangle_range
                    t_start = t_end
                # Last residue
                if displays[-1]:
                    seg = capped and seg_cap or seg_blend
                    centers, tangents, normals = ribbon.segment(ribbon.num_segments - 1, ribbon.BACK, seg, last=True)
                    s = xss[-1].extrude(centers, tangents, normals, colors[-1],
                                        capped, True, offset)
                    offset += len(s.vertices)
                    t_end = t_start + len(s.triangles)
                    vertex_list.append(s.vertices)
                    normal_list.append(s.normals)
                    triangle_list.append(s.triangles)
                    color_list.append(s.colors)
                    if prev_band:
                        triangle_list.append(xss[-1].blend(prev_band, s.front_band))
                        t_end += len(triangle_list[-1])
                    triangle_range = RibbonTriangleRange(t_start, t_end, rp, rlist[-1])
                    t2r.append(triangle_range)
                    self._ribbon_r2t[rlist[-1]] = triangle_range
                    t_start = t_end
                # Create drawing from arrays
                rp.display = True
                rp.vertices = concatenate(vertex_list)
                rp.normals = concatenate(normal_list)
                rp.triangles = concatenate(triangle_list)
                rp.vertex_colors = concatenate(color_list)
                rp.atomic_structure = self
                from types import MethodType
                rp.first_intercept = MethodType(_ribbon_first_intercept, rp)
                # Save mappings for picking
                self._ribbon_t2r[rp] = t2r

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

    def _get_polymer_spline(self, rlist):
            # Get coordinates for spline and orientation atoms
            coords = []
            guides = []
            has_guides = True
            for r in rlist:
                c = None
                g = None
                for a in r.atoms:
                    atom_name = a.name
                    if atom_name in [ "CA", "C5'" ]:
                        c = a.coord
                    elif atom_name in [ "O", "C1'" ]:
                        g = a.coord
                if c is None:
                    continue
                coords.append(c)
                if g is None:
                    has_guides = False
                else:
                    guides.append(g)
            if has_guides:
                return coords, guides
            else:
                return coords, None

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
        xyz = a.coords[a.displays]
        from ..geometry import bounds
        b = bounds.point_bounds(xyz)
        self._cached_atom_bounds = b
        self._atom_bounds_needs_update = False
        return b

    def _ribbon_bounds(self):
        rd = self._ribbon_drawing
        if rd is None or not rd.display:
            return None
        return rd.bounds()

    def select_atom(self, atom, toggle=False, selected=True):
        atom.selected = (not atom.selected) if toggle else selected
        self._selection_changed()

    def select_atoms(self, atoms, toggle=False, selected=True):
        asel = self.atoms.selected
        m = self.atoms.mask(atoms)
        from numpy import logical_not
        asel[m] = logical_not(asel[m]) if toggle else selected
        self.atoms.selected = asel
        self._selection_changed()

    def select_residue(self, residue, toggle=False, selected=True):
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
        from .molsurf import MolecularSurface
        surfs = [s for s in self.child_models() if isinstance(s, MolecularSurface)]
        return surfs

def selected_atoms(session):
    from .molarray import Atoms
    atoms = Atoms()
    for m in session.models.list():
        if isinstance(m, AtomicStructure):
            for matoms in m.selected_items('atoms'):
                atoms = atoms | matoms
    return atoms

def _atom_first_intercept(self, mxyz1, mxyz2, exclude = None):
    # TODO check intercept of bounding box as optimization
    xyzr = self.positions.shift_and_scale_array()
    dp = self.display_positions
    xyz,r = (xyzr[:,:3], xyzr[:,3]) if dp is None else (xyzr[dp,:3], xyzr[dp,3])

    # Check for atom sphere intercept
    from .. import graphics
    fmin, anum = graphics.closest_sphere_intercept(xyz, r, mxyz1, mxyz2)

    if fmin is None:
        return None

    if not dp is None:
        anum = dp.nonzero()[0][anum]    # Remap index to include undisplayed positions
    atom = self.parent.atoms[anum]

    # Create pick object
    s = PickedAtom(atom, fmin)

    return s

def _bond_first_intercept(self, mxyz1, mxyz2, exclude = None):
    return None

def _ribbon_first_intercept(self, mxyz1, mxyz2, exclude=None):
    # TODO check intercept of bounding box as optimization
    from ..graphics import Drawing
    pd = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
    if pd is None:
        return None
    best = pd.distance
    t2r = self.atomic_structure._ribbon_t2r[pd.drawing()]
    from bisect import bisect_right
    n = bisect_right(t2r, pd.triangle_number)
    if not n:
        return None
    triangle_range = t2r[n - 1]
    return PickedResidue(triangle_range.residue, pd.distance)

# -----------------------------------------------------------------------------
#
from ..graphics import Pick
class PickedAtom(Pick):
    def __init__(self, atom, distance):
        Pick.__init__(self, distance)
        self.atom = atom
    def description(self):
        return atom_description(self.atom)
    def drawing(self):
        return self.atom.structure
    def select(self, toggle = False):
        a = self.atom
        a.structure.select_atom(a, toggle)

# -----------------------------------------------------------------------------
#
def atom_description(atom):
    m = atom.structure
    r = atom.residue
    d = '%s #%s.%s %s %d %s' % (m.name, m.id_string(), r.chain_id, r.name, r.number, atom.name)
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
    def drawing(self):
        return self.residue.structure
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
# Return 4x4 matrices taking prototype cylinder to bond location.
#
def _bond_cylinder_placements(axyz0, axyz1, radius, half_bond):

  # TODO: Allow per-bond variation in half-bond mode.
  half_bond = half_bond.any()

  n = len(axyz0)
  from numpy import empty, float32, transpose, sqrt, array
  nc = 2*n if half_bond else n
  p = empty((nc,4,4), float32)
  
  p[:,3,:] = (0,0,0,1)
  if half_bond:
    p[:n,:3,3] = 0.75*axyz0 + 0.25*axyz1
    p[n:,:3,3] = 0.25*axyz0 + 0.75*axyz1
  else:
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
  h = 0.5*d if half_bond else d
  rs = array(((r*(cx*wx + c), r*cx*wy,  h*wy),
              (r*cy*wx, r*(cy*wy + c), -h*wx),
              (-r*wy, r*wx, h*c)), float32).transpose((2,0,1))
  if half_bond:
    p[:n,:3,:3] = rs
    p[n:,:3,:3] = rs
  else:
    p[:,:3,:3] = rs
  pt = transpose(p,(0,2,1))
  from ..geometry import Places
  pl = Places(opengl_array = pt)
  return pl

# -----------------------------------------------------------------------------
#
def _pseudobond_geometry(segments = 9):
    from .. import surface
    return surface.dashed_cylinder_geometry(segments)

# -----------------------------------------------------------------------------
#
def all_atomic_structures(session):
    return [m for m in session.models.list() if isinstance(m,AtomicStructure)]

# -----------------------------------------------------------------------------
#
def all_atoms(session):
    from .molarray import Atoms
    atoms = Atoms()
    for m in all_atomic_structures(session):
        atoms = atoms | m.atoms
    return atoms
