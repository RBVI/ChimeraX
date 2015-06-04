# vi: set expandtab shiftwidth=4 softtabstop=4:
from . import io
from . import models
from .session import RestoreError
from .molecule import CAtomicStructure

CATEGORY = io.STRUCTURE


class AtomicStructure(CAtomicStructure, models.Model):
    """Commom base class for atomic structures"""

    STRUCTURE_STATE_VERSION = 0

    # Atom display styles, Atom.draw_modes
    SPHERE_STYLE = 1
    BALL_STYLE = 2
    STICK_STYLE = 3

    def __init__(self, name, atomic_structure_pointer = None):

        CAtomicStructure.__init__(self, atomic_structure_pointer)
        from . import molecule
        molecule.add_to_object_map(self)

        models.Model.__init__(self, name)

        self._atoms = None              # Cached atoms array
        self._residues = None           # Cached residues array

        self.ball_scale = 0.3		# Scales sphere radius in ball and stick style
        self.bond_radius = 0.2
        self.pseudobond_radius = 0.05
        self.ribbon_divisions = 10
        self._atoms_drawing = None
        self._bonds_drawing = None
        self._pseudobond_group_drawings = {}    # Map name to drawing
        self._ribbon_drawing = None
        self._selected_atoms = None	# Numpy array of bool, size equal number of atoms
        self.triangles_per_sphere = None
        self._atom_bounds = None
        self._atom_bounds_needs_update = True

        self.make_drawing()

    def delete(self):
        self._atoms = None
        CAtomicStructure.delete(self)
        models.Model.delete(self)

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
        if self._atoms is None:
            self._atoms = CAtomicStructure.atoms.fget(self)
        return self._atoms

    @property
    def pseudobond_groups(self):
        return self.pbg_map

    def shown_atom_count(self):
        na = sum(self.atoms.displays) if self.display else 0
        return na

    def solvent_atoms(self):
        atoms = self.atoms
        from numpy import array
        return atoms.filter(array(atoms.residues.names) == 'HOH')

    def show_atoms(self, atoms = None):
        if atoms is None:
            atoms = self.atoms
        atoms.displays = True
        self.update_graphics()

    def hide_atoms(self, atoms = None):
        if atoms is None:
            atoms = self.atoms
        atoms.displays = False
        self.update_graphics()

    def initialize_graphical_attributes(self):
        a = self.atoms
        a.draw_modes = self.SPHERE_STYLE
        a.colors = element_colors(a.element_numbers)
        b = self.bonds
        b.radii = self.bond_radius
        pb_colors = {'metal coordination bonds':(147,112,219,255)}
        for name, pb in self.pseudobond_groups.items():
            pb.radii = self.pseudobond_radius
            pb.halfbonds = False
            pb.colors = pb_colors.get(name, (255,255,0,255))

    def make_drawing(self):

        self.initialize_graphical_attributes()

        a = self.atoms
        coords = a.coords
        radii = self.atom_display_radii()
        colors = a.colors
        display = a.displays
        b = self.bonds
        pbg = self.pseudobond_groups

        # Create graphics
        self.create_atom_spheres(coords, radii, colors, display)
        self.update_bond_graphics(b.atoms, a.draw_modes, b.radii, b.colors, b.halfbonds)
        for name, pb in pbg.items():
            self.update_pseudobond_graphics(name, pb.atoms, pb.radii, pb.colors, pb.halfbonds)
        self.update_ribbon_graphics()

    def create_atom_spheres(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            self._atoms_drawing = p = self.new_drawing('atoms')

        n = len(coords)
        self.triangles_per_sphere = 320 if n < 30000 else 80 if n < 120000 else 20

        # Set instanced sphere triangulation
        from . import surface
        va, na, ta = surface.sphere_geometry(self.triangles_per_sphere)
        p.geometry = va, ta
        p.normals = na

        self.update_atom_graphics(coords, radii, colors, display)

    def new_atoms(self):
        self._atoms = None
        self.update_graphics()

    def update_graphics(self):
        a = self.atoms
        b = self.bonds
        self.update_atom_graphics(a.coords, self.atom_display_radii(), a.colors, a.displays)
        self.update_bond_graphics(b.atoms, a.draw_modes, b.radii, b.colors, b.halfbonds)
        pbg = self.pseudobond_groups
        for name, pb in pbg.items():
            self.update_pseudobond_graphics(name, pb.atoms, pb.radii, pb.colors, pb.halfbonds)
        self.update_ribbon_graphics()
        self._atom_bounds_needs_update = False

    def update_atom_graphics(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            return

        # Set instanced sphere center position and radius
        n = len(coords)
        from numpy import empty, float32, multiply
        xyzr = empty((n, 4), float32)
        xyzr[:, :3] = coords
        xyzr[:, 3] = radii

        from .geometry import place
        p.positions = place.Places(shift_and_scale=xyzr)
        p.display_positions = display

        # Set atom colors
        p.colors = colors

        asel = self._selected_atoms
        if not asel is None:
            # If the selected atoms array has wrong length, atoms deleted, clear selection.
            p.selected_positions = asel if asel.sum() > 0 and len(asel) == n else None

    def atom_display_radii(self):
        a = self.atoms
        r = a.radii.copy()
        dm = a.draw_modes
        r[dm == self.BALL_STYLE] *= self.ball_scale
        r[dm == self.STICK_STYLE] = self.bond_radius
        return r

    def set_atom_style(self, style, atoms = None):
        if atoms is None:
            atoms = self.atoms
        atoms.draw_modes = style
        self.update_graphics()

    def color_by_element(self, atoms = None):
        if atoms is None:
            atoms = self.atoms
        atoms.colors = element_colors(atoms.element_numbers)
        self.update_graphics()

    def color_by_chain(self, atoms = None):
        if atoms is None:
            atoms = self.atoms
        atoms.colors = chain_colors(atoms.residues.chain_ids)
        self.update_graphics()

    def update_bond_graphics(self, bond_atoms, draw_mode, radii,
                             bond_colors, half_bond_coloring):
        p = self._bonds_drawing
        if p is None:
            if (draw_mode == self.SPHERE_STYLE).all():
                return
            self._bonds_drawing = p = self.new_drawing('bonds')
            from . import surface
            # Use 3 z-sections so cylinder ends match in half-bond mode.
            va, na, ta = surface.cylinder_geometry(nz = 3, caps = False)
            p.geometry = va, ta
            p.normals = na

        p.positions = bond_cylinder_placements(bond_atoms, radii, half_bond_coloring)
        p.display_positions = self.shown_bond_cylinders(bond_atoms, half_bond_coloring)
        self.set_bond_colors(p, bond_atoms, bond_colors, half_bond_coloring)

    def set_bond_colors(self, drawing, bond_atoms, bond_colors, half_bond_coloring):
        p = drawing
        if p is None:
            return

        if half_bond_coloring.any():
            bc0,bc1 = bond_atoms[0].colors, bond_atoms[1].colors
            from numpy import concatenate
            p.colors = concatenate((bc0,bc1))
        else:
            p.colors = bond_colors

    def shown_bond_cylinders(self, bond_atoms, half_bond_coloring):
        sb = bond_atoms[0].displays & bond_atoms[1].displays  # Show bond if both atoms shown
        ns = ((bond_atoms[0].draw_modes != self.SPHERE_STYLE) |
              (bond_atoms[1].draw_modes != self.SPHERE_STYLE))       # Don't show if both atoms in sphere style
        import numpy
        numpy.logical_and(sb,ns,sb)
        if half_bond_coloring.any():
            sb2 = numpy.concatenate((sb,sb))
            return sb2
        return sb

    def update_pseudobond_graphics(self, name, bond_atoms, radii,
                                   bond_colors, half_bond_coloring):
#        print ('pseudobond chain ids', bond_atoms[0].residues.chain_ids, bond_atoms[1].residues.chain_ids)
        pg = self._pseudobond_group_drawings
        if not name in pg:
            pg[name] = p = self.new_drawing(name)
            va, na, ta = pseudobond_geometry()
            p.geometry = va, ta
            p.normals = na
        else:
            p = pg[name]

        p.positions = bond_cylinder_placements(bond_atoms, radii, half_bond_coloring)
        p.display_positions = self.shown_bond_cylinders(bond_atoms, half_bond_coloring)
        self.set_bond_colors(p, bond_atoms, bond_colors, half_bond_coloring)

    def update_ribbon_graphics(self):
        from .ribbon import Ribbon, XSection
        from .geometry import place
        from numpy import concatenate, array, uint8
        polymers = self.polymers(False, False)
        xsc = [(0.5,0.1),(-0.5,0.1),(-0.5,-0.1),(0.5,-0.1)]
        # xsc = [(0.5,-0.1),(-0.5,-0.1),(-0.5,0.1),(0.5,0.1)]
        # xsc = [(0.1,0.1),(-0.1,0.1),(-0.1,-0.1),(0.1,-0.1)]
        # xsc = [( 0.5, 0.1),(0.0, 0.15),(-0.5, 0.1),(-0.6,0.0),
        #        (-0.5,-0.1),(0.0,-0.15),( 0.5,-0.1),( 0.6,0.0)]
        xs = XSection(xsc, faceted=True)
        if self._ribbon_drawing is None:
            self._ribbon_drawing = p = self.new_drawing('ribbon')
            p.display = True
        else:
            p = self._ribbon_drawing
            p.remove_all_drawings()
        for rlist in polymers:
            displays = rlist.ribbon_displays
            if displays.sum() == 0:
                continue
            coords, guides = self._get_polymer_spline(rlist)
            if len(coords) < 4:
                continue
            ribbon = Ribbon(coords, guides)
            offset = 0
            vertex_list = []
            normal_list = []
            color_list = []
            triangle_list = []
            for seg in range(ribbon.num_segments):
                show = 0
                if displays[seg]:
                    show |= XSection.FRONT
                if displays[seg + 1]:
                    show |= XSection.BACK
                if not show:
                    continue
                centers, tangents, normals = ribbon.segment(seg, self.ribbon_divisions)
                colors = [rlist[seg].ribbon_color, rlist[seg + 1].ribbon_color]
                va, na, ta, ca = xs.extrude(centers, tangents, normals, colors,
                                            show, XSection.BOTH, offset)
                offset += len(va)
                vertex_list.append(va)
                normal_list.append(na)
                triangle_list.append(ta)
                color_list.append(ca)
            rp = p.new_drawing(rlist.strs[seg])
            rp.display = True
            rp.vertices = concatenate(vertex_list)
            rp.normals = concatenate(normal_list)
            rp.triangles = concatenate(triangle_list)
            #rp.color = array((160,160,160,255), uint8)
            rp.vertex_colors = concatenate(color_list)

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

    def hide_chain(self, cid):
        a = self.atoms
        a.displays &= self.chain_atom_mask(cid, invert = True)
        self.update_graphics()

    def show_chain(self, cid):
        a = self.atoms
        a.displays |= self.chain_atom_mask(cid)
        self.update_graphics()

    def chain_atom_mask(self, cid, invert = False):
        a = self.atoms
        cids = a.residues.chain_ids
        from numpy import array, bool, logical_not
        d = array(tuple((cid != c) for c in cids), bool)
        if invert:
            logical_not(d,d)
        return d

    def first_intercept(self, mxyz1, mxyz2, exclude = None):
        # TODO check intercept of bounding box as optimization
        p = self._atoms_drawing
        if p is None:
            return None
        xyzr = p.positions.shift_and_scale_array()
        xyz = xyzr[:,:3]
        r = xyzr[:,3]

        f = fa = None
        from . import graphics
        for tf in self.positions:
            cxyz1, cxyz2 = tf.inverse() * (mxyz1, mxyz2)
            # Check for atom sphere intercept
            fmin, anum = graphics.closest_sphere_intercept(xyz, r, cxyz1, cxyz2)
            if not fmin is None and (f is None or fmin < f):
                f, fa = fmin, anum

        # Create pick object
        if fa is None:
            s = None
        else:
            s = PickedAtom(self, fa, f)

        return s

    def bounds(self, positions = True):
        # TODO: Cache bounds
        ab = self.atom_bounds()
        rb = self.ribbon_bounds()
        from .geometry import bounds
        b = bounds.union_bounds((ab, rb))
        if positions:
            b = bounds.copies_bounding_box(b, self.positions)
        return b

    def atom_bounds(self):
        if not self._atom_bounds_needs_update:
            return self._atom_bounds
        a = self.atoms
        xyz = a.coords[a.displays]
        from .geometry import bounds
        b = bounds.point_bounds(xyz)
        self._atom_bounds = b
        self._atom_bounds_needs_update = False
        return b

    def ribbon_bounds(self):
        rd = self._ribbon_drawing
        if rd is None or not rd.display:
            return None
        return rd.bounds()

    def atom_index_description(self, a):
        atoms = self.atoms
        r = atoms.residues
        id = '.'.join(str(i) for i in self.id)
        d = '%s %s.%s %s %d %s' % (self.name, id, r.chain_ids[a], r.names[a], r.numbers[a], atoms.names[a])
        return d

    def select_atom(self, a, toggle = False):
        asel = self._selected_atoms
        if asel is None:
            na = self.num_atoms
            from numpy import zeros, bool
            asel = self._selected_atoms = zeros(na, bool)
        asel[a] = (not asel[a]) if toggle else True
        self._selection_changed()

    def selected_items(self, itype):
        if itype == 'atoms':
            asel = self._selected_atoms
            if not asel is None and asel.sum() > 0:
                atoms = self.atoms
                sa = atoms.filter(asel)
                return [sa]
        return []

    def any_part_selected(self):
        asel = self._selected_atoms
        return not asel is None and asel.sum() > 0

    def clear_selection(self):
        asel = self._selected_atoms
        if not asel is None and asel.sum() > 0:
            asel[:] = False
        self._selection_changed()

    def _selection_changed(self, promotion = False):
        if not promotion:
            self._selection_promotion_history = []
        self.update_graphics()

    def promote_selection(self):
        asel = self._selected_atoms
        if asel is None:
            return
        n = asel.sum()
        if n == 0 or n == len(asel):
            return
        self._selection_promotion_history.append(asel.copy())

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
        asel[:] = psel
        self._selection_changed(promotion = True)

    def demote_selection(self):
        pt = self._selection_promotion_history
        if len(pt) > 0:
            self._selected_atoms[:] = pt.pop()
            self._selection_changed(promotion = True)

    def clear_selection_promotion_history(self):
        self._selection_promotion_history = []

def selected_atoms(session):
    from .molecule import Atoms
    atoms = Atoms()
    for m in session.models.list():
        if isinstance(m, AtomicStructure):
            for matoms in m.selected_items('atoms'):
                atoms = atoms | matoms
    return atoms

# -----------------------------------------------------------------------------
#
from .graphics import Pick
class PickedAtom(Pick):
  def __init__(self, mol, a, distance):
    Pick.__init__(self, distance)
    self.molecule = mol
    self.atom = a
  def description(self):
    m, a = self.molecule, self.atom
    if a is None:
      return m.name
    return m.atom_index_description(a)
  def drawing(self):
    return self.molecule
  def select(self, toggle = False):
    m = self.molecule
    m.select_atom(self.atom, toggle)

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking prototype cylinder to bond location.
#
def bond_cylinder_placements(bond_atoms, radius, half_bond):

  # TODO: Allow per-bound variation in half-bond mode.
  half_bond = half_bond.any()

  n = len(bond_atoms[0])
  from numpy import empty, float32, transpose, sqrt, array
  nc = 2*n if half_bond else n
  p = empty((nc,4,4), float32)
  
  p[:,3,:] = (0,0,0,1)
  axyz0, axyz1 = bond_atoms[0].coords, bond_atoms[1].coords
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
  from .geometry import place
  pl = place.Places(opengl_array = pt)
  return pl

# -----------------------------------------------------------------------------
#
def pseudobond_geometry(segments = 9):
    from . import surface
    return surface.dashed_cylinder_geometry(segments)

# -----------------------------------------------------------------------------
#
element_rgba_256 = None
def element_colors(element_numbers):
    global element_rgba_256
    if element_rgba_256 is None:
        from numpy import empty, uint8
        element_rgba_256 = ec = empty((256, 4), uint8)
        ec[:,:3] = 180
        ec[:,3] = 255
        # jmol element colors
        colors = (
            (1,	(255,255,255)),	 # H
            (2,	(217,255,255)),	 # He
            (3,	(204,128,255)),	 # Li
            (4,	(194,255,0)),	 # Be  
            (5,	(255,181,181)),	 # B
            (6,	(144,144,144)),	 # C
            (7,	(48,80,248)),	 # N  
            (8,	(255,13,13)),	 # O  
            (9, (144,224,80)),	 # F 
            (10, (179,227,245)), # Ne
            (11, (171,92,242)),	 # Na 
            (12, (138,255,0)),	 # Mg  
            (13, (191,166,166)), # Al
            (14, (240,200,160)), # Si
            (15, (255,128,0)),	 # P  
            (16, (255,255,48)),	 # S 
            (17, (31,240,31)),	 # Cl  
            (18, (128,209,227)), # Ar
            (19, (143,64,212)),	 # K 
            (20, (61,255,0)),	 # Ca   
            (21, (230,230,230)), # Sc
            (22, (191,194,199)), # Ti
            (23, (166,166,171)), # V
            (24, (138,153,199)), # Cr
            (25, (156,122,199)), # Mn
            (26, (224,102,51)),	 # Fe 
            (27, (240,144,160)), # Co
            (28, (80,208,80)),	 # Ni  
            (29, (200,128,51)),	 # Cu 
            (30, (125,128,176)), # Zn
            (31, (194,143,143)), # Ga
            (32, (102,143,143)), # Ge
            (33, (189,128,227)), # As
            (34, (255,161,0)),	 # Se  
            (35, (166,41,41)),	 # Br  
            (36, (92,184,209)),	 # Kr 
            (37, (112,46,176)),	 # Rb 
            (38, (0,255,0)),	 # Sr    
            (39, (148,255,255)), # Y
            (40, (148,224,224)), # Zr
            (41, (115,194,201)), # Nb
            (42, (84,181,181)),	 # Mo 
            (43, (59,158,158)),	 # Tc 
            (44, (36,143,143)),	 # Ru 
            (45, (10,125,140)),	 # Rh 
            (46, (0,105,133)),	 # Pd  
            (47, (192,192,192)), # Ag
            (48, (255,217,143)), # Cd
            (49, (166,117,115)), # In
            (50, (102,128,128)), # Sn
            (51, (158,99,181)),	 # Sb 
            (52, (212,122,0)),	 # Te  
            (53, (148,0,148)),	 # I  
            (54, (66,158,176)),	 # Xe 
            (55, (87,23,143)),	 # Cs  
            (56, (0,201,0)),	 # Ba    
            (57, (112,212,255)), # La
            (58, (255,255,199)), # Ce
            (59, (217,255,199)), # Pr
            (60, (199,255,199)), # Nd
            (61, (163,255,199)), # Pm
            (62, (143,255,199)), # Sm
            (63, (97,255,199)),	 # Eu 
            (64, (69,255,199)),	 # Gd 
            (65, (48,255,199)),	 # Tb 
            (66, (31,255,199)),	 # Dy 
            (67, (0,255,156)),	 # Ho  
            (68, (0,230,117)),	 # Er  
            (69, (0,212,82)),	 # Tm   
            (70, (0,191,56)),	 # Yb   
            (71, (0,171,36)),	 # Lu   
            (72, (77,194,255)),	 # Hf 
            (73, (77,166,255)),	 # Ta 
            (74, (33,148,214)),	 # W 
            (75, (38,125,171)),	 # Re 
            (76, (38,102,150)),	 # Os 
            (77, (23,84,135)),	 # Ir  
            (78, (208,208,224)), # Pt
            (79, (255,209,35)),	 # Au 
            (80, (184,184,208)), # Hg
            (81, (166,84,77)),	 # Tl  
            (82, (87,89,97)),	 # Pb   
            (83, (158,79,181)),	 # Bi 
            (84, (171,92,0)),	 # Po   
            (85, (117,79,69)),	 # At  
            (86, (66,130,150)),	 # Rn 
            (87, (66,0,102)),	 # Fr   
            (88, (0,125,0)),	 # Ra    
            (89, (112,171,250)), # Ac
            (90, (0,186,255)),	 # Th  
            (91, (0,161,255)),	 # Pa  
            (92, (0,143,255)),	 # U  
            (93, (0,128,255)),	 # Np  
            (94, (0,107,255)),	 # Pu  
            (95, (84,92,242)),	 # Am  
            (96, (120,92,227)),	 # Cm 
            (97, (138,79,227)),	 # Bk 
            (98, (161,54,212)),	 # Cf 
            (99, (179,31,212)),	 # Es 
            (100, (179,31,186)), # Fm 
            (101, (179,13,166)), # Md 
            (102, (189,13,135)), # No 
            (103, (199,0,102)),	 # Lr  
            (104, (204,0,89)),	 # Rf   
            (105, (209,0,79)),	 # Db   
            (106, (217,0,69)),	 # Sg   
            (107, (224,0,56)),	 # Bh   
            (108, (230,0,46)),	 # Hs   
            (109, (235,0,38)),	 # Mt   
        )
        for e, rgb in colors:
            ec[e,:3] = rgb

    colors = element_rgba_256[element_numbers]
    return colors

# -----------------------------------------------------------------------------
#
rgba_256 = None
def chain_colors(cids):

    global rgba_256
    if rgba_256 is None:
        rgba_256 = {
          'a':(123,104,238,255),
          'b':(240,128,128,255),
          'c':(143,188,143,255),
          'd':(222,184,135,255),
          'e':(255,127,80,255),
          'f':(128,128,128,255),
          'g':(107,142,35,255),
          'h':(100,100,100,255),
          'i':(255,255,0,255),
          'j':(55,19,112,255),
          'k':(255,255,150,255),
          'l':(202,62,94,255),
          'm':(205,145,63,255),
          'n':(12,75,100,255),
          'o':(255,0,0,255),
          'p':(175,155,50,255),
          'q':(105,205,48,255),
          'r':(37,70,25,255),
          's':(121,33,135,255),
          't':(83,140,208,255),
          'u':(0,154,37,255),
          'v':(178,220,205,255),
          'w':(255,152,213,255),
          'x':(200,90,174,255),
          'y':(175,200,74,255),
          'z':(63,25,12,255),
          '1': (87, 87, 87,255),
          '2': (173, 35, 35,255),
          '3': (42, 75, 215,255),
          '4': (29, 105, 20,255),
          '5': (129, 74, 25,255),
          '6': (129, 38, 192,255),
          '7': (160, 160, 160,255),
          '8': (129, 197, 122,255),
          '9': (157, 175, 255,255),
          '0': (41, 208, 208,255),
        }

    for cid in set(cids):
        c = str(cid).lower()
        if not c in rgba_256:
            from random import randint, seed
            seed(c)
            rgba_256[c] = (randint(128,255),randint(128,255),randint(128,255),255)

    from numpy import array, uint8
    c = array(tuple(rgba_256[cid.lower()] for cid in cids), uint8)
    return c

# -----------------------------------------------------------------------------
#
def chain_rgba(cid):
    return tuple(float(c/255.0) for c in chain_colors([cid])[0])

# -----------------------------------------------------------------------------
#
def chain_rgba8(cid):
    return chain_colors([cid])[0]

# -----------------------------------------------------------------------------
#
from . import cli
from . import atomspec
_ccolor_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)],
                           synopsis='color by chain')
def ccolor_command(session, atoms = None):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.color_by_chain()
    else:
        asr = atoms.evaluate(session)
        a = asr.atoms
        a.colors = chain_colors(a.residues.chain_ids)
        for m in asr.models:
            if isinstance(m, AtomicStructure):
                m.update_graphics()

# -----------------------------------------------------------------------------
#
from . import cli
from . import atomspec
_celement_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)],
                             synopsis='color by element')
def celement_command(session, atoms = None):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.color_by_element()
    else:
        asr = atoms.evaluate(session)
        a = asr.atoms
        a.colors = element_colors(a.element_numbers)
        update_model_graphics(asr.models)

def update_model_graphics(models):
    for m in models:
        if isinstance(m, AtomicStructure):
            m.update_graphics()

# -----------------------------------------------------------------------------
#
from . import cli
_style_desc = cli.CmdDesc(required = [('atom_style', cli.EnumOf(('sphere', 'ball', 'stick')))],
                          optional=[("atoms", atomspec.AtomSpecArg)],
                          synopsis='change atom depiction')
def style_command(session, atom_style, atoms = None):
    s = {'sphere':AtomicStructure.SPHERE_STYLE,
         'ball':AtomicStructure.BALL_STYLE,
         'stick':AtomicStructure.STICK_STYLE,
         }[atom_style.lower()]
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.set_atom_style(s)
    else:
        asr = atoms.evaluate(session)
        asr.atoms.draw_modes = s
        update_model_graphics(asr.models)

# -----------------------------------------------------------------------------
#
from . import cli
_hide_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)],
                         synopsis='hide atoms')
def hide_command(session, atoms = None):
    show_atoms(False, atoms, session)

# -----------------------------------------------------------------------------
#
from . import cli
_show_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)],
                         synopsis='show atoms')
def show_command(session, atoms = None):
    show_atoms(True, atoms, session)

# -----------------------------------------------------------------------------
#
def show_atoms(show, atoms, session):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.atoms.displays = show
                m.update_graphics()
    else:
        asr = atoms.evaluate(session)
        asr.atoms.displays = show
        update_model_graphics(asr.models)

# -----------------------------------------------------------------------------
# Wrap an AtomBlob and have a molecules attribute.
#
from . import cli
class AtomsArg(cli.Annotation):
    """Annotation for atoms"""
    name = "atoms"

    @staticmethod
    def parse(text, session):
        from . import atomspec
        aspec, text, rest = atomspec.AtomSpecArg.parse(text, session)
        atoms = aspec.evaluate(session).atoms
        return atoms, text, rest

# -----------------------------------------------------------------------------
#
def register_molecule_commands():
    from . import cli
    cli.register('style', _style_desc, style_command)
    cli.register('ccolor', _ccolor_desc, ccolor_command)
    cli.register('celement', _celement_desc, celement_command)
    cli.register('hide', _hide_desc, hide_command)
    cli.register('show', _show_desc, show_command)
