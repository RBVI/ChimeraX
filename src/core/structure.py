# vi: set expandtab shiftwidth=4 softtabstop=4:
from . import io
from . import models

CATEGORY = io.STRUCTURE


class StructureModel(models.Model):
    """Commom base class for atomic structures"""

    STRUCTURE_STATE_VERSION = 0

    # Atom display styles, Atom.draw_modes
    SPHERE_STYLE = 1
    BALL_STYLE = 2
    STICK_STYLE = 3

    def __init__(self, name):

        models.Model.__init__(self, name)
        self.ball_scale = 0.3		# Scales sphere radius in ball and stick style
        self.bond_radius = 0.2
        self._atoms_drawing = None
        self._bonds_drawing = None

    def take_snapshot(self, session, flags):
        data = {}
        return [self.STRUCTURE_STATE_VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.STRUCTURE_STATE_VERSION or len(data) > 0:
            raise RuntimeError("Unexpected version or data")

    def reset_state(self):
        pass

    def initialize_graphical_attributes(self):
        m = self.mol_blob
        a = m.atoms
        a.draw_modes = self.SPHERE_STYLE
        a.colors = element_colors(a.element_numbers)
        b = m.bonds
        b.radii = self.bond_radius

    def make_drawing(self):

        self.initialize_graphical_attributes()

        m = self.mol_blob
        a = m.atoms
        coords = a.coords
        radii = self.atom_display_radii()
        colors = a.colors
        display = a.displays
        b = m.bonds
        bradii = b.radii
        halfbond = b.halfbonds
        bcolors = b.colors

        # Create graphics
        self.create_atom_spheres(coords, radii, colors, display)
        self.update_bond_graphics(b.atoms, a.draw_modes, bradii, bcolors, halfbond)

    def create_atom_spheres(self, coords, radii, colors, display):
        p = self._atoms_drawing
        if p is None:
            self._atoms_drawing = p = self.new_drawing('atoms')

        n = len(coords)
        triangles_per_sphere = 320 if n < 30000 else 80 if n < 120000 else 20

        # Set instanced sphere triangulation
        from . import surface
        va, na, ta = surface.sphere_geometry(triangles_per_sphere)
        p.geometry = va, ta
        p.normals = na

        self.update_atom_graphics(coords, radii, colors, display)

    def update_graphics(self):
        m = self.mol_blob
        a = m.atoms
        b = m.bonds
        self.update_atom_graphics(a.coords, self.atom_display_radii(), a.colors, a.displays)
        self.update_bond_graphics(b.atoms, a.draw_modes, b.radii, b.colors, b.halfbonds)

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

    def atom_display_radii(self):
        m = self.mol_blob
        a = m.atoms
        r = a.radii.copy()
        dm = a.draw_modes
        r[dm == self.BALL_STYLE] *= self.ball_scale
        r[dm == self.STICK_STYLE] = self.bond_radius
        return r

    def set_atom_style(self, style):
        self.mol_blob.atoms.draw_modes = style
        self.update_graphics()

    def color_by_element(self):
        a = self.mol_blob.atoms
        a.colors = element_colors(a.element_numbers)
        self.update_graphics()

    def color_by_chain(self):
        a = self.mol_blob.atoms
        a.colors = chain_colors(a.residues.chain_ids)
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
        self.set_bond_colors(bond_atoms, bond_colors, half_bond_coloring)

    def set_bond_colors(self, bond_atoms, bond_colors, half_bond_coloring):
        p = self._bonds_drawing
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

    def hide_chain(self, cid):
        a = self.mol_blob.atoms
        a.displays &= self.chain_atom_mask(cid, invert = True)
        self.update_graphics()

    def show_chain(self, cid):
        a = self.mol_blob.atoms
        a.displays |= self.chain_atom_mask(cid)
        self.update_graphics()

    def chain_atom_mask(self, cid, invert = False):
        a = self.mol_blob.atoms
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
            return None, None
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

        # Create selection object
        if fa is None:
            s = None
        else:
            ai = self.mol_blob.atoms.displays.nonzero()[0]
            fa = ai[fa]
            s = Picked_Atom(self, fa)

        return f, s

    def bounds(self, positions = True):
        # TODO: Cache bounds
        a = self.mol_blob.atoms
        xyz = a.coords[a.displays]
        if len(xyz) == 0:
            return None
        xyz_min, xyz_max = xyz.min(axis = 0), xyz.max(axis = 0)
        from .geometry import bounds
        b = bounds.Bounds(xyz_min, xyz_max)
        if positions:
            b = bounds.copies_bounding_box(b, self.positions)
        return b

    def atom_index_description(self, a):
        atoms = self.mol_blob.atoms
        r = atoms.residues
        id = '.'.join(str(i) for i in self.id)
        d = '%s %s.%s %s %d %s' % (self.name, id, r.chain_ids[a], r.names[a], r.numbers[a], atoms.names[a])
        return d


# -----------------------------------------------------------------------------
#
from .graphics import Pick
class Picked_Atom(Pick):
  def __init__(self, mol, a):
    self.molecule = mol
    self.atom = a
  def description(self):
    m, a = self.molecule, self.atom
    if a is None:
      return m.name
    return m.atom_index_description(a)
  def drawing(self):
    return self.molecule

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
element_rgba_256 = None
def element_colors(element_numbers):
    global element_rgba_256
    if element_rgba_256 is None:
        from numpy import empty, uint8
        element_rgba_256 = ec = empty((256, 4), uint8)
        ec[:, :3] = 180
        ec[:, 3] = 255
        ec[6, :] = (255, 255, 255, 255)     # H
        ec[6, :] = (144, 144, 144, 255)     # C
        ec[7, :] = (48, 80, 248, 255)       # N
        ec[8, :] = (255, 13, 13, 255)       # O
        ec[15, :] = (255, 128, 0, 255)      # P
        ec[16, :] = (255, 255, 48, 255)     # S
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
_ccolor_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)])
def ccolor_command(session, atoms = None):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, StructureModel):
                m.color_by_chain()
    else:
        asr = atoms.evaluate(session)
        a = asr.atoms
        a.colors = chain_colors(a.residues.chain_ids)
        for m in asr.models:
            if isinstance(m, StructureModel):
                m.update_graphics()

# -----------------------------------------------------------------------------
#
from . import cli
from . import atomspec
_celement_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)])
def celement_command(session, atoms = None):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, StructureModel):
                m.color_by_element()
    else:
        asr = atoms.evaluate(session)
        a = asr.atoms
        a.colors = element_colors(a.element_numbers)
        update_model_graphics(asr.models)

def update_model_graphics(models):
    for m in models:
        if isinstance(m, StructureModel):
            m.update_graphics()

# -----------------------------------------------------------------------------
#
from . import cli
_style_desc = cli.CmdDesc(required = [('atom_style', cli.EnumOf(('sphere', 'ball', 'stick')))],
                          optional=[("atoms", atomspec.AtomSpecArg)])
def style_command(session, atom_style, atoms = None):
    s = {'sphere':StructureModel.SPHERE_STYLE,
         'ball':StructureModel.BALL_STYLE,
         'stick':StructureModel.STICK_STYLE,
         }[atom_style.lower()]
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, StructureModel):
                m.set_atom_style(s)
    else:
        asr = atoms.evaluate(session)
        asr.atoms.draw_modes = s
        update_model_graphics(asr.models)

# -----------------------------------------------------------------------------
#
from . import cli
_hide_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)])
def hide_command(session, atoms = None):
    show_atoms(False, atoms, session)

# -----------------------------------------------------------------------------
#
from . import cli
_show_desc = cli.CmdDesc(optional=[("atoms", atomspec.AtomSpecArg)])
def show_command(session, atoms = None):
    show_atoms(True, atoms, session)

# -----------------------------------------------------------------------------
#
def show_atoms(show, atoms, session):
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, StructureModel):
                m.mol_blob.atoms.displays = show
                m.update_graphics()
    else:
        asr = atoms.evaluate(session)
        asr.atoms.displays = show
        update_model_graphics(asr.models)

# -----------------------------------------------------------------------------
#
def register_molecule_commands():
    from . import cli
    cli.register('style', _style_desc, style_command)
    cli.register('ccolor', _ccolor_desc, ccolor_command)
    cli.register('celement', _celement_desc, celement_command)
    cli.register('hide', _hide_desc, hide_command)
    cli.register('show', _show_desc, show_command)
