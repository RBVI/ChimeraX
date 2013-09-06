from ..surface import Surface
class Molecule(Surface):

  def __init__(self, path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names):

    from os.path import basename
    name = basename(path)
    Surface.__init__(self, name)

    self.path = path
    self.xyz = xyz
    self.element_nums = element_nums
    from .. import _image3d
    self.radii = _image3d.element_radii(element_nums)
    #
    # Chain ids, residue names and atom names are Numpy string arrays.
    # The strings have maximum fixed length.  The strings are python byte arrays,
    # not Python unicode strings.  Comparison to a Python unicode string will
    # always give false.  Need to compare to a byte array.
    # For example, atom_names == 'CA'.encode('utf-8')
    #
    self.chain_ids = chain_ids
    self.residue_nums = res_nums
    self.residue_names = res_names
    self.atom_names = atom_names
    self.atom_colors = None

    self.bonds = None                   # N by 2 array of atom indices
    self.bond_radius = 0.2
    self.bond_radii = None

    # Graphics settings
    self.atom_shown_count = n = len(xyz)
    from numpy import ones, bool
    self.atom_shown = ones((n,), bool)
    self.atoms_surface_piece = None
    self.atom_style = 'sphere'          # sphere, stick or ballstick
    self.ball_scale = 0.3               # Atom radius scale factor in ball and stick.
    self.bonds_surface_piece = None
    self.bond_color = (150,150,150,255)
    self.half_bond_coloring = True
    self.show_ribbons = False
    self.ribbon_radius = 1.0
    self.ribbon_subdivisions = (5,10)   # per-residue along length, and circumference
    self.ribbon_surface_pieces = {}     # Map chain id to surface piece

    self.color = (180,180,180,255)      # RGBA 0-255 integer values
    self.color_mode = 'by chain'        # by chain, by element, or single

    # Graphics objects
    self.triangles_per_sphere = 20      # Current level of detail
    self.need_graphics_update = True    # Update is done before drawing
    self.atom_surface_piece = None

  def draw(self, viewer, draw_pass):

    self.update_graphics(viewer)

    Surface.draw(self, viewer, draw_pass)

  def update_graphics(self, viewer):

    if not self.need_graphics_update:
      return
    self.need_graphics_update = False

    self.update_atom_graphics(viewer)
    self.update_bond_graphics(viewer)
    self.update_ribbon_graphics()

  def update_atom_graphics(self, viewer):

    self.create_atom_spheres()
    self.set_atom_colors()

  def create_atom_spheres(self):

    p = self.atoms_surface_piece
    if self.atom_shown_count == 0:
      if p:
        self.atoms_surface_piece = None
        self.removePiece(p)
      return

    if p is None:
      self.atoms_surface_piece = p = self.newPiece()

    ntri = self.triangles_per_sphere
    va, na, ta = sphere_geometry(ntri)
    p.geometry = va, ta
    p.normals = na

    xyz = self.xyz
    n = len(xyz)
    from numpy import empty, float32
    xyzr = empty((n,4), float32)
    xyzr[:,:3] = xyz

    astyle = self.atom_style
    if astyle == 'sphere':
      r = self.radii
    elif astyle == 'ballstick':
      r = self.radii * self.ball_scale
    elif astyle == 'stick':
      r = self.bond_radius
    xyzr[:,3] = r

    sas = self.shown_atom_array_values(xyzr)
    p.shift_and_scale = sas

  def shown_atom_array_values(self, a):
    if self.all_atoms_shown():
      return a
    else:
      return a[self.atom_shown]

  def all_atoms_shown(self):
    return self.atom_count() == self.atom_shown_count

  def set_atom_colors(self):

    p = self.atoms_surface_piece
    if p is None:
      return

    p.instance_colors = self.shown_atom_array_values(self.atom_rgba())

  def update_bond_graphics(self, viewer):

    bonds = self.shown_bonds()
    self.create_bond_cylinders(bonds)
    self.set_bond_colors(bonds)

  # Both atoms need to be shown to show bond.
  def shown_bonds(self):

    b = self.bonds
    if b is None or self.all_atoms_shown():
      return b
    sa = self.atom_shown
    sb = sa[b[:,0]] & sa[b[:,1]]
    return b[sb]

  def create_bond_cylinders(self, bonds):

    p = self.bonds_surface_piece
    if self.atom_shown_count == 0 or bonds is None or self.atom_style == 'sphere':
      if p:
        self.bonds_surface_piece = None
        self.removePiece(p)
      return

    if p is None:
      self.bonds_surface_piece = p = self.newPiece()

    va, na, ta = cylinder_geometry()
    p.geometry = va, ta
    p.normals = na

    r = self.bond_radius if self.bond_radii is None else self.bond_radii
    p.copies44 = bond_cylinder_placements(bonds, self.xyz, r, self.half_bond_coloring)

  def set_bond_colors(self, bonds):

    p = self.bonds_surface_piece
    if p is None:
      return

    p.color = tuple(c/255.0 for c in self.bond_color)

    if self.half_bond_coloring:
      acolors = self.atom_rgba()
      bc0,bc1 = acolors[bonds[:,0],:], acolors[bonds[:,1],:]
      from numpy import concatenate
      p.instance_colors = concatenate((bc0,bc1))
    else:
      p.instance_colors = None

  def set_color_mode(self, mode):
    if mode == self.color_mode:
      return
    self.color_mode = mode
    self.need_graphics_update = True
    self.redraw_needed = True

  def atom_rgba(self):

    if not self.atom_colors is None:
      return self.atom_colors

    cm = self.color_mode
    if cm == 'by chain':
      colors = chain_colors(self.chain_ids)
    elif cm == 'by element':
      colors = element_colors(self.element_nums)
    else:
      from numpy import empty, uint8
      colors = empty((self.atom_count(),4),uint8)
      colors[:,:] = self.color
    return colors

  def update_ribbon_graphics(self):

    import sys
    rsp = self.ribbon_surface_pieces
    if not self.show_ribbons:
      self.removePieces(rsp.values())
      rsp.clear()
      return

    cids = set(self.chain_ids)
    from .colors import rgba_256
    from numpy import uint8
    for cid in cids:
      s = self.atom_subset('CA', cid)
      if len(s) <= 1:
        s = self.atom_subset("C5'", cid)
        if len(s) <= 1:
          continue
      path = self.xyz[s]
    
      sd, cd = self.ribbon_subdivisions
      from ..geometry import tube
      va,na,ta,ca = tube.tube_through_points(path, radius = self.ribbon_radius,
                                             color = rgba_256[cid[0]],
                                             segment_subdivisions = sd,
                                             circle_subdivisions = cd)
      if cid in rsp:
        p = rsp[cid]
      else:
        rsp[cid] = p = self.newPiece()
      p.geometry = va, ta
      p.normals = na
      p.vertex_colors = ca

  def ribbon_residues(self):
    cres = []
    cids = set(self.chain_ids)
    for cid in cids:
      s = self.atom_subset('CA', cid)
      if len(s) <= 1:
        s = self.atom_subset("C5'", cid)
        if len(s) <= 1:
          continue
      cres.append((cid, self.residue_nums[s]))
    return cres

  # Ligand atoms are everything except ribbon and HOH residues
  def show_ligand_atoms(self):
    cres = self.ribbon_residues()
    cr = set()
    for cid, rnums in cres:
      for rnum in rnums:
        cr.add((cid, rnum))
    n = self.atom_count()
    cid = self.chain_ids
    rnum = self.residue_nums
    rname = self.residue_names
    sa = self.atom_shown
    for i in range(n):
      sa[i] |= (((cid[i], rnum[i]) not in cr) and rname[i] != b'HOH')
    self.atom_shown_count = sa.sum()
    self.need_graphics_update = True
    self.redraw_needed = True

  def set_ribbon_radius(self, r):

    if r != self.ribbon_radius:
      self.ribbon_radius = r
      self.need_graphics_update = True
      self.redraw_needed = True

  def show_solvent(self):
    self.atom_shown |= (self.residue_names == b'HOH')
    self.atom_shown_count = self.atom_shown.sum()
    self.need_graphics_update = True
    self.redraw_needed = True

  def hide_solvent(self):
    self.atom_shown &= (self.residue_names != b'HOH')
    self.atom_shown_count = self.atom_shown.sum()
    self.need_graphics_update = True
    self.redraw_needed = True
    
  def all_atoms(self):

    from numpy import arange
    return arange(self.atom_count())

  def atom_subset(self, name = None, chain_id = None, residue_range = None,
                   restrict_to_atoms = None):

    anames = self.atom_names
    na = self.atom_count()
    from numpy import zeros, uint8, logical_or, logical_and
    nimask = zeros((na,), uint8)
    if name is None:
      nimask[:] = 1
    else:
      logical_or(nimask, (anames == name.encode('utf-8')), nimask)

    if not chain_id is None:
      if isinstance(chain_id, (list,tuple)):
        chain_ids = chain_id
        cmask = self.chain_atom_mask(chain_ids[0])
        for cid in chain_ids[1:]:
          logical_or(cmask, self.chain_atoms(cid), cmask)
      else:
        cmask = self.chain_atom_mask(chain_id)
      logical_and(nimask, cmask, nimask)

    if not residue_range is None:
      r1, r2 = residue_range
      if not r1 is None:
        logical_and(nimask, (self.residue_nums >= r1), nimask)
      if not r2 is None:
        logical_and(nimask, (self.residue_nums <= r2), nimask)

    if not restrict_to_atoms is None:
      ramask = zeros((na,), uint8)
      ramask[restrict_to_atoms] = 1
      logical_and(nimask, ramask, nimask)

    i = nimask.nonzero()[0]
    return i

  def chain_atom_mask(self, chain_id):
    cid = chain_id.encode('utf-8') if isinstance(chain_id, str) else chain_id
    cmask = (self.chain_ids == cid)
    return cmask

  def update_level_of_detail(self, viewer):

    # TODO: adjust ribbon level of detail
    n = viewer.atoms_shown
    ntmax = 10000000
    ntri = 320 if 320*n <= ntmax else 80 if 80*n <= ntmax else 20
    if ntri != self.triangles_per_sphere:
      self.triangles_per_sphere = ntri
      if self.atom_shown_count > 0:
        self.need_graphics_update = True

  def show_all_atoms(self):
    n = self.atom_count()
    if self.atom_shown_count < n:
      self.atom_shown[:] = True
      self.atom_shown_count = n
      self.need_graphics_update = True
      self.redraw_needed = True

  def hide_all_atoms(self):
    if self.atom_shown_count > 0:
      self.atom_shown[:] = False
      self.atom_shown_count = 0
      self.need_graphics_update = True
      self.redraw_needed = True

  def set_ribbon_display(self, display):
    if display != self.show_ribbons:
      self.show_ribbons = display
      self.need_graphics_update = True
      self.redraw_needed = True

  def set_atom_style(self, style):
    if style != self.atom_style:
      self.atom_style = style
      self.need_graphics_update = True
      self.redraw_needed = True

  def first_intercept(self, mxyz1, mxyz2):
    # TODO check intercept of bounding box as optimization
    # TODO using wrong radius for atoms in stick and ball and stick
    xyz = self.shown_atom_array_values(self.xyz)
    r = self.shown_atom_array_values(self.radii)
    from .. import _image3d
    if self.copies:
      intercepts = []
      for tf in self.copies:
        cxyz1, cxyz2 = tf.inverse() * (mxyz1, mxyz2)
        fmin, anum = _image3d.closest_sphere_intercept(xyz, r, cxyz1, cxyz2)
        if not fmin is None:
          intercepts.append((fmin,anum))
      fmin, anum = min(intercepts) if intercepts else (None,None)
    else:
      fmin, anum = _image3d.closest_sphere_intercept(xyz, r, mxyz1, mxyz2)
    if not anum is None and not self.all_atoms_shown():
      anum = self.atom_shown.nonzero()[0][anum]
    return fmin, Atom_Selection(self, anum)

  def atom_count(self):
    return len(self.xyz)

  def atoms_shown(self):
    nc = max(1, len(self.copies))
    na = self.atom_shown_count
    return na * nc

  def chain_atoms(self, chain_id):
    from numpy import fromstring
    cid = fromstring(chain_id, self.chain_ids.dtype)
    atoms = (self.chain_ids == cid).nonzero()[0]
    return atoms

  def bounds(self):
    # TODO: bounds should only include displayed atoms.
    xyz = self.xyz
    if len(xyz) == 0:
      return None
    return xyz.min(axis = 0), xyz.max(axis = 0)

# Only produces 20, 80, 320, ... (multiples of 4) triangle count.
def sphere_geometry(ntri):
  from ..geometry import icosahedron
  va, ta = icosahedron.icosahedron_geometry()
  from numpy import int32, sqrt
  ta = ta.astype(int32)
  from .. import _image3d
  while 4*len(ta) <= ntri:
    va, ta = _image3d.subdivide_triangles(va, ta)
  vn = sqrt((va*va).sum(axis = 1))
  for a in (0,1,2):
    va[:,a] /= vn
  return va, va, ta

def cylinder_geometry():
  from ..geometry import tube
  return tube.cylinder_geometry(radius = 1, height = 1,
                                nz = 2, nc = 10, caps = False)

def chain_colors(cids):

  # Use first character of multi-character chain ids.
  from numpy import uint8
  cids = cids.view(uint8)[::cids.itemsize]
  from .colors import rgba_256
  return rgba_256[cids]

def chain_colors_old(cids):

  from .colors import chain_colors as c, default_color
  rgb = tuple(c.get(cid.lower(), default_color) for cid in cids)
  return rgb

def element_colors(elnums):

  from .colors import element_rgba_256
  return element_rgba_256[elnums]

# -----------------------------------------------------------------------------
# Return 4x4 matrices taking prototype cylinder to bond location.
#
def bond_cylinder_placements(bonds, xyz, radius, half_bond):

  n = len(bonds)
  from numpy import empty, float32, transpose, sqrt, array
  nc = 2*n if half_bond else n
  p = empty((nc,4,4), float32)
  
  p[:,3,:] = (0,0,0,1)
  axyz0, axyz1 = xyz[bonds[:,0],:], xyz[bonds[:,1],:]
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
  return pt

# -----------------------------------------------------------------------------
#
class Atom_Set:
  def __init__(self):
    self.molatoms = []      # Pairs (molecule, atom index array)
  def add_molecules(self, molecules):
    for m in molecules:
      self.molatoms.append((m, m.all_atoms()))
  def add_atoms(self, mol, atoms):
    self.molatoms.append((mol, atoms))
  def molecules(self):
    return list(set(m for m,a in self.molatoms))
  def count(self):
    return sum(len(a) for m,a in self.molatoms)
  def coordinates(self):
    coords = []
    for m,a in self.molatoms:
        xyz = m.xyz[a]
        m.place.move(xyz)
        coords.append(xyz)
    import numpy
    if len(coords) == 0:
        a = numpy.zeros((0,3), numpy.float32)
    elif len(coords) == 1:
        a = coords[0]
    else:
        a = numpy.concatenate(coords)
    return a

  def move_atoms(self, tf):
    # Transform tf acts on scene coordinates
    for m,a in self.molatoms:
      axyz = m.xyz[a]
      atf = m.place.inverse() * tf * m.place
      atf.move(axyz)
      m.xyz[a] = axyz
      m.need_graphics_update = True
      m.redraw_needed = True

  def element_numbers(self):
    elnums = []
    for m,a in self.molatoms:
        elnums.append(m.element_nums[a])
    if len(elnums) == 1:
        return elnums[0]
    import numpy
    return numpy.concatenate(elnums)

  def separate_chains(self):
    clist = []
    csets = {}
    for m,a in self.molatoms:
      cids = set(m.chain_ids[a])
      for cid in cids:
        aset = csets.get((m,cid))
        if aset is None:
          csets[(m,cid)] = aset = Atom_Set()
          clist.append(aset)
        catoms = m.atom_subset(chain_id = cid, restrict_to_atoms = a)
        aset.add_atoms(m, catoms)
    return clist

  def extend_to_chains(self):
    aset = Atom_Set()
    for m,a in self.molatoms:
      cids = tuple(set(m.chain_ids[a]))
      catoms = m.atom_subset(chain_id = cids)
      aset.add_atoms(m, catoms)
    return aset

  def separate_molecules(self):
    mlist = []
    msets = {}
    for m,a in self.molatoms:
      aset = msets.get(m)
      if aset is None:
        msets[m] = aset = Atom_Set()
        mlist.append(aset)
      aset.add_atoms(m, a)
    return mlist

# -----------------------------------------------------------------------------
#
class Atom_Selection:
  def __init__(self, mol, a):
    self.molecule = mol
    self.atom = a
  def description(self):
    m = self.molecule
    a = self.atom
    d = '%s %s %s %d %s' % (m.name,
                            m.chain_ids[a].decode('utf-8'),
                            m.residue_names[a].decode('utf-8'),
                            m.residue_nums[a],
                            m.atom_names[a].decode('utf-8'))
    return d
  def models(self):
    return [self.molecule]
