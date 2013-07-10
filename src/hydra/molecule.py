from .surface import Surface
class Molecule(Surface):

#  def __init__(self, path, xyz, elements, chain_ids):
  def __init__(self, path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names):
    Surface.__init__(self)

    self.path = path
    from os.path import basename
    self.name = basename(path)
    self.xyz = xyz
    self.element_nums = element_nums
    from . import _image3d
    self.radii = _image3d.element_radii(element_nums)
    self.chain_ids = chain_ids
    self.residue_nums = res_nums
    self.residue_names = res_names
    #
    # Chain ids, residue names and atom names are Numpy string arrays.
    # The strings have maximum fixed length.  The strings are python byte arrays,
    # not Python unicode strings.  Comparison to a Python unicode string will
    # always give false.  Need to compare to a byte array.
    # For example, atom_names == 'CA'.encode('utf-8')
    #
    self.atom_names = atom_names

    # Graphics settings
    self.show_atoms = True
    self.shown_atoms = None             # Array of atom indices. None means all.
    self.atoms_surface_piece = None
    self.show_ribbons = False
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
    self.update_ribbon_graphics()

  def update_atom_graphics(self, viewer):

    self.create_atom_spheres()
    self.set_sphere_colors()

  def create_atom_spheres(self):

    p = self.atoms_surface_piece
    if not self.show_atoms:
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
    xyzr[:,3] = self.radii
    s = self.shown_atoms
    sas = xyzr if s is None else xyzr[s]
    p.shift_and_scale = sas

  def set_sphere_colors(self):

    cm = self.color_mode
    if cm == 'by chain':
      self.color_by_chain()
    elif cm == 'by element':
      self.color_by_element()
    else:
      self.color_one_color()
    
  def set_color_mode(self, mode):
    if mode == self.color_mode:
      return
    self.color_mode = mode
    self.need_graphics_update = True
    self.redraw_needed = True

  def color_by_chain(self):
    p = self.atoms_surface_piece
    if p:
      s = self.shown_atoms
      cids = self.chain_ids if s is None else self.chain_ids[s]
      p.instance_colors = chain_colors(cids)

  def color_by_element(self):
    p = self.atoms_surface_piece
    if p:
      s = self.shown_atoms
      elnums = self.element_nums if s is None else self.element_nums[s]
      p.instance_colors = element_colors(elnums)

  def color_one_color(self):
    p = self.atoms_surface_piece
    if p:
      s = self.shown_atoms
      n = len(self.xyz) if s is None else len(s)
      from numpy import empty, uint8
      c = empty((n,4),uint8)
      c[:,:] = self.color
      p.instance_colors = c

  def update_ribbon_graphics(self):

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
        s = self.atom_subset('P', cid)
        if len(s) <= 1:
          continue
      path = self.xyz[s]
    
      from . import tube
      va,na,ta,ca = tube.tube_through_points(path, radius = 1.0,
                                             color = rgba_256[cid[0]],
                                             segment_subdivisions = 5,
                                             circle_subdivisions = 10)
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
        s = self.atom_subset('P', cid)
        if len(s) <= 1:
          continue
      cres.append((cid, self.residue_nums[s]))
    return cres

  def show_nonribbon_atoms(self):
    cres = self.ribbon_residues()
    cr = set()
    for cid, rnums in cres:
      for rnum in rnums:
        cr.add((cid, rnum))
    n = len(self.residue_nums)
    cid = self.chain_ids
    rnum = self.residue_nums
    from numpy import array, int32
    atoms = array([i for i in range(n) if not (cid[i], rnum[i]) in cr], int32)
    self.shown_atoms = atoms
    self.show_atoms = True
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
      if self.show_atoms:
        self.need_graphics_update = True

  def set_display_style(self, atoms = False, ribbons = False):
    if atoms != self.show_atoms or ribbons != self.show_ribbons:
      self.show_atoms = atoms
      self.show_ribbons = ribbons
      self.need_graphics_update = True
      self.redraw_needed = True

  def first_intercept(self, mxyz1, mxyz2):
    # TODO check intercept of bounding box as optimization
    from . import _image3d
    if self.copies:
      intercepts = []
      from . import matrix
      for tf in self.copies:
        cxyz1, cxyz2 = matrix.apply_matrix(matrix.invert_matrix(tf), (mxyz1, mxyz2))
        intercepts.append(_image3d.closest_sphere_intercept(self.xyz, self.radii, cxyz1, cxyz2))
      f = [fmin for fmin, snum in intercepts if not fmin is None]
      fmin = min(f) if f else None
    else:
      fmin, snum = _image3d.closest_sphere_intercept(self.xyz, self.radii, mxyz1, mxyz2)
    return fmin

  def atom_count(self):
    return len(self.xyz)

  def atoms_shown(self):
    if not self.show_atoms:
      return 0
    nc = max(1, len(self.copies))
    na = len(self.xyz) if self.shown_atoms is None else len(self.shown_atoms)
    return na * nc

  def chain_atoms(self, chain_id):
    from numpy import fromstring
    cid = fromstring(chain_id, self.chain_ids.dtype)
    atoms = (self.chain_ids == cid).nonzero()[0]
    return atoms

  def bounds(self):
    xyz = self.xyz
    if len(xyz) == 0:
      return None
    return xyz.min(axis = 0), xyz.max(axis = 0)

# Only produces 20, 80, 320, ... (multiples of 4) triangle count.
def sphere_geometry(ntri):
  from . import icosahedron
  va, ta = icosahedron.icosahedron_geometry()
  from numpy import int32, sqrt
  ta = ta.astype(int32)
  from . import _image3d
  while 4*len(ta) <= ntri:
    va, ta = _image3d.subdivide_triangles(va, ta)
  vn = sqrt((va*va).sum(axis = 1))
  for a in (0,1,2):
    va[:,a] /= vn
  return va, va, ta

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
    from .matrix import transform_points
    for m,a in self.molatoms:
        xyz = m.xyz[a]
        transform_points(xyz, m.place)
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
    from .matrix import transform_points, invert_matrix, multiply_matrices
    for m,a in self.molatoms:
      axyz = m.xyz[a]
      atf = multiply_matrices(invert_matrix(m.place), tf, m.place)
      transform_points(axyz, atf)
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
