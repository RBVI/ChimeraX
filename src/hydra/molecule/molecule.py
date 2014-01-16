from ..surface import Surface
class Molecule(Surface):
  '''
  A Molecule represents atoms, bonds, residues and chains, typically read from file formats
  defined by the Protein Data Bank.  The data includes atomic coordinates, atom names,
  residue names and numbers, and chain identifiers.  A molecule represents both the data and
  the display style, color, and visibility used for drawing the molecule.
  '''

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
    from numpy import ones, zeros, bool
    self.atom_shown = ones((n,), bool)
    self.atoms_surface_piece = None
    self.atom_style = 'sphere'          # sphere, stick or ballstick
    self.ball_scale = 0.3               # Atom radius scale factor in ball and stick.
    self.bonds_surface_piece = None
    self.bond_color = (150,150,150,255)
    self.half_bond_coloring = True
    self.ribbon_shown_count = 0
    self.ribbon_shown = zeros((n,), bool)        # Only ribbon guide atoms used
    self.ribbon_radius = 1.0
    self.ribbon_subdivisions = (5,10)   # per-residue along length, and circumference
    self.ribbon_surface_pieces = {}     # Map chain id to surface piece

    self.color = (180,180,180,255)      # RGBA 0-255 integer values
    self.color_mode = 'by chain'        # by chain, by element, or single

    # Graphics objects
    self.triangles_per_sphere = 20      # Current level of detail
    self.need_graphics_update = True    # Update is done before drawing
    self.atom_surface_piece = None

  def atoms(self):
    '''Return an Atoms object containing all the molecule atoms.'''
    a = Atoms()
    a.add_molecules([self])
    return a

  def draw(self, viewer, draw_pass):
    '''Draw the molecule using the current style.'''

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
        self.remove_piece(p)
      return

    if p is None:
      self.atoms_surface_piece = p = self.new_piece()

    ntri = self.triangles_per_sphere
    from .. import surface
    va, na, ta = surface.sphere_geometry(ntri)
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

  def any_ribbons_shown(self):
    return self.ribbon_shown_count > 0

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
        self.remove_piece(p)
      return

    if p is None:
      self.bonds_surface_piece = p = self.new_piece()

    from .. import surface
    va, na, ta = surface.cylinder_geometry(caps = False)
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
    if not self.any_ribbons_shown():
      self.remove_pieces(rsp.values())
      rsp.clear()
      return

    cids = set(self.chain_ids)
    from .colors import rgba_256
    from numpy import uint8
    for cid in cids:
      s = self.ribbon_guide_atom_indices(cid)
      if len(s) <= 1:
        continue
      rshow = self.ribbon_shown[s]
      if rshow.sum() == 0:
        if cid in rsp:
          self.remove_piece(rsp.pop(cid))
        continue
      path = self.xyz[s]
    
      color = rgba_256[cid[0]]
      va,na,ta,ca = self.ribbon_geometry(path, color)

      if cid in rsp:
        p = rsp[cid]
      else:
        rsp[cid] = p = self.new_piece()
      p.geometry = va, ta
      p.normals = na
      p.vertex_colors = ca

  def ribbon_geometry(self, path, color):
    sd, cd = self.ribbon_subdivisions
    from ..surface import tube
    va,na,ta,ca = tube.tube_through_points(path, radius = self.ribbon_radius,
                                           color = color,
                                           segment_subdivisions = sd,
                                           circle_subdivisions = cd)
    return va, na, ta, ca

  def ribbon_guide_atom_indices(self, chain_id):
    s = self.atom_index_subset('CA', chain_id)
    if len(s) <= 1:
      s = self.atom_index_subset("C5'", chain_id)
    return s

  def ribbon_residues(self):
    cres = []
    cids = set(self.chain_ids)
    for cid in cids:
      s = self.ribbon_guide_atom_indices(cid)
      if len(s) > 1:
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
    
  def all_atom_indices(self):

    from numpy import arange
    return arange(self.atom_count())

  def atom_subset(self, atom_name = None, chain_id = None, residue_range = None,
                  residue_name = None, invert = False, restrict_to_atoms = None):
    '''
    Return a subset of atoms with specifie atom name, chain id, residue range,
    and residue name.
    '''
    ai = self.atom_index_subset(atom_name, chain_id, residue_range, residue_name,
                                invert, restrict_to_atoms)
    a = Atoms()
    if len(ai) > 0:
      a.add_atom_indices(self, ai)
    return a

  def atom_index_subset(self, atom_name = None, chain_id = None, residue_range = None,
                        residue_name = None, invert = False, restrict_to_atoms = None):

    anames = self.atom_names
    na = self.atom_count()
    from numpy import zeros, bool, logical_or, logical_and, logical_not
    nimask = zeros((na,), bool)
    if atom_name is None:
      nimask[:] = 1
    else:
      logical_or(nimask, (anames == atom_name.encode('utf-8')), nimask)

    if not chain_id is None:
      if isinstance(chain_id, (list,tuple)):
        chain_ids = chain_id
        cmask = self.chain_atom_mask(chain_ids[0])
        for cid in chain_ids[1:]:
          logical_or(cmask, self.chain_atom_indices(cid), cmask)
      else:
        cmask = self.chain_atom_mask(chain_id)
      logical_and(nimask, cmask, nimask)

    if not residue_range is None:
      r1, r2 = residue_range
      if not r1 is None:
        logical_and(nimask, (self.residue_nums >= r1), nimask)
      if not r2 is None:
        logical_and(nimask, (self.residue_nums <= r2), nimask)

    if not residue_name is None:
      rnames = self.residue_names
      logical_and(nimask, (rnames == residue_name.encode('utf-8')), nimask)

    if invert:
      logical_not(nimask, nimask)

    if not restrict_to_atoms is None:
      ramask = zeros((na,), bool)
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

  def show_index_atoms(self, atom_indices, only_these = False):
    a = self.atom_shown
    if only_these:
      a[:] = False
    if len(atom_indices) > 0:
      a[atom_indices] = True
    self.atom_shown_count = a.sum()
    self.need_graphics_update = True
    self.redraw_needed = True

  def hide_index_atoms(self, atom_indices):
    a = self.atom_shown
    if len(atom_indices) > 0:
      a[atom_indices] = False
      self.atom_shown_count = a.sum()
      self.need_graphics_update = True
      self.redraw_needed = True

  def show_ribbon_for_index_atoms(self, atom_indices, only_these = False):
    rs = self.ribbon_shown
    if only_these:
      rs[:] = False
    if len(atom_indices) > 0:
      rs[atom_indices] = True
    self.ribbon_shown_count = rs.sum()
    self.need_graphics_update = True
    self.redraw_needed = True

  def hide_ribbon_for_index_atoms(self, atom_indices):
    rs = self.ribbon_shown
    if len(atom_indices) > 0:
      rs[atom_indices] = False
      self.ribbon_shown_count = rs.sum()
      self.need_graphics_update = True
      self.redraw_needed = True

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
    self.ribbon_shown[:] = (1 if display else 0)
    self.ribbon_shown_count = self.ribbon_shown.sum()
    self.need_graphics_update = True
    self.redraw_needed = True

  def set_atom_style(self, style):
    '''Set the atom display style to "sphere", "stick", or "ballstick".'''
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
    '''Return the number of atoms in the molecule. Does not include molecule copies.'''
    return len(self.xyz)

  def shown_atom_count(self):
    '''Return the number of displayed atoms in the molecule. Includes molecule copies.'''
    nc = max(1, len(self.copies))
    na = self.atom_shown_count
    return na * nc

  def chain_atom_indices(self, chain_id):
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

  def atom_index_description(self, a):
    d = '%s %s %s %d %s' % (self.name,
                            self.chain_ids[a].decode('utf-8'),
                            self.residue_names[a].decode('utf-8'),
                            self.residue_nums[a],
                            self.atom_names[a].decode('utf-8'))
    return d

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
class Atoms:
  '''
  An atom set is a collection of atoms from one or more molecules.
  Properties of the atoms such as their x,y,z coordinates or radii can be
  accessed as arrays for efficient computation.
  '''
  def __init__(self):
    self.molatoms = []      # Pairs (molecule, atom index array)
  def add_molecules(self, molecules):
    '''Add all atoms of the specified molecules to the set.'''
    for m in molecules:
      self.molatoms.append((m, m.all_atom_indices()))
  def add_atom_indices(self, mol, ai):
    self.molatoms.append((mol, ai))
  def add_atoms(self, atoms):
    '''Add atoms to the set.'''
    self.molatoms.extend(atoms.molatoms)
  def molecules(self):
    '''List of molecules in set.'''
    return list(set(m for m,a in self.molatoms))
  def count(self):
    '''Number of atoms in set.'''
    return sum(len(a) for m,a in self.molatoms)
  def coordinates(self):
    '''Return a numpy array of atom coordinates in the global coordinate system.'''
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

  def radii(self):
    '''Return a numpy array of atom radii.'''
    rlist = []
    for m,a in self.molatoms:
        rlist.append(m.radii[a])
    import numpy
    if len(rlist) == 0:
        a = numpy.zeros((0,), numpy.float32)
    elif len(rlist) == 1:
        a = rlist[0]
    else:
        a = numpy.concatenate(rlist)
    return a

  def show_atoms(self, only_these = False):
    '''Display the atoms.'''
    for m, ai in self.molatoms:
      m.show_index_atoms(ai, only_these)

  def hide_atoms(self):
    '''Undisplay the atoms.'''
    for m, ai in self.molatoms:
      m.hide_index_atoms(ai)

  def show_ribbon(self, only_these = False):
    '''Show ribbons for residues containing the specified atoms.'''
    for m, ai in self.molatoms:
      m.show_ribbon_for_index_atoms(ai, only_these)

  def hide_ribbon(self):
    '''Hide ribbons for residues containing the specified atoms.'''
    for m, ai in self.molatoms:
      m.hide_ribbon_for_index_atoms(ai)

  def move_atoms(self, tf):
    '''Move atoms using a transform acting in scene global coordinates.'''
    for m,a in self.molatoms:
      axyz = m.xyz[a]
      atf = m.place.inverse() * tf * m.place
      atf.move(axyz)
      m.xyz[a] = axyz
      m.need_graphics_update = True
      m.redraw_needed = True

  def element_numbers(self):
    '''Return a numpy array of atom element numbers (e.g. 6 = carbon, 8 = oxygen).'''
    elnums = []
    for m,a in self.molatoms:
        elnums.append(m.element_nums[a])
    if len(elnums) == 1:
        return elnums[0]
    import numpy
    return numpy.concatenate(elnums)

  def residue_numbers(self):
    '''Return a numpy array of residue numbers for each atom.'''
    resnums = []
    for m,a in self.molatoms:
        resnums.append(m.residue_nums[a])
    if len(resnums) == 1:
        return resnums[0]
    import numpy
    return numpy.concatenate(resnums)

  def subset(self, indices):
    '''
    Return an Atoms object containing the atoms in the specified position in this set.
    The indices must be in increasing order.
    '''
    asubset = Atoms()
    mi = self.indices_by_molecule(indices)
    for (m,a), mind in zip(self.molatoms, mi):
      if mind:
        asubset.add_atom_indices(m, a[mind])
    return asubset

  def indices_by_molecule(self, indices):
    mi = []
    i = na = 0
    for m,a in self.molatoms:
      na += len(a)
      mind = []
      while i < len(indices) and indices[i] < na:
        mind.append(indices[i])
        i += 1
      mi.append(mind)
    return mi

  def separate_chains(self):
    '''Return copies of this atom set where each copy has atoms from a separate chain.'''
    clist = []
    csets = {}
    for m,a in self.molatoms:
      cids = set(m.chain_ids[a])
      for cid in cids:
        aset = csets.get((m,cid))
        if aset is None:
          csets[(m,cid)] = aset = Atoms()
          clist.append(aset)
        catoms = m.atom_index_subset(chain_id = cid, restrict_to_atoms = a)
        aset.add_atom_indices(m, catoms)
    return clist

  def extend_to_chains(self):
    '''
    Return a copy of this atom set extended to include all atoms of
    chains which have atoms in the current set.
    '''
    aset = Atoms()
    for m,a in self.molatoms:
      cids = tuple(set(m.chain_ids[a]))
      catoms = m.atom_index_subset(chain_id = cids)
      aset.add_atom_indices(m, catoms)
    return aset

  def separate_molecules(self):
    '''Return copies of this atoms set each having atoms from just one molecule.'''
    mlist = []
    msets = {}
    for m,a in self.molatoms:
      aset = msets.get(m)
      if aset is None:
        msets[m] = aset = Atoms()
        mlist.append(aset)
      aset.add_atom_indices(m, a)
    return mlist

  def exclude_water(self):
    '''Return a copy of this atom set with waters (residue name HOH) removed.'''
    aset = Atoms()
    for m,a in self.molatoms:
      aset.add_atom_indices(m, a[m.residue_names[a] != b'HOH'])
    return aset

  def names(self):
    '''Return a list of text names (strings) for each atom in this set.'''
    names = []
    for m,alist in self.molatoms:
      for a in alist:
        names.append(m.atom_index_description(a))
    return names
    
# -----------------------------------------------------------------------------
#
def all_atoms(session):
  '''Return an atom set containing all atoms of all open molecules.'''
  aset = Atoms()
  aset.add_molecules(session.molecules())
  return aset

# -----------------------------------------------------------------------------
#
class Atom_Selection:
  def __init__(self, mol, a):
    self.molecule = mol
    self.atom = a
  def description(self):
    return self.molecule.atom_index_description(self.atom)
  def models(self):
    return [self.molecule]
