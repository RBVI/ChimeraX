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

# -----------------------------------------------------------------------------
# Create tube surface geometry passing through specified points.
#
def tube_through_points(path, tangents, radius = 1.0, circle_subdivisions = 15):

    circle = circle_points(circle_subdivisions, radius)
    circle_normals = circle_points(circle_subdivisions, 1.0)
    from ._surface import tube_geometry
    return tube_geometry(path, tangents, circle, circle_normals)

# -----------------------------------------------------------------------------
# Create tube surface geometry passing through a natural cubic spline through
# specified points.
#
def tube_spline(path, radius = 1.0, segment_subdivisions = 10, circle_subdivisions = 15):

    from chimerax.geometry import natural_cubic_spline
    spath, stan = natural_cubic_spline(path, segment_subdivisions)
    return tube_through_points(spath, stan, radius, circle_subdivisions)

# -----------------------------------------------------------------------------
# Return array of tube geometry vertex colors given path point colors.
#
def tube_geometry_colors(colors, segment_subdivisions, circle_subdivisions,
                         start_divisions, end_divisions):

    from . import _surface
    return _surface.tube_geometry_colors(colors, segment_subdivisions, circle_subdivisions,
                                         start_divisions, end_divisions)

# -----------------------------------------------------------------------------
# Return triangle mask corresponding to tube segment mask.
#
def tube_triangle_mask(segmask, segment_subdivisions, circle_subdivisions,
                       start_divisions, end_divisions):

    from . import _surface
    tmask = _surface.tube_triangle_mask(segmask, segment_subdivisions, circle_subdivisions,
                                        start_divisions, end_divisions)
    return tmask.view(bool)

# -----------------------------------------------------------------------------
#
def circle_points(n, radius):

    from numpy import arange, float32, empty, sin, cos
    from math import pi
    a = arange(n) * (2*pi/n)
    c = empty((n,3), float32)
    c[:,0] = radius*cos(a)
    c[:,1] = radius*sin(a)
    c[:,2] = 0
    return c

# -----------------------------------------------------------------------------
# Create tube surface geometry passing through specified points.
#
def tube_through_points_old(path, radius = 1.0, band_length = 0,
                        segment_subdivisions = 10, circle_subdivisions = 15,
                        color = (.745,.745,.745,1)):

    def shape(path, tangents, pcolors, r=radius, nc=circle_subdivisions):
        return Tube(path, tangents, pcolors, r, nc)

    va,na,ta,ca = extrusion(path, shape, band_length, segment_subdivisions, color)

    return va,na,ta,ca

# -----------------------------------------------------------------------------
#
class Tube:

    def __init__(self, path, tangents, pcolors, radius, circle_subdivisions,
                 end_caps = True):

        self.path = path                # Center line points
        self.tangents = tangents        # Tangents for each point
        self.pcolors = pcolors  # Point colors.
        self.radius = radius
        self.circle_subdivisions = circle_subdivisions
        self.end_caps = end_caps

    def geometry(self):

        nz = len(self.path)
        nc = self.circle_subdivisions
        height = 0
        tflist = extrusion_transforms(self.path, self.tangents)
        from .shapes import cylinder_geometry
        varray, narray, tarray = cylinder_geometry(self.radius, height, nz, nc,
                                                   caps = self.end_caps)
        # Transform circles.
        for i in range(nz):
            tflist[i].transform_points(varray[nc*i:nc*(i+1),:], in_place = True)
            tflist[i].transform_vectors(narray[nc*i:nc*(i+1),:], in_place = True)

        if self.end_caps:
            # Transform caps. Each cap is nc+1 vertices at end of varray.
            c0, c1 = nc*nz, nc*nz + nc+1
            tflist[0].transform_points(varray[c0:c1,:], in_place = True)
            tflist[0].transform_vectors(narray[c0:c1,:], in_place = True)
            tflist[-1].transform_points(varray[c1:,:], in_place = True)
            tflist[-1].transform_vectors(narray[c1:,:], in_place = True)
        return varray, narray, tarray

    def colors(self):

        nc = self.circle_subdivisions
        nz = len(self.pcolors)
        from numpy import empty, float32
        vcount = nz*nc+2*(nc+1) if self.end_caps else nz*nc
        carray = empty((vcount,4), float32)
        for i in range(nz):
            carray[nc*i:nc*(i+1),:] = self.pcolors[i]
        if self.end_caps:
            # Color caps. Each cap is nc+1 vertices at end of varray.
            c0, c1 = nc*nz, nc*nz + nc+1
            carray[c0:c1,:] = self.pcolors[0]
            carray[c1:,:] = self.pcolors[-1]
        return carray

# -----------------------------------------------------------------------------
#
def extrusion(path, shape, band_length = 0, segment_subdivisions = 10,
              color = (.745,.745,.745,1)):

    point_colors = [color]*len(path)
    segment_colors = [color] * (len(path) - 1)
    va,na,ta,ca = banded_extrusion(path, point_colors, segment_colors,
                                   segment_subdivisions, band_length, shape)
    return va,na,ta,ca

# -----------------------------------------------------------------------------
# Create an extruded surface along a path with banded coloring.
#
def banded_extrusion(xyz_path, point_colors, segment_colors,
                     segment_subdivisions, band_length, shape):

    if len(xyz_path) <= 1:
        return None             # No path

    from chimerax.geometry import natural_cubic_spline
    spath, stan = natural_cubic_spline(xyz_path, segment_subdivisions)

    pcolors = band_colors(spath, point_colors, segment_colors,
                          segment_subdivisions, band_length)

    s = shape(spath, stan, pcolors)
    va, na, ta = s.geometry()
    ca = s.colors()

    return va,na,ta,ca

# -----------------------------------------------------------------------------
# Compute transforms mapping (0,0,0) origin to points along path with z axis
# along path tangent.
#
def extrusion_transforms(path, tangents, yaxis = None):

    from chimerax.geometry import identity, vector_rotation, translation
    tflist = []
    if yaxis is None:
        # Make xy planes for coordinate frames at each path point not rotate
        # from one point to next.
        tf = identity()
        n0 = (0,0,1)
        for p1,n1 in zip(path, tangents):
            tf = vector_rotation(n0,n1) * tf
            tflist.append(translation(p1)*tf)
            n0 = n1
    else:
        # Make y-axis of coordinate frames at each point align with yaxis.
        from chimerax.geometry import normalize_vector, cross_product, Place
        for p,t in zip(path, tangents):
            za = t
            xa = normalize_vector(cross_product(yaxis, za))
            ya = cross_product(za, xa)
            tf = Place(((xa[0], ya[0], za[0], p[0]),
                        (xa[1], ya[1], za[1], p[1]),
                        (xa[2], ya[2], za[2], p[2])))
            tflist.append(tf)
    return tflist

# -----------------------------------------------------------------------------
# Calculate point colors for an interpolated set of points.
# Point colors are extended to interpolated points and segments within
# band_length/2 arc distance.
#
def band_colors(plist, point_colors, segment_colors,
                segment_subdivisions, band_length):

  n = len(point_colors)
  pcolors = []
  for k in range(n-1):
    j = k * (segment_subdivisions + 1)
    from chimerax.geometry import spline
    arcs = spline.arc_lengths(plist[j:j+segment_subdivisions+2])
    bp0, mp, bp1 = band_points(arcs, band_length)
    scolors = ([point_colors[k]]*bp0 +
               [segment_colors[k]]*mp +
               [point_colors[k+1]]*bp1)
    pcolors.extend(scolors[:-1])
  if band_length > 0:
      last = point_colors[-1]
  else:
      last = segment_colors[-1]
  pcolors.append(last)
  return pcolors
  
# -----------------------------------------------------------------------------
# Count points within band_length/2 of each end of an arc.
#
def band_points(arcs, band_length):
      
  arc = arcs[-1]
  half_length = min(.5 * band_length, .5 * arc)
  bp0 = mp = bp1 = 0
  for p in range(len(arcs)):
    l0 = arcs[p]
    l1 = arc - arcs[p]
    if l0 < half_length:
      if l1 < half_length:
        if l0 <= l1:
          bp0 += 1
        else:
          bp1 += 1
      else:
        bp0 += 1
    elif l1 < half_length:
      bp1 += 1
    else:
      mp += 1

  return bp0, mp, bp1

# -----------------------------------------------------------------------------
# Create a tube surface passing through specified atoms.
#
def tube_through_atoms(path_atoms, radius = 0, band_length = 0,
                       segment_subdivisions = 10, circle_subdivisions = 15,
                       follow_bonds = True, color = None):

    def shape(path, tangents, pcolors, r=radius, nc=circle_subdivisions):
        return Tube(path, tangents, pcolors, r, nc)

    va,na,ta,ca = _atom_extrusion(path_atoms, shape, band_length, segment_subdivisions,
                                  follow_bonds, color)
    return va,na,ta,ca

# -----------------------------------------------------------------------------
# Create a ribbon surface passing through specified atoms.
#
def ribbon_through_atoms(path_atoms, width = 0, yaxis = None, twist = 0,
                         band_length = 0,
                         segment_subdivisions = 10, width_subdivisions = 15,
                         follow_bonds = True, color = None):

    def shape(path, tangents, pcolors, w=width, nw=width_subdivisions, y=yaxis, t=twist):
        return Ribbon(path, tangents, pcolors, w, nw, y, t)

    va,na,ta,ca = _atom_extrusion(path_atoms, shape, band_length, segment_subdivisions,
                                  follow_bonds, color)
    return va,na,ta,ca

# -----------------------------------------------------------------------------
#
def _atom_extrusion(path_atoms, shape, band_length = 0, segment_subdivisions = 10,
                    follow_bonds = True, color = None):

    if len(path_atoms) == 0:
        return None, None, None, None

    if follow_bonds:
        chains = _atom_chains(path_atoms)
    else:
        chains = [(path_atoms,None)]
        if color is None:
            color = (190,190,190,255)

    glist = []
    for atoms, bonds in chains:
        xyz_path = atoms.scene_coords
        point_colors = atoms.colors
        segment_colors =  bonds.colors if color is None else ([color] * (len(atoms) - 1))
        geom = banded_extrusion(xyz_path, point_colors, segment_colors,
                                segment_subdivisions, band_length, shape)
        if geom is not None:
            glist.append(geom)

    if len(glist) == 0:
        return None, None, None, None
    
    from . import combine_geometry_vntc
    va,na,ta,ca = combine_geometry_vntc(glist)
    
    return va,na,ta,ca
    
# -----------------------------------------------------------------------------
# Return a list of atom chains.  An atom chain is a sequence
# of atoms connected by bonds where all non-end-point atoms have exactly 2
# bonds.  A chain is represented by a 2-tuple, the first element being the
# ordered list of atoms, and the second being the ordered list of bonds.
# In a chain which is a cycle all atoms have 2 bonds and the first and
# last atom in the chain are the same.  Non-cycles have end point atoms
# with more or less than 2 bonds.
#
def _atom_chains(atoms):

  atom_bonds = {}       # Bonds connecting specified atoms.
  aset = set(atoms)
  for a in atoms:
      atom_bonds[a] = [b for b in a.bonds if b.other_atom(a) in aset]

  used_bonds = {}
  chains = []
  for a in atoms:
    if len(atom_bonds[a]) != 2:
      for b in atom_bonds[a]:
        if b not in used_bonds:
          used_bonds[b] = 1
          c = _trace_chain(a, b, atom_bonds)
          chains.append(c)
          end_bond = c[1][-1]
          used_bonds[end_bond] = 1

  #
  # Pick up cycles
  #
  reached_atoms = {}
  for catoms, bonds in chains:
    for a in catoms:
      reached_atoms[a] = 1

  for a in atoms:
    if a not in reached_atoms:
      bonds = atom_bonds[a]
      if len(bonds) == 2:
        b = bonds[0]
        c = _trace_chain(a, b, atom_bonds)
        chains.append(c)
        for a in c[0]:
          reached_atoms[a] = 1
      
  return chains
          
# -----------------------------------------------------------------------------
#
def _trace_chain(atom, bond, atom_bonds):

  atoms = [atom]
  bonds = [bond]

  a = atom
  b = bond
  while 1:
    a = b.other_atom(a)
    atoms.append(a)
    if a == atom:
      break                     # loop
    blist = list(atom_bonds[a])
    blist.remove(b)
    if len(blist) != 1:
      break
    b = blist[0]
    bonds.append(b)

  from chimerax.atomic import Atoms, Bonds
  return (Atoms(atoms), Bonds(bonds))

# -----------------------------------------------------------------------------
#
class Ribbon:

    def __init__(self, path, tangents, pcolors, width, width_subdivisions,
                 yaxis = None, twist = 0):

        self.path = path    # Center line points
        self.tangents = tangents        # Tangents for each point
        self.pcolors = pcolors  # Point colors.
        self.width = width
        self.width_subdivisions = width_subdivisions
        self.yaxis = yaxis
        self.twist = twist

    def geometry(self):

        nz = len(self.path)
        nw = self.width_subdivisions + 1
        height = 0
        tflist = extrusion_transforms(self.path, self.tangents, self.yaxis)
        from chimerax.shape.shape import rectangle_geometry
        varray, tarray = rectangle_geometry(self.width, height, nw, nz)
        narray = varray.copy()
        narray[:] = (0,-1,0)
        
        if self.twist != 0:
            from chimerax.geometry import rotation
            twist_tf = rotation((0,0,1), self.twist)
            twist_tf.transform_points(varray, in_place = True)
            twist_tf.transform_vectors(narray, in_place = True)

        # Transform transverse lines.
        va = varray.reshape((nz,nw,3))
        na = narray.reshape((nz,nw,3))
        for i in range(nz):
            tflist[i].transform_points(va[i,:,:], in_place = True)
            tflist[i].transform_vectors(na[i,:,:], in_place = True)
        return varray, narray, tarray

    def colors(self):

        nz = len(self.pcolors)
        nw = self.width_subdivisions + 1
        from numpy import empty, float32
        carray = empty((nz*nw,4), float32)
        for i in range(nz):
            carray[nw*i:nw*(i+1),:] = self.pcolors[i]
        return carray
