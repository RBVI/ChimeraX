from . import io
from . import models

CATEGORY = io.STRUCTURE


class StructureModel(models.Model):
    """Commom base class for atomic structures"""

    STRUCTURE_STATE_VERSION = 0

    def take_snapshot(self, session, flags):
        data = {}
        return [self.STRUCTURE_STATE_VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.STRUCTURE_STATE_VERSION or len(data) > 0:
            raise RuntimeError("Unexpected version or data")

    def reset_state(self):
        pass

    def make_drawing(self):
        m = self.mol_blob
        a = m.atoms
        coords = a.coords
        radii = a.radii
        scale = 0.3     # Ball and stick
        a.colors = colors = element_colors(a.element_numbers)
        b = m.bonds
        bradii = b.radii
#        halfbond = b.halfbonds
# TODO: Support per-bond half bond mode.
        halfbond = True
        bcolors = b.colors
        bonds = m.bond_indices

        # Create graphics
        self.create_atom_spheres(coords, radii, scale, colors)
        self.create_bond_cylinders(bonds, coords, bradii, scale, bcolors, colors, halfbond)

    def create_atom_spheres(self, coords, radii, scale, colors, triangles_per_sphere = 320):
        if not hasattr(self, '_atoms_drawing'):
            self._atoms_drawing = self.new_drawing('atoms')
        p = self._atoms_drawing

        # Set instanced sphere triangulation
        from . import surface
        va, na, ta = surface.sphere_geometry(triangles_per_sphere)
        p.geometry = va, ta
        p.normals = na

        # Set instanced sphere center position and radius
        n = len(coords)
        from numpy import empty, float32, multiply
        xyzr = empty((n, 4), float32)
        xyzr[:, :3] = coords
        multiply(radii, scale, xyzr[:, 3])
        from .geometry import place
        p.positions = place.Places(shift_and_scale=xyzr)

        # Set atom colors
        p.colors = colors

    def create_bond_cylinders(self, bonds, atom_coords, radii, scale,
                              bond_colors, atom_colors, half_bond_coloring):
        if not hasattr(self, '_bonds_drawing'):
            self._bonds_drawing = self.new_drawing('bonds')
        p = self._bonds_drawing

        from . import surface
        va, na, ta = surface.cylinder_geometry(caps = False)
        p.geometry = va, ta
        p.normals = na

        p.positions = bond_cylinder_placements(bonds, atom_coords, radii*scale, half_bond_coloring)

        self.set_bond_colors(bonds, bond_colors, atom_colors, half_bond_coloring)

    def set_bond_colors(self, bonds, bond_colors, atom_colors, half_bond_coloring):
        p = self._bonds_drawing
        if p is None:
            return

        if half_bond_coloring:
            bc0,bc1 = atom_colors[bonds[:,0],:], atom_colors[bonds[:,1],:]
            from numpy import concatenate
            p.colors = concatenate((bc0,bc1))
        else:
            p.colors = bond_colors

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
  from .geometry import place
  pl = place.Places(opengl_array = pt)
  return pl

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
