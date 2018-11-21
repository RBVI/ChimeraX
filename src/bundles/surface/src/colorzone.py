# -----------------------------------------------------------------------------
# Color surfaces near specified points.
#

# -----------------------------------------------------------------------------
# Color the surface model within specified distances of the given
# list of points using the corresponding point colors.
# The points are in model object coordinates.
#
def color_zone(surface, points, point_colors, distance,
               sharp_edges = False, auto_update = True):

    zc = ZoneColor(surface, points, point_colors, distance, sharp_edges)
    zc.set_vertex_colors()
    
    if auto_update:
        from .updaters import add_updater_for_session_saving
        add_updater_for_session_saving(surface.session, zc)
    else:
        zc = None
    surface.auto_recolor_vertices = zc

# -----------------------------------------------------------------------------
#
def points_and_colors(atoms, bonds, bond_point_spacing = None):

    points = atoms.scene_coords
    colors = atoms.colors

    if bonds is not None:
        raise ValueError('bond points currently not supported')
        from .bondzone import bond_points_and_colors, concatenate_points
        bpoints, bcolors = bond_points_and_colors(bonds, bond_point_spacing)
        if not bpoints is None:
            points = concatenate_points(points, bpoints)
            colors.extend(bcolors)

    return points, colors

# -----------------------------------------------------------------------------
#
def color_surface(surf, points, point_colors, distance):

    varray = surf.vertices
    from chimerax.core.geometry import find_closest_points
    i1, i2, n1 = find_closest_points(varray, points, distance)

    from numpy import empty, uint8
    rgba = empty((len(varray),4), uint8)
    rgba[:,:] = surf.color
    for k in range(len(i1)):
        rgba[i1[k],:] = point_colors[n1[k]]
        
    surf.vertex_colors = rgba
    surf.coloring_zone = True

# -----------------------------------------------------------------------------
# Stop updating surface zone.
#
def uncolor_zone(model):
    model.vertex_colors = None
    model.auto_recolor_vertices = None

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class ZoneColor(State):
    def __init__(self, surface, points, point_colors, distance, sharp_edges):
        self.surface = surface
        self.points = points
        self.point_colors = point_colors
        self.distance = distance
        self.sharp_edges = sharp_edges

    def __call__(self):
        self.set_vertex_colors()

    def set_vertex_colors(self):
        surf = self.surface
        if surf.vertices is not None:
            color_surface(surf, self.points, self.point_colors, self.distance)
            if self.sharp_edges:
                color_zone_sharp_edges(surf, self.points, self.point_colors, self.distance,
                                       replace = True)
        surf.auto_recolor_vertices = self

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'surface': self.surface,
            'points': self.points,
            'point_colors': self.point_colors,
            'distance': self.distance,
            'sharp_edges': self.sharp_edges,
            'version': 1,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        surf = data['surface']
        if surf is None:
            return None		# Surface to color is gone.
        c = cls(surf, data['points'], data['point_colors'], data['distance'], data['sharp_edges'])
        c.set_vertex_colors()
        return c

        
# -----------------------------------------------------------------------------
#
def color_zone_sharp_edges(surface, points, colors, distance, replace = False):
    # Transform points to surface coordinates
    surface.scene_position.inverse().transform_points(points, in_place = True)

    varray = surface.vertices
    from chimerax.core.geometry import find_closest_points
    i1, i2, n1 = find_closest_points(varray, points, distance)

    tarray = surface.triangles
    ec = _edge_cuts(varray, tarray, i1, n1, points, colors, distance)
    
    from numpy import empty, uint8
    carray = empty((len(varray),4), uint8)
    carray[:,:] = surface.color
    for vi,ai in zip(i1, n1):
        carray[vi,:] = colors[ai]

    va, na, ta, ca = _cut_triangles(ec, varray, surface.normals, tarray, carray)

    if replace:
        surface.set_geometry(va, na, ta)
        surface.vertex_colors = ca
        
    return va, na, ta, ca

# -----------------------------------------------------------------------------
#
def _edge_cuts(varray, tarray, vi, pi, points, colors, distance):
    ec = []	# List of triples, two vertex indices and fraction (0-1) indicating cut point
    edges = _triangle_edges(tarray)
    vp = dict(zip(vi,pi))
    for v1,v2 in edges:
        p1,p2 = vp.get(v1), vp.get(v2)
        f = _edge_cut_position(varray, v1, v2, p1, p2, points, colors, distance)
        if f is not None:
            ec.append((v1,v2,f))
    return ec

# -----------------------------------------------------------------------------
#
def _triangle_edges(triangles):
    edges = set()
    for v1,v2,v3 in triangles:
        edges.add((v1,v2) if v1 < v2 else (v2,v1))
        edges.add((v2,v3) if v2 < v3 else (v3,v2))
        edges.add((v3,v1) if v3 < v1 else (v1,v3))
    return edges

# -----------------------------------------------------------------------------
#
def _edge_cut_position(varray, v1, v2, p1, p2, points, colors, distance):
    if p1 == p2:
        return None
    x1, x2 = varray[v1], varray[v2]
    if p2 is None:
        f = _cut_at_range(x1, x2, points[p1], distance)
    elif p1 is None:
        f = _cut_at_range(x1, x2, points[p2], distance)
    else:
        if (colors[p1] == colors[p2]).all():
            return None
        dx = x2-x1
        dp = points[p2]-points[p1]
        px = 0.5*(points[p2]+points[p1]) - x1
        from chimerax.core.geometry import inner_product
        f = inner_product(px, dp) / inner_product(dx, dp)
        if f <= -0.1 or f >= 1.1:
            raise ValueError('Cut fraction %.5g is out of range (0,1)' % f)
        if f < 0:
            f = 0
        elif f > 1:
            f = 1
    return f

# -----------------------------------------------------------------------------
#
def _cut_at_range(x1, x2, p, distance):
    from chimerax.core.geometry import distance as dist
    d1 = dist(x1, p)
    d2 = dist(x2, p)
    f = (d1-distance)/(d1-d2)
    return f
    
# -----------------------------------------------------------------------------
#
def _cut_triangles(edge_cuts, varray, narray, tarray, carray):
    e = {}
    vae = []
    nae = []
    cae = []
    nv = len(varray)
    from chimerax.core.geometry import normalize_vector
    for v1, v2, f in edge_cuts:
        p = (1-f)*varray[v1] + f*varray[v2]
        n = normalize_vector((1-f)*narray[v1] + f*narray[v2])
        vi = nv + len(vae)
        vae.extend((p,p))
        nae.extend((n,n))
        cae.extend((carray[v1], carray[v2]))
        e[(v1,v2)] = vi
        e[(v2,v1)] = vi+1

    if len(vae) == 0:
        return varray, narray, tarray, carray

    tae = []
    for v1,v2,v3 in tarray:
        p12, p23, p31 = e.get((v1,v2)), e.get((v2,v3)), e.get((v3,v1))
        cuts = 3 - [p12, p23, p31].count(None)
        p21, p32, p13 = e.get((v2,v1)), e.get((v3,v2)), e.get((v1,v3))
        if cuts == 3:
            # Add triangle center point, 3 copies
            p = (vae[p12-nv] + vae[p23-nv] + vae[p31-nv]) / 3
            n = normalize_vector(nae[p12-nv] + nae[p23-nv] + nae[p31-nv])
            tc1 = nv + len(vae)
            tc2 = tc1 + 1
            tc3 = tc1 + 2
            vae.extend((p,p,p))
            nae.extend((n,n,n))
            cae.extend((carray[v1], carray[v2], carray[v3]))
            tae.extend(((v1, p12, tc1), (v1, tc1, p13),
                        (v2, p23, tc2), (v2, tc2, p21),
                        (v3, p31, tc3), (v3, tc3, p32)))
        elif cuts == 2:
            if p31 is None:
                tae.extend(((v1, p12, p32), (v1, p32, v3), (v2, p23, p21)))
            elif p12 is None:
                tae.extend(((v2, p23, p13), (v2, p13, v1), (v3, p31, p32)))
            elif p23 is None:
                tae.extend(((v3, p31, p21), (v3, p21, v2), (v1, p12, p13)))
        elif cuts == 1:
            raise ValueError('Triangle with one cut edge')
        elif cuts == 0:
            tae.append((v1,v2,v3))

    from numpy import concatenate, array
    va = concatenate((varray, array(vae, varray.dtype)))
    na = concatenate((narray, array(nae, narray.dtype)))
    ta = array(tae, tarray.dtype)
    ca = concatenate((carray, array(cae, carray.dtype)))
    return va, na, ta, ca

# ---------------------------------------------------------------------------
#
def volume_zone_color(volume):
    for surf in volume.surfaces:
        zc = getattr(surf, 'auto_recolor_vertices', None)
        if isinstance(zc, ZoneColor):
            return zc
    return None
        
# ---------------------------------------------------------------------------
#
def split_volume_by_color_zone(volume):

    zc = volume_zone_color(volume)
    if zc is None:
        from chimerax.core.errors import UserError
        raise UserError('Volume %s does not have zone coloring' % volume.name_with_id())

    grids = split_zones_by_color(volume, zc.points, zc.point_colors, zc.distance)
    session = volume.session
    from chimerax.map import volume_from_grid_data
    vlist = [volume_from_grid_data(g, session, open_model = False) for g in grids]
    for v in vlist:
        v.copy_settings_from(volume, copy_region = False)
        rgba = tuple(c/255 for c in v.data.zone_color)
        v.set_parameters(surface_colors = [rgba]*len(v.surfaces))
        v.display = True
    volume.display = False

    if len(vlist) == 1:
        session.models.add(vlist)
    else:
        session.models.add_group(vlist, name = volume.name + ' split')
  
    return vlist
  
# ---------------------------------------------------------------------------
#
def split_zones_by_color(volume, points, point_colors, radius):

  ctable = {}
  cc = 0
  for c in point_colors:
    tc = tuple(c)
    if not tc in ctable:
      cc += 1
      ctable[tc] = cc
  point_indices = [ctable[tuple(c)] for c in point_colors]

  ijk_min, ijk_max, ijk_step = volume.region
  from chimerax.map.data import GridSubregion
  sg = GridSubregion(volume.data, ijk_min, ijk_max)

  # Get volume mask with values indicating nearest color within given radius.
  from chimerax.map.data import zone_mask, masked_grid_data
  mask = zone_mask(sg, points, radius, zone_point_mask_values = point_indices)

  grids = []
  for m in range(cc+1):
      g = masked_grid_data(sg, mask, m)
      g.name = volume.data.name + (' %d' % m)
      grids.append(g)

  # Record colors.
  for color, m in ctable.items():
    grids[m].zone_color = color
  grids[0].zone_color = volume.surfaces[0].rgba	# Outside zone color same as original map.
  
  return grids

# ---------------------------------------------------------------------------
#
def split_volumes_by_color_zone(session, volume):
    for v in volume:
        split_volume_by_color_zone(v)

# ---------------------------------------------------------------------------
#
def register_volume_split_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required = [('volume', MapsArg)],
        synopsis = 'split volume by color zone')
    register('volume splitbyzone', desc, split_volumes_by_color_zone, logger=logger)
