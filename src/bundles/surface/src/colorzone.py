# -----------------------------------------------------------------------------
# Color surfaces near specified points.
#

# -----------------------------------------------------------------------------
# Color the surface model within specified distances of the given
# list of points using the corresponding point colors.
# The points are in model object coordinates.
#
def color_zone(surface, points, point_colors, distance,
               sharp_edges = False, far_color = None, auto_update = True):
    '''
    Color a surface according to the nearest of specified points, with a color
    associated with each point.  Surface vertices are colored if they are within
    the specified distance of some point.  Surface vertices farther away are colored
    by far_color or retain their original color if no far_color is specified.

    surface : Surface model
      Surface to color.
    points : N x 3 array of float
      Point positions in scene coordinates.
    point_colors : N x 4 array of uint8 RGBA values
      RGBA color for each point.
    distance : float
      Maximum distance of surface to point for coloring.
    sharp_edges : bool
      Whether to divide surface triangles so that the boundaries between
      surface patches near different points show sharp color transitions
      and the boundary curves are less jagged.
    far_color : RGBA 4-tuple 0-255 range or None
      Color for surface vertices further than distance from all points
    auto_update : bool
      Whether to automatically update the surface coloring when the surface shape changes.
    '''
    
    zc = ZoneColor(surface, points, point_colors, distance, sharp_edges,
                   far_color = far_color)
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

    if bonds is not None and len(bonds) > 0:
        from .bondzone import bond_points_and_colors
        bpoints, bcolors = bond_points_and_colors(bonds, bond_point_spacing)
        if not bpoints is None:
            from numpy import concatenate
            points = concatenate((points, bpoints))
            colors = concatenate((colors, bcolors))

    return points, colors

# -----------------------------------------------------------------------------
#
def color_surface(surf, points, point_colors, distance, far_color = None):

    varray = surf.vertices
    from chimerax.geometry import find_closest_points
    i1, i2, n1 = find_closest_points(varray, points, distance)

    if isinstance(far_color, str) and far_color == 'keep':
        rgba = surf.get_vertex_colors(create = True)
    else:
        from numpy import empty, uint8
        rgba = empty((len(varray),4), uint8)
        rgba[:,:] = (surf.color if far_color is None else far_color)
        
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
    def __init__(self, surface, points, point_colors, distance, sharp_edges, far_color = None):
        self.surface = surface
        self.points = points
        self.point_colors = point_colors
        self.distance = distance
        self.sharp_edges = sharp_edges
        self.far_color = far_color
        
    def __call__(self):
        self.set_vertex_colors()

    def set_vertex_colors(self):
        surf = self.surface
        if surf.vertices is not None:
            color_surface(surf, self.points, self.point_colors, self.distance,
                          far_color = self.far_color)
            if self.sharp_edges:
                color_zone_sharp_edges(surf, self.points, self.point_colors, self.distance,
                                       far_color = self.far_color, replace = True)
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
            'far_color': self.far_color,
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
        c = cls(surf, data['points'], data['point_colors'], data['distance'], data['sharp_edges'],
                far_color = data.get('far_color'))
        c.set_vertex_colors()
        return c

# -----------------------------------------------------------------------------
#
def color_zoning(surface):
    zc = surface.auto_recolor_vertices
    return zc if isinstance(zc, ZoneColor) else None
        
# -----------------------------------------------------------------------------
#
def color_zone_sharp_edges(surface, points, colors, distance, far_color = None,
                           replace = False):
    # Transform points to surface coordinates
    surface.scene_position.inverse().transform_points(points, in_place = True)

    varray, narray, tarray = surface.vertices, surface.normals, surface.triangles
    if hasattr(surface, '_unsharp_geometry'):
        va_us, na_us, ta_us, va_sh, na_sh, ta_sh = surface._unsharp_geometry
        if varray is va_sh and narray is na_sh and tarray is ta_sh:
            varray, narray, tarray = va_us, na_us, ta_us
        
    from chimerax.geometry import find_closest_points
    i1, i2, n1 = find_closest_points(varray, points, distance)

    ec = _edge_cuts(varray, tarray, i1, n1, points, colors, distance)

    from numpy import empty, uint8
    carray = empty((len(varray),4), uint8)
    carray[:,:] = (surface.color if far_color is None or far_color == 'keep' else far_color)
    for vi,ai in zip(i1, n1):
        carray[vi,:] = colors[ai]

    va, na, ta, ca = _cut_triangles(ec, varray, narray, tarray, carray)

    if replace:
        surface._unsharp_geometry = (varray, narray, tarray, va, na, ta)
        surface.set_geometry(va, na, ta)
        surface.vertex_colors = ca
        from . import unique_vertex_map
        vmap = unique_vertex_map(va)
        surface.joined_triangles = vmap[ta]
        
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
        from chimerax.geometry import inner_product
        dxdp = inner_product(dx, dp)
        f = 0 if dxdp == 0 else inner_product(px, dp) / dxdp
        # Floating point precision limits can put f outside 0-1.
        if f < 0:
            f = 0
        elif f > 1:
            f = 1
    return f

# -----------------------------------------------------------------------------
#
def _cut_at_range(x1, x2, p, distance):
    from chimerax.geometry import distance as dist
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
    from chimerax.geometry import normalize_vector
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
    '''Create new volumes for each color zoned region of a specified volume.'''
    
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
  from chimerax.map_data import GridSubregion
  sg = GridSubregion(volume.data, ijk_min, ijk_max)

  # Get volume mask with values indicating nearest color within given radius.
  from chimerax.map_data import zone_mask, masked_grid_data
  mask = zone_mask(sg, points, radius, zone_point_mask_values = point_indices)

  grids = []
  for m in range(cc+1):
      g = masked_grid_data(sg, mask, m)
      g.name = volume.data.name + (' %d' % m)
      grids.append(g)

  # Record colors.
  for color, m in ctable.items():
    grids[m].zone_color = color
  grids[0].zone_color = volume.surfaces[0].color	# Outside zone color same as original map.
  
  return grids

# ---------------------------------------------------------------------------
#
def split_volumes_by_color_zone(session, volumes):
    vlist = []
    for v in volumes:
        vlist.extend(split_volume_by_color_zone(v))
    return vlist

# ---------------------------------------------------------------------------
#
def register_volume_split_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required = [('volumes', MapsArg)],
        synopsis = 'split volume by color zone')
    register('volume splitbyzone', desc, split_volumes_by_color_zone, logger=logger)
