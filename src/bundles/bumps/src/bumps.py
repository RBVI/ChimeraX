# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def bumps(session, volume, center = None, range = None, base_area = 10.0, height = 1.0,
          marker_radius = 1.0, marker_color = (100,200,100,255), color_surface = True,
          name = 'bumps', all_extrema = False):
    '''
    Find protrusions on T-cells in 3d light microscopy.

    Algorithm finds points on contour surface whose distance from a center point is locally maximal
    then extends to neighbor grid points inside the surface as long as the border area (ie protrusion
    base area) is less than a specified value.  Protrusions of sufficient height are marked.  A marker
    closer to the center and within another protrusion is not marked.

    Parameters
    ----------
    volume : Volume
        Map to find protrusions on.  Highest surface contour level used.
    center : Center
        Point which is the cell center for finding radial protrusions.
    range : float or None
        How far out from center to look for protrusions.
    base_area : float
        Area of base of protrusion.  Protrusion is extended inward until this
        area is attained and that defines the protrusion height.
    height : float
        Minimum height of a protrusion to be marked.
    marker_radius : float
        Size of marker spheres to place at protrusion tips.
    marker_color : uint8 4-tuple
        Color of markers.  Default light green.
    color_surface : bool
        Whether to color the protrusion surface near the protrusion grid points.
        Each protrusion is assigned a random color.  Default true.
    name : string
        Name of created marker model. Default "bumps".
    all_extrema : bool
        Whether to mark all radial extrema even if the don't meet the protrusion height minimum.
        Markers within another protrusion are colored yellow, ones that never attain the specified
        protrusion base_area (often smal disconnected density blobs) are colored pink, markers
        on protrusions that are too short are colored blue.
    '''

    c = center.scene_coordinates()
    b = Bumps(session, name, volume, c, range=range, base_area=base_area, height=height,
              marker_radius=marker_radius, marker_color=marker_color, color_surface=color_surface,
              all_extrema=all_extrema)

    msg = 'Found %d bumps, minimum height %.3g, base area %.3g' % (b.num_atoms, height, base_area)
    session.logger.status(msg, log=True)
    
def register_bumps_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import CenterArg, FloatArg, Color8Arg, StringArg, BoolArg, SaveFileNameArg, ModelsArg
    from chimerax.map import MapArg

    desc = CmdDesc(
        required = [('volume', MapArg)],
        keyword = [('center', CenterArg),
                   ('range', FloatArg),
                   ('base_area', FloatArg),
                   ('height', FloatArg),
                   ('marker_radius', FloatArg),
                   ('marker_color', Color8Arg),
                   ('color_surface', BoolArg),
                   ('name', StringArg),
                   ('all_extrema', BoolArg),],
        required_arguments = ['center'],
        synopsis = 'Mark protrusions in 3D image data',
    )
    register('bumps', desc, bumps, logger=logger)
    desc = CmdDesc(
        optional = [('bumps', ModelsArg)],
        keyword = [('save', SaveFileNameArg),
                   ('signal_map', MapArg)],
        synopsis = 'Output table reporting protrusions in 3D image data'
    )
    register('bumps report', desc, bumps_report, logger=logger)


def bumps_report(session, bumps = None, save = None, signal_map = None):
    '''
    Output a text table of protrusions, tip locations, volumes, heights....

    Parameters
    ----------
    bumps : list of Bumps models
        Previously computed Bumps models to produce table for.
    save : filepath
        Path to save text table.  If not specified then table is output to log.
    signal : Volume
        Report the sum of intensity values from this map for each protrusion.
    '''
    if bumps is None:
        bmodels = session.models.list(type=Bumps)
    else:
        bmodels = [m for m in bumps if isinstance(m, Bumps)]
    text = '\n\n'.join([m.bumps_report(signal_map) for m in bmodels])
    if save:
        f = open(save, 'w')
        f.write(text)
        f.close()
    else:
        session.logger.info(text)
    
from chimerax.markers import MarkerSet
class Bumps(MarkerSet):
    def __init__(self, session, name, volume, center, range = None, base_area = 10.0, height = 1.0,
                 marker_radius = 1.0, marker_color = (100,200,100,255), color_surface = True,
                 all_extrema = False):

        MarkerSet.__init__(self, session, name)

        self.bump_map = volume
        self.bump_center = center
        self.bump_range = range
        self.bump_base_area = base_area
        self.bump_min_height = height
                 
        r, ijk = radial_extrema(volume, center, range)
        indices, size_hvc, mask = protrusion_sizes(r, ijk, volume.data, base_area, height,
                                                   all_extrema, log = session.logger)
        self.bump_mask = mask

        xyz = volume.data.ijk_to_xyz_transform * ijk[indices]
        colors = marker_colors(size_hvc, height, marker_color)
        markers = [self.create_marker(p, rgba, marker_radius, i+1)
                   for i,(p,rgba) in enumerate(zip(xyz,colors))]
            
        for i,m in enumerate(markers):
            h,v,c = size_hvc[i]
            m.bump_id = i + 1
            m.bump_tip_ijk = ijk[indices[i]]
            m.bump_height = h
            m.bump_points = v
            m.bump_connected = c

        session.models.add([self])

        if color_surface:
            color_surface_from_mask(volume, mask)

    def bumps_report(self, signal_map = None):
        lines = ['# ' + self.bump_map.name]
        lines.append('#  %d protrusions, base area %.4g, minimum height %.4g,' %
                     (self.num_atoms, self.bump_base_area, self.bump_min_height)
                     + ' cell center ijk %.4g %.4g %.4g, ' % tuple(self.bump_center)
                     + 'range %s' % ('none' if self.bump_range is None else '%.4g' % self.bump_range))
        columns = '# id    i    j    k   points   height'
        if signal_map:
            columns += '  signal'
            sig_array = signal_map.full_matrix()
        lines.append(columns)
        
        for a in self.atoms:
            if hasattr(a, 'bump_id'):
                i,j,k = a.bump_tip_ijk
                line = '%4d %4d %4d %4d %6d %9.4g' % (a.bump_id, i, j, k, a.bump_points, a.bump_height)
                if signal_map:
                    sig = sig_array[self.bump_mask == a.bump_id].sum()
                    line += ' %8.6g' % sig
                lines.append(line)
        text = '\n'.join(lines)
        return text

def radial_extrema(volume, center_point, max_radius):
    level = volume.maximum_surface_level
    m = volume.full_matrix()
    d = volume.data
    r = radius_map(d, center_point)
    r *= (m >= level)
    rmax = local_maxima(r)
    if max_radius is not None:
        rmax *= (rmax <= max_radius)
    from numpy import array
    ijk = array(rmax.nonzero()[::-1]).transpose()
    return r, ijk

def protrusion_sizes(r, ijk, data, base_area, height,
                     keep_all = False, log = None):
    covered = set()
    rval = r[ijk[:,2],ijk[:,1],ijk[:,0]]
    from numpy import argsort, zeros, int32, array
    ro = argsort(rval)[::-1]
    sizes = []
    indices = []
    mask = zeros(r.shape, int32)
    for c,o in enumerate(ro):
        p = ijk[o]
        if tuple(p) in covered:
            h = v = None
        else:
            voxel_volume = data.step[0]*data.step[1]*data.step[2]
            from math import pow
            voxel_area = pow(voxel_volume, 2/3)
            base_count = base_area / voxel_area
            h, v, con, points = protrusion_height(r, p, base_count)
            covered.update(points)
            v *= voxel_volume
        if keep_all or (h and h >= height and con):
            indices.append(o)
            sizes.append((h,v,con))
            if h is not None:
                pp = array(tuple(points), int32)
                mask[pp[:,2],pp[:,1],pp[:,0]] = len(indices)
        if c % 100 == 0 and log is not None:
            log.status('Protrusion height %d of %d' % (c, len(ijk)))
    return indices, sizes, mask

def radius_map(data, center_point):
    # Compute radius map.
    from numpy import indices, float32, sqrt
    i = indices(data.size[::-1], dtype = float32)
    cijk = data.xyz_to_ijk(center_point)
    step = data.step
    for a in (0,1,2):
        i[a] -= cijk[2-a]
        i[a] *= step[2-a]
    i *= i
    r2 = i.sum(axis=0)
    r = sqrt(r2)
    return r

def local_maxima(a):
    ac = a.copy()
    ksz,jsz,isz = a.shape
    #dirs = ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))	# 6 principal axes directions
    # Use 26 nearest neighbor directions.
    dirs = [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if i != 0 or j != 0 or k != 0]
    for i,j,k in dirs:
        is1,is2 = (slice(0,isz-i),slice(i,isz)) if i >= 0 else (slice(-i,isz),slice(0,isz+i))
        js1,js2 = (slice(0,jsz-j),slice(j,jsz)) if j >= 0 else (slice(-j,jsz),slice(0,jsz+j))
        ks1,ks2 = (slice(0,ksz-k),slice(k,ksz)) if k >= 0 else (slice(-k,ksz),slice(0,ksz+k))
        ac[ks1,js1,is1] *= (a[ks1,js1,is1] > a[ks2,js2,is2])
    return ac

def marker_colors(size_hvc, height, normal_color):
    colors = []
    for h,v,con in size_hvc:
        if h is None:
            color = (255,255,0,255)  # Covered by another peak
        elif not con:
            color = (255,100,100,255)  # Not connected to cell
        elif h < height:
            color = (0,0,255,255)    # Too short
        else:
            color = normal_color
        colors.append(color)
    return colors

def protrusion_height(a, start, base_count):
    s = tuple(start)
    r0 = a[s[2],s[1],s[0]]
    hmax = 0
    border = [(0,s)]
    reached = set()
    reached.add(s)
    prot = set()	# Points that are part of protrusion.
    fill = set()	# Watershed spill points
    bounds = (a.shape[2], a.shape[1], a.shape[0])
    volume = 0
    from heapq import heappop, heappush
    while border and len(border) <= base_count:
        h, b = heappop(border)
        if h >= hmax:
            hmax = h
            volume += 1
            prot.add(b)
            if fill:
                # Only add watershed spill points if the filled basin
                # can be added before base_count is reached.
                prot.update(fill)
                volume += len(fill)
                fill.clear()
        else:
            fill.add(b)
        for s in neighbors(b, bounds):
            if s not in reached:
                reached.add(s)
                r = a[s[2],s[1],s[0]]
                if r > 0:
                    heappush(border, (r0-r,s))
    con = (len(border) > 0)
    return hmax, volume, con, prot

def neighbors(ijk, ijk_max):
    i0,j0,k0 = ijk
    isz,jsz,ksz = ijk_max
    n = [(i0+i,j0+j,k0+k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)
         if ((i != 0 or j != 0 or k != 0)
             and k0+k>=0 and j0+j>=0 and i0+i>=0
             and k0+k<ksz and j0+j<jsz and i0+i<isz)]
    return n

def color_surface_from_mask(volume, mask):
    # Color maps using protrusion mask.
    emask = extend_mask(mask)
    from chimerax.geometry import Place
    tf = Place().matrix
    n = mask.max()
    from numpy import random, uint8, int32, float32, empty
    pcolors = random.randint(0, 255, (n+1,4), dtype = uint8)
    pcolors[:,3] = 255
    from chimerax.map import _map
    for d in volume.surfaces:
        values = empty((len(d.vertices),), float32)
        vijk = volume.data.xyz_to_ijk(d.vertices)
        _map.interpolate_volume_data(vijk, tf, emask, 'nearest', values)
        mi = values.astype(int32)      # Interpolated values are float, not int.
        pcolors[0,:] = d.color
        d.vertex_colors = pcolors[mi]
    
def extend_mask(mask):
    mn = max_neighbors(mask)
    mn *= (mask == 0)
    mn += mask
    return mn

def max_neighbors(a):
    m = a.copy()
    from numpy import maximum
    maximum(m[:-1,:,:], a[1:,:,:], m[:-1,:,:])
    maximum(m[1:,:,:], a[:-1,:,:], m[1:,:,:])
    maximum(m[:,:-1,:], a[:,1:,:], m[:,:-1,:])
    maximum(m[:,1:,:], a[:,:-1,:], m[:,1:,:])
    maximum(m[:,:,:-1], a[:,:,1:], m[:,:,:-1])
    maximum(m[:,:,1:], a[:,:,:-1], m[:,:,1:])
    return m
