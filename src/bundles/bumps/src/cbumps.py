#
# Adds command cbumps to identify protrusions on surfaces of cells made for En Cai.
#
# Uses convexity to identify bumps.  Has a variety of features to obtain outward facing bumps.
#
# 1) Filter out small area connected blobs.
# 2) Filter out inward facing bumps by checking if average bump patch normal faces away from a center marker.
# 3) Filter out bump patches with centroid too far from center marker to avoid other nearby cells.
# 4) Filter out bumps of small surface area.
# 5) Separate overlapped bumps if both have sufficient surface area.
#
# Also can extract signal on bumps (e.g. membrane protein imaging) on each patch and on non-patches,
# giving average and standard deviation.
#

def cbumps(session, surfaces, convexity_minimum = 0.3, area = None,
           dust = None, radius = None, center = None,
           outward = None, smoothing_iterations = 5, signal_map = None, output = None):
    '''
    Identify bumps on cell surfaces using convexity and various filters.

    Parameters
    ----------
    surfaces : Surface list
    convexity_minimum : float
      Instead of coloring by convexity value, color each connected patch with convexity
      above the specified value a unique color.
    area : float
      Exclude bumps with less than this surface area.
    dust : float
      Do not identify bumps on connected surface components with less than this area.
    radius : float
      Do not identify bumps where patch centroid distance from center marker is greater than this value.
    center : Center
        Point which is the cell center for radius cutoff.
    outward : float
      Exclude patches where fewer than this fraction of patch vertex normals point out from center.
    smoothing_iterations : int
      Convexity values are averaged with neighbor vertices connected by an edge.
      This value specifies how many rounds of smoothing to perform.  Default 5.
    signal_map : Volume
        Report the area weighted average intensity values from this map for each surface patch.
    output : string
        File path to write output from signal_map option.
    '''
    for s in surfaces:
        if s.empty_drawing():
            continue
        va,ta = s.vertices, s.triangles
        if dust is not None:
            from chimerax.surface.dust import Blob_Masker
            bmask = Blob_Masker(va, ta)
            ti = bmask.triangle_mask('area', dust)
            ta = ta[ti]
        from chimerax.surface import vertex_convexity
        c = vertex_convexity(va, ta, smoothing_iterations)
        vc = s.get_vertex_colors(create = True)
        patches = _patches(convexity_minimum, c, ta)
        if area is not None:
            from chimerax.surface import surface_area
            patches = [(vi,ti) for vi,ti in patches
                       if surface_area(va, ta[ti]) >= area]
        c0 = None if center is None else s.position.inverse() * center.scene_coordinates()
        if radius is not None and center is not None:
            from chimerax.geometry import distance
            patches = [(vi,ti) for vi,ti in patches
                       if distance(va[vi].sum(axis=0)/len(vi), c0) <= radius]
        if outward is not None and center is not None:
            patches = _outward_facing_patches(patches, va, s.normals, c0, outward)
        from chimerax.core.colors import random_colors
        rc = random_colors(len(patches))
        for i,(vi,ti) in enumerate(patches):
            vc[vi] = rc[i]
        s.vertex_colors = vc
        if signal_map is not None:
            msg = _report_patch_intensities(patches, va, ta, signal_map, s.position)
            if output:
                f = open(output, 'w')
                f.write(msg)
                f.close()
            else:
                session.logger.info(msg)
        msg = ('Convexity %.3g - %.3g, mean %.3g, std deviation %.3g at %d vertices of %s'
               % (c.min(), c.max(), c.mean(), c.std(), len(va), s.name))
        from chimerax.core.models import Model
        if isinstance(s, Model):
            msg += ' ' + s.id_string
        session.logger.status(msg, log = True)

def register_cbumps_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfacesArg, IntArg, FloatArg, CenterArg, SaveFileNameArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        keyword = [('convexity_minimum', FloatArg),
                   ('area', FloatArg),
                   ('dust', FloatArg),
                   ('radius', FloatArg),
                   ('center', CenterArg),
                   ('outward', FloatArg),
                   ('smoothing_iterations', IntArg),
                   ('signal_map', MapArg),
                   ('output', SaveFileNameArg),],
        synopsis = 'find surface bumps')
    register('cbumps', desc, cbumps, logger=logger)

def _patches(threshold, vertex_values, triangles):
    '''
    Find connected surface patches with vertex values above a specified threshold.
    Return list of vertex index arrays, one array for each connected patch.
    '''
    vset = set((vertex_values >= threshold).nonzero()[0])
    tabove = [v1 in vset and v2 in vset and v3 in vset for v1,v2,v3 in triangles]
    ta = triangles[tabove]
    from chimerax.surface import connected_pieces
    patches = connected_pieces(ta)
    return patches

def _outward_facing_patches(patches, vertices, normals, center, outward_frac):
    pout = []
    for vi,ti in patches:
        pc = vertices[vi].sum(axis=0)/len(vi)	# Center of patch.
        rv = pc - center			# Radial outward direction
        from numpy import dot
        nout = (dot(normals[vi], rv) > 0).sum()	# Number of normals pointing outward
        if nout >= outward_frac * len(vi):
            pout.append((vi,ti))
    return pout

def _report_patch_intensities(patches, va, ta, signal_map, surf_position):
    lines = ['# Area average intensities of map %s for %d patches' % (signal_map.name, len(patches)),
             '# pnum area intensity']
    from chimerax.surface import surface_area, vertex_areas
    from chimerax.geometry import inner_product
    varea = vertex_areas(va, ta)
    for i, (vi,ti) in enumerate(patches):
        pva, pta = va[vi], ta[ti]
        area = surface_area(va, pta)
        vsig = signal_map.interpolated_values(pva, surf_position)
        pvarea = varea[vi]
        vasig = inner_product(pvarea, vsig)
        sig = vasig / pvarea.sum()
        lines.append('%d %.6g %.6g' % (i+1, area, sig))
    return '\n'.join(lines)
