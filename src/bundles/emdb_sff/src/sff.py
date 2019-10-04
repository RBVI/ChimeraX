# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"""
EMDB SFF: Segmentation file reader
==================================

Read EMDB SFF segmentation files
"""
def read_sff(session, path):
    """
    Create a model that is either polygon traces or a volume index map.
    """

    from sfftk.schema import SFFSegmentation
    seg = SFFSegmentation(path)

    report_segmentation_info(seg)
    lmodels = lattice_models(session, seg)
    mmodels = mesh_models(session, seg)

    models = lmodels + mmodels
    
    msg = 'Read segmentation file %s' % path
    return models, msg

def report_segmentation_info(seg):
    print ('seg name', seg.name, 'software', seg.software, 'num transforms', len(seg.transforms),
           'num segments', len(seg.segments), 'num lattices', len(seg.lattices), 'sff version', seg.version)
    for tf in seg.transforms:
        print('transform', tf.id, 'data', tf.data_array)

    for segment in seg.segments:
        v = segment.volume
        print ('segment id', segment.id, 'num meshes', len(segment.meshes),
               'descrip', segment.biologicalAnnotation,
               'color', segment.colour, 'num shapes', len(segment.shapes),
               'parent', segment.parentID, 'lattice id', v.latticeId,
               'transform id', v.transformId, 'value', v.value)

        for i,m in enumerate(segment.meshes):
            print('mesh', i+1, 'nv', m.numVertices, 'npoly', m.numPolygons, 'transf id', m.transformId)

def lattice_models(session, seg):
    lattice_segs = {}	# Map lattice id to dictionary mapping segment index to (descrip, color)
    for segment in seg.segments:
        v = segment.volume
        lseg = lattice_segs.setdefault(v.latticeId, {})
        if v.value is not None:
            lseg[int(v.value)] = (segment.biologicalAnnotation, segment.colour)

    scale, shift = guess_scale_and_shift(seg)
    
    models = []
    for lattice in seg.lattices:
        dd = lattice.decode()
        d = lattice.data	# Docs say number array, but emd 1547 gives bytes
        print('lattice', lattice.id, 'shape', d.shape, 'value type', lattice.mode)
        name = seg.name + ' segmentation'
        from chimerax.map.data import ArrayGridData
        g = ArrayGridData(d, step=scale, origin=shift, name = name)
        from chimerax.map import volume_from_grid_data
        v = volume_from_grid_data(g, session)
        if lattice.id in lattice_segs:
            v.segments = lseg = lattice_segs[lattice.id]
            set_segmentation_image_colors(v, lseg)
            # Make a surface for each segment.
            surfs = [segment_surface(v, sindex, descrip, color)
                     for sindex, (descrip, color) in lseg.items()]
            from chimerax.core.models import Model
            surf_group = Model('segment surfaces', v.session)
            surf_group.add(surfs)
            v.add([surf_group])
        # TODO: Don't see how to get the transform id for the lattice.
        #  EMD 1547 segmentation has two transforms, identity and the
        #  map transform (2.8A voxel size, shift to center) but I didn't
        #  find any place where transform 1 is associated with the lattice.
        #  Appears it should be in segment.volume.transformId but this
        #  attribute is optional and is None in this case.
        models.append(v)

    return models

def guess_scale_and_shift(seg):
    # Segmentation for emd 1547 has a non-identity transform but
    # the lattices don't reference it.  Guess by using any non-identity
    # transform
    scale = (1,1,1)
    origin = (0,0,0)
    for tf in seg.transforms:
        tfa = tf.data_array
        if (tfa != ((1,0,0,0),(0,1,0,0),(0,0,1,0))).any():
            if (tfa[0,1] == 0 and tfa[0,2] == 0 and tfa[1,2] == 0
                and tfa[1,0] == 0 and tfa[2,0] == 0 and tfa[2,1] == 0):
                scale = (tfa[0,0], tfa[1,1], tfa[2,2])
                origin = (tfa[0,3], tfa[1,3], tfa[2,3])
    return scale, origin
    
def set_segmentation_image_colors(v, seg_colors):
    sindices = [si for si in seg_colors.keys() if si > 0]
    sindices.sort()
    levels = []
    colors = []
    for si in sindices:
        levels.extend([(si-0.1, 0.99), (si+0.1, 0.99)])
        c = segment_color(seg_colors[si][1], float=True)
        colors.extend((c,c))
    v.image_levels = levels
    v.image_colors = colors
    v.transparency_depth = 0.1		# Make more opaque
    v.set_display_style('image')
    v.set_parameters(projection_mode = '3d')

def segment_surface(v, sindex, descrip, color):
    from chimerax.core.models import Surface
    s = Surface('segment %d %s' % (sindex, descrip), v.session)
    va, na, ta = segment_surface_geometry(v, sindex)
    s.set_geometry(va, na, ta)
    s.color = segment_color(color)
    return s

def segment_surface_geometry(v, sindex):    
    m = v.full_matrix()
    from numpy import uint8
    mask = (m == sindex).astype(uint8)
    from chimerax.map import contour_surface
    varray, tarray, narray = contour_surface(mask, 0.5, cap_faces = True, calculate_normals = True)
    tf = v.data.ijk_to_xyz_transform
    # Transform vertices and normals from index coordinates to model coordinates
    tf.transform_points(varray, in_place = True)
    tf.transform_normals(narray, in_place = True)
    return varray, narray, tarray

def segment_color(sff_color, float = False):
    c = sff_color
    from chimerax.core.colors import rgba_to_rgba8
    rgba = (c.red, c.green, c.blue, c.alpha)
    return rgba if float else rgba_to_rgba8(rgba)

def mesh_models(session, seg):
    surfs = []
    from chimerax.core.models import Surface
    for segment in seg.segments:
        geoms = [mesh_geometry(mesh, seg) for mesh in segment.meshes]
        if len(geoms) == 0:
            continue
        va,na,ta = combine_geometry(geoms)
        s = Surface('mesh %d' % segment.id, session)
        s.set_geometry(va, na, ta)
#        s.display_style = s.Mesh
#        s.use_lighting = False
        s.color = segment_color(segment.colour)
        
        surfs.append(s)
    return surfs
    
def mesh_geometry(mesh, seg):
    # TODO: SFF format data structure mix vertices and normals, calling both vertices -- a nightmare.
    #   Semantics of which normal belong with which vertices unclear (consecutive in polygon?).
    #   Efficiency reading is horrible.  Ask Paul K to make separate vertex and normal lists.
    nv = mesh.numVertices // 2
    from numpy import empty, float32
    va = empty((nv,3), float32)
    na = empty((nv,3), float32)
    for i, v in enumerate(mesh.vertices):
        vid = v.vID
        if vid != i:
            raise ValueError('Require mesh vertices be numbers consecutively from 0, got vertex id %d in position %d' % (vid, i))
        d = v.designation # 'surface' or 'normal'
        if d == 'surface':
            if vid % 2 == 1:
                raise ValueError('Require odd mesh indices to be normals, got a vertex at position %d' % vid)
            va[vid//2] = v.point
        elif d == 'normal':
            if vid % 2 == 0:
                raise ValueError('Require even mesh indices to be vertices, got a normal at position %d' % vid)
            na[vid//2] = v.point
        else:
            raise ValueError('Vertex %d designation "%s" is not "surface" or "normal"' % (v.vID, d))

    '''
    vids = list(set(v.vID for v in mesh.vertices if v.designation == 'surface'))
    vids.sort()
    print ('vertex ids', vids[:3], 'num', len(vids), 'last', vids[-1])
    '''

    if mesh.transformId is None:
        from chimerax.core.geometry import Place, scale
#        transform = scale((160,160,160)) * Place(seg.transforms[0].data_array)
        transform = Place(seg.transforms[0].data_array) * scale((160,160,160))
    else:
        transform = transform_by_id(seg, mesh.transformId)
    transform.transform_points(va, in_place = True)
    transform.transform_normals(na, in_place = True)

    tri = []
    for p in mesh.polygons:
#        print ('poly', len(p.vertex_ids), p.vertex_ids[:6])
        t = tuple(vid//2 for vid in p.vertex_ids if vid % 2 == 0)
        if len(t) != 3:
            raise ValueError('Require polygons to be single triangles, got polygon with %d vertices' % len(t))
        tri.append(t)
        '''
        last_vid = None
        for vid in p.vertex_ids:
            if vid % 2 == 0:
                if last_vid is not None:
#                    tri.append((vid//2,last_vid//2))
                    tri.append((vid//2,last_vid//2,vid//2))
                last_vid = vid
        first_vid = p.vertex_ids[0]
        tri.append((last_vid//2,first_vid//2,last_vid//2))
        '''
    from numpy import array, int32
    ta = array(tri, int32)

    return va,na,ta

def transform_by_id(seg, tf_id):
    from chimerax.core.geometry import Place, scale
    for tf in seg.transforms:
        if tf.id == tf_id:
            return scale((160,160,160)) * Place(tf.data_array)
    return Place()
    
def combine_geometry(geoms):
    from numpy import concatenate
    cva = concatenate([va for va,na,ta in geoms])
    cna = concatenate([na for va,na,ta in geoms])
    cta = concatenate([ta for va,na,ta in geoms])
    voffset = 0
    toffset = 0
    for va,na,ta in geoms:
        cta[toffset:toffset+len(ta)] += voffset
        voffset += len(va)
        toffset += len(ta)
    return va,na,ta
