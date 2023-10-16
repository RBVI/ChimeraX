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

"""
EMDB SFF: Segmentation file reader
==================================

Read EMDB SFF segmentation files
"""
def read_sff(session, path, max_surfaces = 100, debug = False):
    """
    Create a model that is either polygon traces or a volume index map.
    """

    from sfftkrw import SFFSegmentation
    seg = SFFSegmentation.from_file(path)

    if debug:
        info = segmentation_info(seg)
        session.logger.info(info)
    lmodels = lattice_models(session, seg, max_surfaces)
    mmodels = mesh_models(session, seg)

    models = lmodels + mmodels
    
    msg = 'Read segmentation file %s\n%s' % (path, seg.name)
    if len(models) == 0:
        msg += '\nSegmentation file contains no lattices or meshes.'
    return models, msg

def segmentation_info(seg, indent = '....'):
    info = [
        f'sff version {seg.version}',
        f'seg name {seg.name}',
        f'software {[software.name for software in seg.software_list]}',
    ]

    info.append(f'num transforms {len(seg.transforms)}')
    info.extend([f'transform {tf.id}\ndata {tf.data_array}' for tf in seg.transforms])

    info.append(f'num segments {len(seg.segments)}')
    for segment in seg.segments:
        v = segment.three_d_volume
        info.extend(
            [f'{indent}segment id {segment.id}',
             f'{indent}descrip {segment.biological_annotation.name}',
             f'{indent}color {segment.colour}',
             f'{indent}num shapes {len(segment.shape_primitive_list)}',
             f'{indent}parent {segment.parent_id}',
             ])
        if v is None:
            info.append(f'{indent}no volume')
        else:
            info.extend(
                [f'{indent}volume lattice id {v.lattice_id}',
                 f'{indent}volume transform id {v.transform_id}',
                 f'{indent}volume value {v.value}',])
        if segment.mesh_list:
            info.append(f'{indent}num meshes {len(segment.mesh_list)}')
            info.extend(
                [f'{indent}{indent}mesh {i+1}, num vertices {len(m.vertices)}, num triangles {len(m.triangles)}, transform id {m.transform_id}'
                 for i,m in enumerate(segment.mesh_list)])

    info.append(f'num lattices {len(seg.lattices)}')
    for lattice in seg.lattices:
        info.extend(
            [f'{indent}lattice {lattice.id}',
             f'{indent}shape {lattice.data_array.shape}',
             f'{indent}value type {lattice.mode}',])

    return '\n'.join(info)

def lattice_models(session, seg, max_surfaces = 100):
    # Map lattice id to dictionary mapping segment index to (descrip, color)
    lattice_segs = {}
    for segment in seg.segments:
        v = segment.three_d_volume
        if v is not None:
            lseg = lattice_segs.setdefault(v.lattice_id, {})
            if v.value is not None:
                lseg[int(v.value)] = (segment.biological_annotation, segment.colour)

    scale, shift = guess_scale_and_shift(seg)

    # Create Volume model of segment indices for each lattice.
    models = []
    lattices = seg.lattices
    for i,lattice in enumerate(lattices):
        d = lattice.data_array	# Docs say number array, but emd 1547 gives bytes
        name = 'region map' if len(lattices) == 1 else 'region map %d' % i
        from chimerax.map_data import ArrayGridData
        g = ArrayGridData(d, step=scale, origin=shift, name = name)
        from chimerax.map import volume_from_grid_data
        v = volume_from_grid_data(g, session, open_model = False)
        v.display = False
        if lattice.id in lattice_segs:
            v.segments = lseg = lattice_segs[lattice.id]
            set_segmentation_image_colors(v, lseg)
            # Make a surface for each segment.
            regions = list(lseg.items())
            regions.sort()
            surfs = [segment_surface(v, sindex, descrip, color)
                     for sindex, (descrip, color) in regions[:max_surfaces]]
            if surfs:
                ns, nr = len(surfs), len(regions)
                sname = ('%d surfaces' % ns) if ns == nr else ('%d of %d surfaces' % (ns,nr))
                from chimerax.core.models import Model
                surf_group = Model(sname, v.session)
                surf_group.add(surfs)
                models.append(surf_group)
        # TODO: Don't see how to get the transform id for the lattice.
        #  EMD 1547 segmentation has two transforms, identity and the
        #  map transform (2.8A voxel size, shift to center) but I didn't
        #  find any place where transform 1 is associated with the lattice.
        #  Appears it should be in segment.three_d_volume.transform_id but this
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
    from chimerax.surface import combine_geometry_vnt
    for segment in seg.segments:
        if segment.mesh_list is None:
            continue
        geoms = [mesh_geometry(mesh, seg) for mesh in segment.mesh_list]
        if len(geoms) == 0:
            continue
        va,na,ta = combine_geometry_vnt(geoms)
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
    from numpy import int32
    va,ta = mesh.vertices.data_array, mesh.triangles.data_array.astype(int32)
    n = mesh.normals
    if n is None:
        from chimerax.surface import calculate_vertex_normals
        na = calculate_vertex_normals(va, ta)
    else:
        na = n.data_array

    tf_id = 0 if mesh.transform_id is None else mesh.transform_id
    if tf_id is not None:
        transform = transform_by_id(seg, tf_id)
        transform.transform_points(va, in_place = True)
        transform.transform_normals(na, in_place = True)
        
    return va,na,ta

def transform_by_id(seg, tf_id):
    from chimerax.geometry import Place
    for tf in seg.transforms:
        if tf.id == tf_id:
            return Place(tf.data_array)
    return Place()
