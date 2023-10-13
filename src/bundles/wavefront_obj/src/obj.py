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
obj: Wavefront OBJ file format support
======================================

Read and write Wavefront OBJ files.
"""

# -----------------------------------------------------------------------------
#
from chimerax.core.errors import UserError
class OBJError(UserError):
    pass

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class WavefrontOBJ(Surface):
    SESSION_SAVE_DRAWING = True
    clip_cap = True

# -----------------------------------------------------------------------------
#
def read_obj(session, filename, name):
    """Read OBJ model as a surface WavefrontOBJ model.

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        input = filename
        path = getattr(filename, 'name', name)
    else:
        input = open(filename, 'r')
        path = filename

    models = []
    object_name = None
    vertices = []
    texcoords = []
    normals = []
    faces = []
    materials = {}
    cur_material = None
    material = None	# Material before last set of faces
    voffset = 0
    for line_num, line in enumerate(input.readlines()):
        if line_num > 0 and line_num % 100000 == 0:
            session.logger.status('Reading OBJ %s line %d' % (name, line_num))
        if line.startswith('#'):
            continue	# Comment
        fields = line.split()
        if len(fields) == 0:
            continue
        f0, fa = fields[0], fields[1:]
        if f0 == 'v':
            # Vertex
            xyz = [float(x) for x in fa[:3]]
            if len(xyz) != 3:
                raise OBJError('OBJ reader only handles x,y,z vertices, file %s, line %d: "%s"'
                               % (name, line_num, line))
            vertices.append(xyz)
        elif f0 == 'vt':
            # Texture coordinates
            uv = [float(u) for u in fa]
            if len(uv) != 2:
                raise OBJError('OBJ reader only handles u,v texture coordinates, file %s, line %d: "%s"'
                               % (name, line_num, line))
            texcoords.append(uv)
        elif f0 == 'vn':
            # Vertex normal
            n = [float(x) for x in fa]
            if len(n) != 3:
                raise OBJError('OBJ reader only handles x,y,z normals, file %s, line %d: "%s"'
                               % (name, line_num, line))
            normals.append(n)
        elif f0 == 'f':
            # Polygonal face.
            f = _parse_face(fa, line, line_num)
            faces.append(f)
            material = cur_material
        elif f0 == 'o':
            # Object name
            if vertices or object_name is not None:
                oname = object_name if object_name else name
                m = new_object(session, oname, vertices, normals, texcoords, faces, voffset, material)
                models.append(m)
                voffset += len(vertices)
                vertices, normals, texcoords, faces = [], [], [], []
            object_name = line[2:].strip()
        elif f0 == 'mtllib':
            if len(fields) > 1:
                filename = line.split(maxsplit = 1)[1].strip()
                from os.path import join, dirname
                mat_path = join(dirname(path), filename)
                if not _read_materials(mat_path, materials):
                    msg = ('Material file "%s" not found reading OBJ file %s on line %d: %s'
                           % (mat_path, name, line_num, line))
                    session.logger.warning(msg)
        elif f0 == 'usemtl':
            if len(fields) > 1:
                material_name = line.split(maxsplit = 1)[1].strip()
                if material_name in materials:
                    cur_material = materials.get(material_name)
                else:
                    msg = ('Could not find material "%s" referenced in OBJ file %s on line %d: %s'
                           % (material_name, name, line_num, line))
                    session.logger.warning(msg)

    if vertices:
        oname = object_name if object_name else name
        m = new_object(session, oname, vertices, normals, texcoords, faces, voffset, material)
        models.append(m)

    if input != filename:
        input.close()

    if len(models) == 1:
        model = models[0]
    elif len(models) > 1:
        from chimerax.core.models import Model
        model = Model(name, session)
        model.add(models)
    else:
        raise OBJError('OBJ file %s has no objects' % name)

    from os.path import basename
    msg = ('Opened OBJ file %s containing %d objects, %d triangles'
           % (basename(path), len(models), sum(len(m.triangles) for m in models)))
    return [model], msg

# -----------------------------------------------------------------------------
#
def new_object(session, object_name, vertices, normals, texcoords, faces, voffset, material):

    if len(faces) > 100000:
        session.logger.status('Creating OBJ model %s with %d faces' % (object_name, len(faces)))

    if _need_vertex_split(faces):
        # Texture coordinates or normals do not match vertices order.
        # Need to make additional vertices if a vertex has different texture
        # coordinates or normals in different faces.
        vertices, normals, texcoords, triangles = _split_vertices(vertices, normals, texcoords, faces)
    else:
        triangles = faces
        
    if len(vertices) == 0:
        raise OBJError('OBJ file has no vertices')
    if len(normals) > 0 and len(normals) != len(vertices):
        raise OBJError('OBJ file has different number of normals (%d) and vertices (%d)'
                       % (len(normals), len(vertices)))
    if len(texcoords) > 0 and len(texcoords) != len(vertices):
        raise OBJError('OBJ file has different number of texture coordinates (%d) and vertices (%d)'
                       % (len(texcoords), len(vertices)))

    from chimerax.core.models import Surface
#    model = Surface(object_name, session)
#    model.SESSION_SAVE_DRAWING = True
#    model.clip_cap = True
    model = WavefrontOBJ(object_name, session)

    from numpy import array, float32, int32, uint8
    if texcoords:
        model.texture_coordinates = array(texcoords, float32)
    ta = array(triangles, int32)
    if voffset > 0:
        ta -= voffset
    ta -= 1	# OBJ first vertex index is 1 while model first vertex index is 0
    va = array(vertices, float32)
    if normals:
        na = array(normals, float32)
    else:
        # na = None
        from chimerax.surface import calculate_vertex_normals
        na = calculate_vertex_normals(va, ta)
    model.set_geometry(va, na, ta)

    model.color = array((170,170,170,255), uint8)
    if material and 'texture' in material and texcoords:
        filename = material['texture']
        from chimerax.surface.texture import image_file_as_rgba
        try:
            rgba = image_file_as_rgba(filename)
        except Exception as e:
            session.logger.warning(str(e))  # Warn if texture does not exist.
        else:
            from chimerax.graphics import Texture
            model.texture = Texture(rgba)
            model.opaque_texture = (rgba[:,:,3] == 255).all()
            model.color = array((255,255,255,255), uint8)
        
    return model

# -----------------------------------------------------------------------------
#  Parse face vertex/texture/normal indices.
#
#	f 1 2 3
#	f 1/1 2/2 3/3
#	f 1/1/1 2/2/2 3/3/3
#
def _parse_face(fields, line, line_num):
    if len(fields) != 3:
        raise OBJError('OBJ reader only handles triangle faces, line %d: "%s"'
                       % (line_num, line))
    try:
        face = [_parse_face_corner(f) for f in fields]
    except ValueError:
        raise OBJError('OBJ reader could not parse face, non-integer field, line %d: "%s"'
                       % (line_num, line))
    return face

# -----------------------------------------------------------------------------
# Parse vertex/texture/normal.  If only one index or all match return an integer
# otherwise return a tuple.
#
def _parse_face_corner(corner):
    vtn = tuple((None if s == '' else int(s)) for s in corner.split('/'))
    ni = len(vtn)
    if ni == 1:
        return vtn[0]
    if (vtn[1] is None or vtn[1] == vtn[0]) and (ni < 3 or vtn[2] is None or vtn[2] == vtn[0]):
        return vtn[0]
    return vtn

# -----------------------------------------------------------------------------
#
def _need_vertex_split(faces):
    for f in faces:
        for vtn in f:
            if not isinstance(vtn, int):
                return True
    return False

# -----------------------------------------------------------------------------
#
def _split_vertices(vertices, normals, texcoords, faces):
    triangles = []
    cvertex = {}
    v = []
    nv = 0
    tc = []
    n = []
    for face in faces:
        t = []
        for corner in face:
            if corner in cvertex:
                vi = cvertex[corner]
            else:
                nv += 1
                vi = nv
                cvertex[corner] = vi
                if isinstance(corner, int):
                    vo = tco = no = corner
                else:
                    vo = corner[0]
                    tco = corner[1] if len(corner) >= 2 else None
                    no = corner[2] if len(corner) >= 3 else None
                v.append(vertices[vo-1])
                if tco is not None and texcoords:
                    tc.append(texcoords[tco-1])
                if no is not None and normals:
                    n.append(normals[no-1])
            t.append(vi)
        triangles.append(t)
    if len(tc) > 0 and len(tc) < len(v):
        raise OBJError('Some faces specified texture coordinates and others did not.')
    if len(n) > 0 and len(n) < len(v):
        raise OBJError('Some faces specified normals and others did not.')
    return v, n, tc, triangles

# -----------------------------------------------------------------------------
#
def _read_materials(filename, materials):
    try:
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
    except IOError:
        return False
    mat_name = None
    for line in lines:
        if line.startswith('#'):
            continue
        fields = line.split(maxsplit = 1)
        if len(fields) < 2:
            continue
        f0 = fields[0]
        f1 = fields[1].rstrip()	# Remove newline
        if f0 == 'newmtl':
            materials[f1] = {}
            mat_name = f1
        elif f0 == 'map_Kd' and mat_name is not None:
            from os.path import join, dirname
            image_path = join(dirname(filename), f1)
            materials[mat_name]['texture'] = image_path
    return True

# -----------------------------------------------------------------------------
#
def write_obj(session, filename, models, obj_to_unity = True, single_object = False):
    if models is None:
        models = session.models.list()

    # Collect all drawing children of models.
    drawings = set()
    for m in models:
        if not m in drawings:
            for d in m.all_drawings():
                drawings.add(d)
            
    # Collect geometry, not including children, handle instancing
    geom = []
    for d in drawings:
        va, na, tca, ta = d.vertices, d.normals, d.texture_coordinates, d.masked_triangles
        if va is not None and ta is not None and d.display and d.parents_displayed:
            pos = d.get_scene_positions(displayed_only = True)
            if len(pos) > 0:
                geom.append((full_name(d), va, na, tca, ta, pos))

    if single_object:
        from chimerax.surface import combine_geometry_xvntctp
        va, na, tca, ta = combine_geometry_xvntctp(geom)
        geom = [(None, va, na, tca, ta, None)]

    # Write 80 character comment.
    from chimerax import app_dirs as ad
    version  = "%s %s version: %s" % (ad.appauthor, ad.appname, ad.version)
    created_by = '# Created by %s\n' % version

    file = open(filename, 'w')

    # Write comment
    file.write(created_by)

    voffset = 0
    for name, va, na, tca, ta, pos in geom:
        vcount = write_object(file, name, va, na, tca, ta, voffset, pos, obj_to_unity)
        voffset += vcount

    file.close()

# -----------------------------------------------------------------------------
#
def full_name(drawing):
    return ' '.join(d.name for d in drawing.drawing_lineage[1:])

# -----------------------------------------------------------------------------
#
def write_object(file, name, va, na, tca, ta, voffset, pos, obj_to_unity):

    # Write object name
    if name is not None:
        ascii_name = name.encode('ascii', 'replace').decode('ascii')  # Replace non-ascii characters with ?
        file.write('o %s\n' % ascii_name)

    if pos is not None and not pos.is_identity():
        # Expand out positions including instancing.
        from chimerax.surface import combine_geometry_xvntctp
        va, na, tca, ta = combine_geometry_xvntctp([(name, va, na, tca, ta, pos)])

    # Write vertices
    file.write('\n'.join(('v %.5g %.5g %.5g' % tuple(xyz)) for xyz in va))
    file.write('\n')

    # Write texture coordinates
    if tca is not None:
        file.write('\n'.join(('vt %.5g %.5g' % tuple(uv)) for uv in tca))
        file.write('\n')

    # Write normals
    if na is not None:
        file.write('\n'.join(('vn %.5g %.5g %.5g' % tuple(xyz)) for xyz in na))
        file.write('\n')

    # Write triangles
    # For Unity3D 2017.1 to import OBJ texture coordinates, must specify their indices
    # even though they are the same as the vertex indices.
    vo = voffset+1
    if not obj_to_unity:
        tlines = [('f %d %d %d' % (v0+vo,v1+vo,v2+vo)) for v0,v1,v2 in ta]
    elif na is None and tca is None:
        tlines = [('f %d %d %d' % (v0+vo,v1+vo,v2+vo)) for v0,v1,v2 in ta]
    elif tca is None:
        tlines = [('f %d/%d %d/%d %d/%d' % (v0+vo,v0+vo,v1+vo,v1+vo,v2+vo,v2+vo)) for v0,v1,v2 in ta]
    elif na is None:
        tlines = [('f %d//%d %d//%d %d//%d' % (v0+vo,v0+vo,v1+vo,v1+vo,v2+vo,v2+vo)) for v0,v1,v2 in ta]
    else:
        tlines = [('f %d/%d/%d %d/%d/%d %d/%d/%d' % (v0+vo,v0+vo,v0+vo,v1+vo,v1+vo,v1+vo,v2+vo,v2+vo,v2+vo)) for v0,v1,v2 in ta]
    file.write('\n'.join(tlines))
    file.write('\n')

    return len(va)
