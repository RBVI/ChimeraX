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
gltf: glTF file format support
==============================

Read and write glTF 3d scene files.
"""

# -----------------------------------------------------------------------------
#
from chimerax.core.errors import UserError
class glTFError(UserError):
    pass

GLTF_TRIANGLES = 4

# -----------------------------------------------------------------------------
#
from chimerax.core import generic3d
class gltfModel(generic3d.Generic3DModel):
    clip_cap = True

# -----------------------------------------------------------------------------
#
def read_gltf(session, filename, name):
    """Read glTF model as a gltfModel.

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        input = filename
    else:
        input = open(filename, 'r')

    from numpy import fromstring, uint32, float32, array, uint8
    magic = fromstring(input.read(4), uint32)[0]        # magic number
    if magic != 0x46546c67:
        raise glTFError('glTF file does not start with magic number 0x46546c67, got %x' % magic)

    version = fromstring(input.read(4), uint32)[0]        # version number
    if version != 2:
        raise glTFError('Require glTF version 2, got version %d' % version)

    length = fromstring(input.read(4), uint32)[0]        # file length in bytes

    chunks = read_chunks(input)
    if len(chunks) != 2:
        raise glTFError('glTF expected 2 chunks, got %d' % len(chunks))
    if chunks[0][0] != 'JSON' or chunks[1][0] != 'BIN':
        raise glTFError('glTF expected JSON and BIN chunks, got %s' % ' and '.join(c[0] for c in chunks))

    if input != filename:
        input.close()

    jsonc, binc = chunks[0][1].decode('utf-8'), chunks[1][1]
    import json
    j = json.loads(jsonc)

    for attr in ('accessors', 'bufferViews', 'meshes'):
        if attr not in j:
            raise glTFError('glTF JSON contains no "%s": %s' % (attr, str(j)))
    accessors, buffer_views, meshes = j['accessors'], j['bufferViews'], j['meshes']

    ba = buffer_arrays(accessors, buffer_views, binc)
        
    geom = []
    from numpy import int32
    for m in meshes:
        if 'primitives' not in m:
            raise glTFError('glTF mesh has no "primitives": %s' % str(j))
        for p in m['primitives']:
            if 'mode' in p and p['mode'] != GLTF_TRIANGLES:
                raise glTFError('glTF reader only handles triangles, got mode %d' % p['mode'])
            if 'indices' not in p:
                raise glTFError('glTF missing "indices" in primitive %s' % str(p))
            ta = ba[p['indices']]
            if len(ta.shape) == 1:
                ta = ta.reshape((len(ta)//3,3))
            ta = ta.astype(int32, copy=False)
            if 'attributes' not in p:
                raise glTFError('glTF missing "attributes" in primitive %s' % str(p))
            pa = p['attributes']
            if 'POSITION' not in pa:
                raise glTFError('glTF missing "POSITION" attribute in primitive %s' % str(p))
            va = ba[pa['POSITION']]
            if 'NORMAL' in pa:
                na = ba[pa['NORMAL']]
            else:
                from chimerax.core import surface
                na = surface.calculate_vertex_normals(va, ta)
            if 'COLOR_0' in pa:
                vc = ba[pa['COLOR_0']]
            else:
                vc = None
            geom.append((va,na,vc,ta))

    m = gltfModel(name, session)
    if len(geom) == 1:
        set_geometry(m, *geom[0])
    else:
        mlist = []
        for i,g in enumerate(geom):
            mm = gltfModel('mesh %d' % (i+1) , session)
            set_geometry(mm, *g)
            mlist.append(mm)
        m.add(mlist)

    return [m], ('Opened glTF file containing %d meshes' % len(geom))

# -----------------------------------------------------------------------------
#
def read_chunks(input):
    chunks = []
    from numpy import fromstring, uint32
    while True:
        l4 = input.read(4)
        if len(l4) == 0:
            break
        clength = fromstring(l4, uint32)[0]        # chunk length
        ctype = input.read(4).decode('utf-8').replace('\x00','')
        chunks.append((ctype, input.read(clength)))
    return chunks

# -----------------------------------------------------------------------------
#
def buffer_arrays(accessors, buffer_views, binc):
    balist = []
    from numpy import float32, uint32, uint16, int16, uint8, frombuffer
    value_type = {5126:float32, 5125:uint32, 5123:uint16, 5122:int16, 5120:uint8}
    for a in accessors:
        ibv = a['bufferView']	# index into buffer_views
        bv = buffer_views[ibv]
        bo = bv['byteOffset']	# int
        bl = bv['byteLength']	# int
        ct = a['componentType']	# 5123 = uint16, 5126 = float32, 5120 = uint8
        dtype = value_type[ct]
        ba = frombuffer(binc[bo:bo+bl], dtype)
        atype = a['type']		# "VEC3", "SCALAR"
        if atype == 'VEC3':
            ba = ba.reshape((len(ba)//3, 3))
        elif atype == 'VEC4':
            ba = ba.reshape((len(ba)//4, 4))
        elif atype == 'SCALAR':
            pass
        else:
            raise glTFError('glTF accessor type is not VEC3 or SCALAR, got %s' % atype)
        balist.append(ba)
    return balist
        
# -----------------------------------------------------------------------------
#
def set_geometry(model, va, na, vc, ta):
    model.vertices = va
    model.triangles = ta
    if na is not None:
        model.normals = na
    if vc is not None:
        model.vertex_colors = colors_to_uint8(vc)

# -----------------------------------------------------------------------------
#
def colors_to_uint8(vc):
    from numpy import empty, uint8, float32
    nc,ni = len(vc), vc.shape[1]
    if vc.dtype == uint8:
        if ni == 3:
            c = empty((nc,4),uint8)
            c[:,:3] = vc
            c[:,3] = 255
        elif ni == 4:
            c = vc
    elif vc.dtype == float32:
        c = empty((nc,4),uint8)
        c[:,:ni] = (vc[:,:ni]*255).astype(uint8)
        if ni == 3:
            c[:,3] = 255
    else:
        raise glTFError('glTF colors, only handle float32 and uint8, got %s' % str(vc.dtype))

    return c

# -----------------------------------------------------------------------------
#
def write_gltf(session, filename, models):
    if models is None:
        models = session.models.list()

    geom = drawing_geometries(models)

    # Write 80 character comment.
    from chimerax import app_dirs as ad
    app_ver  = "%s %s version: %s" % (ad.appauthor, ad.appname, ad.version)

    prim, buffer_views, accessors, binc = geometry_buffers(geom)
    h = {
        'asset': {'version': '2.0', 'generator': app_ver},
        'scene': 0,
        'scenes': [{'nodes':[0]}],
        'nodes':[{'mesh':0}],
        'meshes': [{'primitives': prim}],
        'accessors': accessors,
        'bufferViews': buffer_views,
    }
    import json
    json_text = json.dumps(h).encode('utf-8')
    from numpy import uint32
    clen = to_bytes(len(json_text), uint32)
    ctype = b'JSON'
    json_chunk = b''.join((clen, ctype, json_text))
    
    blen = to_bytes(len(binc), uint32)
    btype = b'BIN\x00'
    bin_chunk = b''.join((blen, btype, binc))
    
    magic = to_bytes(0x46546c67, uint32)
    version = to_bytes(2, uint32)
    length = to_bytes(12 + len(json_chunk) + len(bin_chunk), uint32)

    file = open(filename, 'wb')
    for b in (magic, version, length, json_chunk, bin_chunk):
        file.write(b)
    file.close()

# -----------------------------------------------------------------------------
#
def drawing_geometries(models):
    # Collect all drawing children of models.
    drawings = set()
    for m in models:
        if not m in drawings:
            for d in m.all_drawings():
                drawings.add(d)
            
    # Collect geometry, not including children, handle instancing
    geom = []
    for d in drawings:
        if d.display and d.parents_displayed:
            va, na, vc, ta = d.vertices, d.normals, d.vertex_colors, d.masked_triangles
            if va is not None and ta is not None:
                pos = d.get_scene_positions(displayed_only = True)
                if pos.is_identity():
                    geom.append((va, na, vc, ta))
                elif len(pos) > 1:
                    # TODO: Need instance colors to take account of parent instances.
                    ic = d.get_colors(displayed_only = True)
                    cg = combine_instance_geometry(va, na, vc, ta, pos, ic)
                    geom.append(cg)
                else:
                    geom.append((pos*va, pos.apply_without_translation(na), vc, ta))

    return geom
        
# -----------------------------------------------------------------------------
#
def geometry_buffers(geom):
    prim = []
    b = Buffers()
    from numpy import float32, uint32
    for (va, na, vc, ta) in geom:
        attr = {'POSITION': b.add_array(va.astype(float32, copy=False), bounds=True)}
        if na is not None:
            attr['NORMAL'] = b.add_array(na)
        if vc is not None:
            attr['COLOR_0'] = b.add_array(vc)
        prim.append({'attributes': attr, 'indices':b.add_array(ta.astype(uint32, copy=False))})

    return prim, b.buffer_views, b.accessors, b.chunk_bytes()


# -----------------------------------------------------------------------------
#
class Buffers:
    def __init__(self):
        self.accessors = []
        self.buffer_views = []
        self.buffer_bytes = []
        self.nbytes = 0

        from numpy import float32, uint32, uint16, int16, uint8, frombuffer
        self.value_types = {float32:5126, uint32:5125, uint16:5123, int16:5122, uint8:5120}

    # -----------------------------------------------------------------------------
    #
    def add_array(self, array, bounds=True):

        a = {}
        a['count'] = array.shape[0]
        if len(array.shape) == 1:
            t = 'SCALAR'
        elif array.shape[1] == 3:
            t = 'VEC3'
        elif array.shape[1] == 4:
            t = 'VEC4'
        else:
            raise glTFError('glTF buffer shape %s not allowed, must be 1 dimensional or N by 3'
                            % repr(tuple(array.shape)))
        a['type'] = t
        a['componentType'] = self.value_types[array.dtype.type]
        a['bufferView'] = len(self.buffer_views)

        if bounds:
            nd = array.ndim
            # TODO: Handle integer min/max
            if nd == 2:
                a['min'],a['max'] = (tuple(float(x) for x in array.min(axis=0)),
                                     tuple(float(x) for x in array.max(axis=0)))
            else:
                a['min'],a['max'] = float(array.min(axis=0)), float(array.max(axis=0))
                
        self.accessors.append(a)

        b = array.tobytes()
        nb = len(b)
        self.buffer_bytes.append(b)
        bv = {"byteLength": nb, "byteOffset": self.nbytes, "buffer": 0}
        self.buffer_views.append(bv)
        self.nbytes += nb

        return len(self.accessors) - 1


    # -----------------------------------------------------------------------------
    #
    def chunk_bytes(self):
        return b''.join(self.buffer_bytes)

# -----------------------------------------------------------------------------
#
def combine_instance_geometry(va, na, vc, ta, places, instance_colors):
    v = []
    n = []
    c = []
    t = []
    offset = 0
    for i,p in enumerate(places):
        v.append(p*va)
        n.append(p.apply_without_translation(na))
        if vc is None:
            from numpy import empty, uint8
            ivc = empty((len(va),4), uint8)
            ivc[:] = instance_colors[i]
            c.append(ivc)
        else:
            c.append(vc)
        t.append(ta+offset)
        offset += len(va)

    from numpy import concatenate
    return concatenate(v), concatenate(n), concatenate(c), concatenate(t)

# -----------------------------------------------------------------------------
#
def to_bytes(x, dtype):
    from numpy import array, little_endian
    ta = array((x,), dtype)
    if not little_endian:
        ta[:] = ta.byteswap()
    return ta.tobytes()
