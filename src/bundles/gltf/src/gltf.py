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

    # Separate header and binary data
    json_chunk, bin_chunk = check_gltf_header(input)
    if input != filename:
        input.close()

    # Parse json header
    import json
    j = json.loads(json_chunk)

    # Check for geometry
    for attr in ('scenes', 'nodes', 'meshes', 'accessors', 'bufferViews'):
        if attr not in j:
            raise glTFError('glTF JSON contains no "%s": %s' % (attr, str(j)))

    # Make model for each scene and each node with hierarchy
    scenes, nodes = scene_and_node_models(j['scenes'], j['nodes'], name, session)

    # Make a Drawing for each mesh.
    colors = material_colors(j.get('materials'))
    ba = buffer_arrays(j['accessors'], j['bufferViews'], bin_chunk)
    mesh_drawings = meshes_as_models(session, j['meshes'], colors, ba)

    # Add mesh drawings to node models.
    already_used = set()
    for nm in nodes:
        if hasattr(nm, 'gltf_mesh'):
            md = mesh_drawings[nm.gltf_mesh]
            if len(md) == 1:
                # Don't make a child model for a mesh if there is just one child.
                copy_model(md[0], nm)
            else:
                for d in md:
                    if d in already_used:
                        # Copy drawing if instance is a child of more than one node
                        d = copy_model(d)
                    already_used.add(d)
                    nm.add_drawing(d)

    return scenes, ('Opened glTF file containing %d scenes, %d nodes, %d meshes'
                    % (len(scenes), len(nodes), len(j['meshes'])))

# -----------------------------------------------------------------------------
#
def check_gltf_header(input):
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

    jsonc, binc = chunks[0][1].decode('utf-8'), chunks[1][1]
    return jsonc, binc

# -----------------------------------------------------------------------------
#
def scene_and_node_models(scenes, nodes, file_name, session):
                    
    smodels = []
    for si, s in enumerate(scenes):
        if 'name' in s:
            sname = s['name']
        elif len(scenes) == 1:
            sname = file_name
        else:
            sname = '%s scene %d' % (file_name, si+1)
        sm = gltfModel(sname, session)
        smodels.append(sm)
        if 'nodes' not in s:
            raise glTFError('glTF scene %d has no nodes' % si)
        sm.gltf_nodes = s['nodes']

    # Make model for each node
    nmodels = []
    for ni,node in enumerate(nodes):
        if 'name' in node:
            nname = node['name']
        else:
            nname = '%d' % (ni+1)
        nm = gltfModel(nname, session)
        if 'mesh' in node:
            nm.gltf_mesh = node['mesh']
        if 'children' in node:
            nm.gltf_child_nodes = node['children']
        if 'matrix' in node:
            m = node['matrix']
            from chimerax.geometry import Place
            nm.position = Place(((m[0],m[4],m[8],m[12]),
                                 (m[1],m[5],m[9],m[13]),
                                 (m[2],m[6],m[10],m[14])))
        if 'scale' in node or 'translation' in node or 'rotation' in node:
            session.logger.warning('glTF node %d has unsupported rotation, scale or translation, ignoring it' % ni)
        nmodels.append(nm)

    # Add node models to scenes.
    for sm in smodels:
        sm.add([nmodels[ni] for ni in sm.gltf_nodes])

    # Add child nodes to parent nodes.
    already_used = set()
    copies = []
    for nm in nmodels:
        if hasattr(nm, 'gltf_child_nodes'):
            cmodels = []
            for ni in nm.gltf_child_nodes:
                c = nmodels[ni]
                if c in already_used:
                    # If a node is a child of multiple nodes copy it.
                    gltf_mesh = getattr(c, 'gltf_mesh', None)
                    c = copy_model(c)
                    if gltf_mesh is not None:
                        c.gltf_mesh = gltf_mesh
                    copies.append(c)
                already_used.add(c)
                cmodels.append(c)
            nm.add(cmodels)

    return smodels, nmodels + copies

# -----------------------------------------------------------------------------
#
def meshes_as_models(session, meshes, material_colors, buf_arrays):

    mesh_models = []
    ba = buf_arrays
    from numpy import int32
    from chimerax.core.models import Surface
    for m in meshes:
        if 'primitives' not in m:
            raise glTFError('glTF mesh has no "primitives": %s' % str(j))
        pdlist = []
        for pi,p in enumerate(m['primitives']):
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
                from chimerax import surface
                na = surface.calculate_vertex_normals(va, ta)
            if 'COLOR_0' in pa:
                vc = ba[pa['COLOR_0']]
            else:
                vc = None
            pd = Surface('p%d' % pi, session)
            if 'material' in p:
                pd.color = material_colors[p['material']]
            set_geometry(pd, va, na, vc, ta)
            pdlist.append(pd)
        mesh_models.append(pdlist)

    return mesh_models

# -----------------------------------------------------------------------------
#
def copy_model(model, to_model = None):
    if to_model is None:
        to_model = gltfModel(model.name, model.session)
        to_model.positions = model.positions
    c = to_model
    c.set_geometry(model.vertices, model.normals, model.triangles)
    c.color = model.color
    c.vertex_colors = model.vertex_colors
    return c

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
    value_type = {5126:float32, 5125:uint32, 5123:uint16, 5122:int16, 5121:uint8}
    atype_size = {'VEC3':3, 'VEC4':4, 'SCALAR':1}
    for a in accessors:
        ibv = a['bufferView']	# index into buffer_views
        bv = buffer_views[ibv]
        bo = bv['byteOffset']	# int
        bl = bv['byteLength']	# int
        bv = binc[bo:bo+bl]
        ct = a['componentType']	# 5123 = uint16, 5126 = float32, 5120 = uint8
        dtype = value_type[ct]
        atype = a['type']		# "VEC3", "SCALAR"
        av = bv
        if 'byteOffset' in a:
            ao = a['byteOffset']
            av = av[ao:]
        if 'count' in a:
            ac = a['count']
            nb = ac*dtype().itemsize*atype_size[atype]
            av = av[:nb]
        ba = frombuffer(av, dtype)
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
    model.set_geometry(va, na, ta)
    if vc is not None:
        model.vertex_colors = colors_to_uint8(vc)

# -----------------------------------------------------------------------------
#
def material_colors(materials):
    if materials is None:
        return []
    colors = []
    from chimerax.core.colors import rgba_to_rgba8
    for material in materials:
        pbr = material.get('pbrMetallicRoughness')
        if pbr and 'baseColorFactor' in pbr:
            color = rgba_to_rgba8(pbr['baseColorFactor'])
        else:
            color = (255,255,255,255)
        colors.append(color)
    return colors
            
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
def write_gltf(session, filename, models = None,
               center = None, size = None, short_vertex_indices = False,
               float_colors = False, preserve_transparency = True,
               instancing = False):
    if models is None:
        models = session.models.list()

    drawings = all_visible_drawings(models)

    buffers = Buffers()
    materials = Materials()
    nodes, meshes = nodes_and_meshes(drawings, buffers, materials,
                                     short_vertex_indices,
                                     float_colors, preserve_transparency,
                                     instancing)
    
    if center is not None or size is not None:
        from chimerax.geometry import union_bounds
        bounds = union_bounds(m.bounds() for m in models if m.is_visible)
        if bounds is not None:
            # Place positioning node above top-level nodes.
            cs_node = center_and_size(top_nodes(nodes), bounds, center, size)
            nodes.append(cs_node)

    encode_gltf(nodes, buffers, meshes, materials, filename)

# -----------------------------------------------------------------------------
#
def encode_gltf(nodes, buffers, meshes, materials, filename):
    
    # Write 80 character comment.
    from chimerax.core import version
    app_ver  = 'UCSF ChimeraX %s' % version
    
    h = {
        'asset': {'version': '2.0', 'generator': app_ver},
        'scenes': [{'nodes':top_nodes(nodes)}],
        'nodes': nodes,
        'meshes': meshes,
        'accessors': buffers.accessors,
        'materials': materials.material_specs,
        'bufferViews': buffers.buffer_views,
        'buffers':[{'byteLength': buffers.nbytes}],
    }

    import json
    json_text = json.dumps(h).encode('utf-8')
    nj = len(json_text)
    if nj % 4 != 0:
        # Pad. Following binary chunk is required to align to 4-byte boundary.
        json_text += b' ' * (4 - nj%4)
    from numpy import uint32
    clen = to_bytes(len(json_text), uint32)
    ctype = b'JSON'
    json_chunk = b''.join((clen, ctype, json_text))

    binc = buffers.chunk_bytes()
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
def top_nodes(nodes):
    # List all top level nodes as scenes.
    child_nodes = set(sum([n.get('children',[]) for n in nodes], []))
    top = [i for i,node in enumerate(nodes) if i not in child_nodes]
    return top

# -----------------------------------------------------------------------------
# Collect all drawings including descendants of specified models, excluding
# ones that show no triangles.
#
def all_visible_drawings(models):
    drawings = set()
    for m in models:
        if m.visible and m not in drawings:
            for d in m.all_drawings(displayed_only = True):
                drawings.add(d)
    # Prune drawings with nothing displayed.
    ts = {}
    dshown = tuple(d for d in drawings if any_triangles_shown(d, drawings, ts))
    return dshown

# -----------------------------------------------------------------------------
#
def any_triangles_shown(d, drawings, ts):
    if d in ts:
        return ts[d]
    if not d.display:
        ts[d] = False
    elif d.num_masked_triangles > 0:
        ts[d] = True
    else:
        for c in d.child_drawings():
            if c in drawings and any_triangles_shown(c, drawings, ts):
                ts[d] = True
                return True
        ts[d] = False
    return ts[d]

# -----------------------------------------------------------------------------
# Expand drawing instances into nodes and meshes.
#
def nodes_and_meshes(drawings, buffers, materials, short_vertex_indices = False,
                     float_colors = False, preserve_transparency = True,
                     leaf_instancing = True):

    # Create tree of nodes with children and matrices set.
    nodes, drawing_nodes = node_tree(drawings, leaf_instancing)

    # Create meshes for nodes.
    meshes = []
    for d, dnodes in drawing_nodes.items():
        va, na, vc, ta = d.vertices, d.normals, d.vertex_colors, d.masked_triangles
        if va is None or ta is None or len(ta) == 0:
            continue

        if not leaf_instancing:
            positions = d.get_positions(displayed_only = True)
            if not positions.is_identity():
                instance_colors = d.get_colors(displayed_only = True)
                va,na,vc,ta = combine_instance_geometry(va, na, vc, ta,
                                                        positions, instance_colors)

        single_colors = [n['single_color'] for n in dnodes]
        prims = primitives_from_geometry(va, na, vc, ta, single_colors,
                                         buffers, materials, short_vertex_indices,
                                         float_colors, preserve_transparency)
        for node,prim in zip(dnodes, prims):
            node['mesh'] = len(meshes)
            meshes.append({'primitives': prim})

    for node in nodes:
        del node['single_color']

    return nodes, meshes

# -----------------------------------------------------------------------------
# The GLTF 2.0 spec requires that a node cannot be the child of more than one
# other node.  So the child nodes for drawing instances need to be duplicated.
#
def node_tree(drawings, leaf_instancing):
    
    # Find top level drawings.
    child_drawings = []
    for d in drawings:
        child_drawings.extend(d.child_drawings())
    child_set = set(child_drawings)
    top_drawings = [d for d in drawings if not d in child_set]

    # Create the node hierarchy expanding instances into nodes.
    nodes = []		# All nodes.
    drawing_nodes = {}	# Maps drawing to list of nodes that are copies of that drawing.
    drawing_set = set(drawings)
    for drawing in top_drawings:
        create_node(drawing, drawing_set, nodes, drawing_nodes, leaf_instancing)

    return nodes, drawing_nodes

# -----------------------------------------------------------------------------
# Create node hierarchy starting at drawing, adding new nodes to nodes list
# and to the drawing_nodes mapping that records all the node copies for each
# drawing.
#
def create_node(drawing, drawing_set, nodes, drawing_nodes, leaf_instancing):
    dn = {'name': drawing.name,
          'single_color': drawing.color}
    dni = len(nodes)
    nodes.append(dn)

    children = [c for c in drawing.child_drawings() if c.display and c in drawing_set]

    positions = drawing.get_positions(displayed_only = True)
    if len(positions) == 1:
        inodes = gnodes = [dn]
        if not positions.is_identity():
            dn['matrix'] = gltf_transform(positions[0])
    elif leaf_instancing or children:
        ic = drawing.get_colors(displayed_only = True)
        inodes = [{'name': '%s %d' % (drawing.name, i+1),
                   'matrix': gltf_transform(p),
                   'single_color': ic[i]}
                  for i,p in enumerate(positions)]
        ni = len(nodes)
        nodes.extend(inodes)
        dn['children'] = list(range(ni,ni+len(inodes)))
        gnodes = inodes if leaf_instancing else [dn]
    else:
        # Copying leaf node geometry so don't make child nodes.
        gnodes = [dn]
        
    if drawing not in drawing_nodes:
        drawing_nodes[drawing] = []
    drawing_nodes[drawing].extend(gnodes)

    if children:
        for node in inodes:
            node['children'] = [create_node(c, drawing_set, nodes, drawing_nodes, leaf_instancing)
                                for c in children]

    return dni

# -----------------------------------------------------------------------------
#
def primitives_from_geometry(vertices, normals, vertex_colors, triangles, instance_colors,
                             buffers, materials, short_vertex_indices,
                             float_colors, preserve_transparency):

    geom = [(vertices, normals, vertex_colors, triangles)]
    if short_vertex_indices:
        geom = limit_vertex_count(geom)

    geom_bufs = geometry_buffers(geom, buffers, short_vertex_indices,
                                 float_colors, preserve_transparency)

    prims = [geometry_primitives(geom_bufs, color, materials, preserve_transparency)
             for color in instance_colors]

    return prims

# -----------------------------------------------------------------------------
#
def geometry_buffers(geom, buffers, short_vertex_indices,
                     float_colors, preserve_transparency):
    geom_bufs = []
    b = buffers
    from numpy import float32, uint32, uint16
    for pva,pna,pvc,pta in geom:
        pi = b.add_array(pva.astype(float32, copy=False), bounds=True)
        ni = b.add_array(pna) if pna is not None else None
        if pvc is None:
            ci = None
        else:
            if not preserve_transparency:
                pvc = pvc[:,:3]
            if float_colors:
                pvc = pvc.astype(float32)
                pvc /= 255
            ci = b.add_array(pvc, normalized = not float_colors)
        etype = uint16 if short_vertex_indices else uint32
        ne = len(pta)
        ea = pta.astype(etype, copy=False).reshape((3*ne,))
        ti = b.add_array(ea)
        geom_bufs.append((pi,ni,ci,ti))

    return geom_bufs

# -----------------------------------------------------------------------------
#
def geometry_primitives(geom_bufs, color, materials, preserve_transparency):
    prims = []
    for vi,ni,ci,ti in geom_bufs:
        attr = {'POSITION': vi}
        prim = {'attributes': attr,
                'indices': ti}
        if ni is not None:
            attr['NORMAL'] = ni
        if ci is None:
            prim['material'] = materials.single_color(color, preserve_transparency)
        else:
            attr['COLOR_0'] = ci
        prims.append(prim)
    return prims

# -----------------------------------------------------------------------------
# Split triangle geometry so vertex arrays are of specified maximum size.
# To handle Unity3D only allowing 16-bit vertex indices.
#
def limit_vertex_count(geom, vmax = 2**16):
    lgeom = []
    for va,na,vc,ta in geom:
        if len(va) <= vmax:
            lgeom.append((va,na,vc,ta))
        else:
            vi = []
            vmap = {}
            ti0 = 0
            nt = len(ta)
            for ti,tv in enumerate(ta):
                for v in tv:
                    if v not in vmap:
                        vs = len(vmap)
                        vmap[v] = vs
                        vi.append(v)
                if len(vmap) > vmax - 3 or ti == nt-1:
                    sva = va[vi]
                    sna = None if na is None else na[vi]
                    svc = None if vc is None else vc[vi]
                    from numpy import array
                    sta = array([vmap[v] for tv in ta[ti0:ti+1] for v in tv])
                    sta = sta.reshape((len(sta)//3,3))
                    lgeom.append((sva,sna,svc,sta))
                    vi = []
                    vmap = {}
                    ti0 = ti+1
    return lgeom

# -----------------------------------------------------------------------------
#
class Buffers:
    def __init__(self):
        self.accessors = []
        self.buffer_views = []
        self.buffer_bytes = []
        self.nbytes = 0

        from numpy import float32, uint32, uint16, int16, uint8, frombuffer
        self.value_types = {float32:5126, uint32:5125, uint16:5123, int16:5122, uint8:5121}

    # -----------------------------------------------------------------------------
    #
    def add_array(self, array, bounds=False, normalized=False):

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
        if normalized:
            a['normalized'] = True	# Required for COLOR_0

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
class Materials:
    def __init__(self):
        self._colors = {}	# color 255 tuple -> index
        self._color_specs = []
        
    def single_color(self, color, preserve_transparency = True):
        r,g,b,a = color
        if not preserve_transparency:
            a = 255
        c = (r,g,b,a)
        ci = self._colors.get(c, None)
        if ci is None:
            self._colors[c] = ci = len(self._color_specs)
            self._color_specs.append(c)
        return ci

    @property
    def material_specs(self):
        from chimerax.core.colors import rgba8_to_rgba
        return [{'pbrMetallicRoughness': {'baseColorFactor': rgba8_to_rgba(rgba)}}
                 for rgba in self._color_specs]

# -----------------------------------------------------------------------------
#
def center_and_size(nodes, bounds, center, size):

    if bounds is None:
        return
    if center is None and size is None:
        return
    
    c = bounds.center()
    s = max(bounds.size())
    f = 1 if size is None else size / s
    tx,ty,tz = (0,0,0) if center is None else [center[a]-f*c[a] for a in (0,1,2)]
    matrix = [f,0,0,0,
              0,f,0,0,
              0,0,f,0,
              tx,ty,tz,1]
    
    cs_node = {'name': 'centering',
               'children': nodes,
               'matrix': matrix}
    return cs_node

# -----------------------------------------------------------------------------
#
def gltf_transform(place):
    (m00,m01,m02,m03),(m10,m11,m12,m13),(m20,m21,m22,m23) = place.matrix
    return [m00,m10,m20,0,
            m01,m11,m21,0,
            m02,m12,m22,0,
            m03,m13,m23,1]

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
        n.append(p.transform_vectors(na))
        if vc is None:
            ivc = single_vertex_color(len(va), instance_colors[i])
            c.append(ivc)
        else:
            c.append(vc)
        t.append(ta+offset)
        offset += len(va)

    from numpy import concatenate
    return concatenate(v), concatenate(n), concatenate(c), concatenate(t)

# -----------------------------------------------------------------------------
#
def single_vertex_color(n, color):
    from numpy import empty, uint8
    vc = empty((n,4), uint8)
    vc[:] = color
    return vc

# -----------------------------------------------------------------------------
#
def to_bytes(x, dtype):
    from numpy import array, little_endian
    ta = array((x,), dtype)
    if not little_endian:
        ta[:] = ta.byteswap()
    return ta.tobytes()
