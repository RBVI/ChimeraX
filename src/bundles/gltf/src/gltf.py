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
gltf: glTF file format support
==============================

Read and write glTF 3d scene files.
"""

# -----------------------------------------------------------------------------
#
from chimerax.core.errors import UserError
class glTFError(UserError):
    pass


# -----------------------------------------------------------------------------
# Mesh styles
#
GLTF_POINTS = 0
GLTF_LINES = 1
GLTF_TRIANGLES = 4

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class gltfModel(Surface):
    SESSION_SAVE_DRAWING = True
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
    colors, textures = material_colors_and_textures(j.get('materials'),
                                                    j.get('textures'),
                                                    j.get('images'))
    bv = buffer_views(j['bufferViews'], bin_chunk)
    ba = buffer_arrays(j['accessors'], bv, bin_chunk)
    mesh_drawings = meshes_as_models(session, j['meshes'], colors, textures, ba, bv)

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
def meshes_as_models(session, meshes, material_colors, material_textures,
                     buffer_arrays, buffer_views):

    mesh_models = []
    ba = buffer_arrays
    from numpy import int32
    from chimerax.core.models import Surface
    for m in meshes:
        if 'primitives' not in m:
            raise glTFError('glTF mesh has no "primitives": %s' % str(j))
        pdlist = []
        for pi,p in enumerate(m['primitives']):
            element_size = _element_size_for_gltf_mode(p.get('mode'))
            if 'indices' not in p:
                raise glTFError('glTF missing "indices" in primitive %s' % str(p))
            ta = ba[p['indices']]
            if len(ta.shape) == 1:
                ta = ta.reshape((len(ta)//element_size, element_size))
            ta = ta.astype(int32, copy=False)
            if 'attributes' not in p:
                raise glTFError('glTF missing "attributes" in primitive %s' % str(p))
            pa = p['attributes']
            if 'POSITION' not in pa:
                raise glTFError('glTF missing "POSITION" attribute in primitive %s' % str(p))
            va = ba[pa['POSITION']]
            if 'NORMAL' in pa:
                na = ba[pa['NORMAL']]
            elif ta.shape[1] == 3:
                # Compute triangle normals
                from chimerax import surface
                na = surface.calculate_vertex_normals(va, ta)
            else:
                na = None	# No computed normals for mesh or dots.
            if 'COLOR_0' in pa:
                vc = ba[pa['COLOR_0']]
            else:
                vc = None
            tc = ba[pa['TEXCOORD_0']] if 'TEXCOORD_0' in pa else None
            pd = Surface('p%d' % pi, session)
            if 'material' in p:
                mi = p['material']
                pd.color = material_colors[mi]
                image_buf = material_textures[mi]
                if image_buf is not None:
                    pd.texture = _create_texture(buffer_views[image_buf])
            set_geometry(pd, va, na, vc, tc, ta)
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
    c.texture_coordinates = model.texture_coordinates
    c.texture = model.texture
    c.display_style = model.display_style
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
def buffer_views(buffer_views, binc):
    bviews = []
    for bv in buffer_views:
        bo = bv.get('byteOffset', 0)	# int
        bl = bv['byteLength']	# int
        bviews.append(binc[bo:bo+bl])
    return bviews
        
# -----------------------------------------------------------------------------
#
def buffer_arrays(accessors, buffer_views, binc):
    balist = []
    from numpy import float32, uint32, uint16, int16, uint8, frombuffer
    value_type = {5126:float32, 5125:uint32, 5123:uint16, 5122:int16, 5121:uint8}
    atype_size = {'VEC2':2, 'VEC3':3, 'VEC4':4, 'SCALAR':1}
    for a in accessors:
        ibv = a['bufferView']	# index into buffer_views
        bv = buffer_views[ibv]
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
        elif atype == 'VEC2':
            ba = ba.reshape((len(ba)//2, 2))
        elif atype == 'SCALAR':
            pass
        else:
            raise glTFError('glTF accessor type is not VEC2, VEC3, VEC4 or SCALAR, got %s' % atype)
        balist.append(ba)
    return balist
        
# -----------------------------------------------------------------------------
#
def set_geometry(model, va, na, vc, tc, ta):
    model.set_geometry(va, na, ta)
    if vc is not None:
        model.vertex_colors = colors_to_uint8(vc)
    if tc is not None:
        model.texture_coordinates = tc
    if ta.ndim == 2 and ta.shape[1] == 2:
        model.display_style = model.Mesh
    elif ta.ndim == 1 or (ta.ndim == 2 and ta.shape[1] == 1):
        model.display_style = model.Dot

# -----------------------------------------------------------------------------
#
def material_colors_and_textures(materials, textures, images):
    if materials is None:
        return [], []
    colors = []
    tex = []
    from chimerax.core.colors import rgba_to_rgba8
    for material in materials:
        pbr = material.get('pbrMetallicRoughness')
        if pbr and 'baseColorFactor' in pbr:
            color = rgba_to_rgba8(pbr['baseColorFactor'])
        else:
            color = (255,255,255,255)
        colors.append(color)
        tex.append(_material_texture_buffer(pbr, textures, images))
    return colors, tex

# -----------------------------------------------------------------------------
#
def _material_texture_buffer(pbr, textures, images):
    if pbr is None:
        return None
    if 'baseColorTexture' not in pbr:
        return None
    pbrt = pbr['baseColorTexture']
    if 'index' not in pbrt:
        return None
    ti = pbrt['index']
    if textures is None:
        return None
    if ti >= len(textures):
        return None
    tex = textures[ti]		# Texture index
    if 'source' not in tex:
        return None
    ii = tex['source']	# Image index
    if images is None:
        return None
    if ii >= len(images):
        return None
    im = images[ii]
    if 'bufferView' not in im:
        return None
    if 'mimeType' not in im:
        return None
    if im['mimeType'] != 'image/png':
        return None
    tbv = im['bufferView']
    return tbv

# -----------------------------------------------------------------------------
#
def _create_texture(png_bytes):
    from io import BytesIO
    stream = BytesIO(png_bytes)
    from PIL import Image
    image = Image.open(stream)
    from numpy import array
    color_array = array(image)
    from chimerax.graphics import Texture
    texture = Texture(color_array)
    return texture

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
def write_gltf(session, filename = None, models = None,
               center = True, size = None, short_vertex_indices = False,
               float_colors = False, preserve_transparency = True,
               texture_colors = False, prune_vertex_colors = True,
               instancing = False,
               metallic_factor = 0, roughness_factor = 1,
               flat_lighting = False, backface_culling = True):
    if models is None:
        models = session.models.list()

    drawings = all_visible_drawings(models)

    buffers = Buffers()
    materials = Materials(buffers, preserve_transparency, float_colors,
                          texture_colors, metallic_factor, roughness_factor,
                          flat_lighting, backface_culling)
    nodes, meshes = nodes_and_meshes(drawings, buffers, materials,
                                     short_vertex_indices, prune_vertex_colors,
                                     instancing)

    if center is True:
        center = (0,0,0)
    elif center is False:
        center = None
        
    if center is not None or size is not None:
        from chimerax.geometry import union_bounds
        bounds = union_bounds(m.bounds() for m in models if m.visible)
        if bounds is not None:
            # Place positioning node above top-level nodes.
            cs_node = center_and_size(top_nodes(nodes), bounds, center, size)
            nodes.append(cs_node)

    glb = encode_gltf(nodes, buffers, meshes, materials)

    if filename is not None:
        file = open(filename, 'wb')
        file.write(glb)
        file.close()

    return glb
        
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
def encode_gltf(nodes, buffers, meshes, materials):
    
    # Write 80 character comment.
    from chimerax.core import version
    app_ver  = 'UCSF ChimeraX %s' % version
    
    h = {
        'asset': {'version': '2.0', 'generator': app_ver},
        'scenes': [{'nodes':top_nodes(nodes)}],
        'nodes': nodes,
        'meshes': meshes.mesh_specs,
        'accessors': buffers.accessors,
        'bufferViews': buffers.buffer_views,
        'buffers':[{'byteLength': buffers.nbytes}],
    }
    if len(materials.material_specs) > 0:
        h['materials'] = materials.material_specs

    if len(materials.textures) > 0:
        h.update(materials.textures.texture_specs)	# adds 'textures', 'images', 'samplers'
        
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
    if len(binc) % 4 != 0:
        # Chunk length required to be a multiple of 4 bytes.
        binc += b'\0' * (4-len(binc)%4)
    blen = to_bytes(len(binc), uint32)
    btype = b'BIN\x00'
    bin_chunk = b''.join((blen, btype, binc))
    
    magic = to_bytes(0x46546c67, uint32)
    version = to_bytes(2, uint32)
    length = to_bytes(12 + len(json_chunk) + len(bin_chunk), uint32)

    glb = b''.join((magic, version, length, json_chunk, bin_chunk))
    return glb

# -----------------------------------------------------------------------------
#
def to_bytes(x, dtype):
    from numpy import array, little_endian
    ta = array((x,), dtype)
    if not little_endian:
        ta[:] = ta.byteswap()
    return ta.tobytes()

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
def nodes_and_meshes(drawings, buffers, materials,
                     short_vertex_indices = False, prune_vertex_colors = True,
                     leaf_instancing = False):

    # Create tree of nodes with children and matrices set.
    nodes, drawing_nodes = node_tree(drawings, leaf_instancing)

    # Create meshes for nodes.
    meshes = Meshes(buffers, materials, short_vertex_indices, prune_vertex_colors, leaf_instancing)
    for drawing, dnodes in drawing_nodes.items():
        if meshes.has_mesh(drawing):
            for node in dnodes:
                node['mesh'] = meshes.mesh_index(drawing, node['single_color'])

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
        dn['single_color'] = (255,255,255,255)	# color factor if texture colors used.
        
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
def gltf_transform(place):
    (m00,m01,m02,m03),(m10,m11,m12,m13),(m20,m21,m22,m23) = place.matrix
    return [m00,m10,m20,0,
            m01,m11,m21,0,
            m02,m12,m22,0,
            m03,m13,m23,1]

# -----------------------------------------------------------------------------
#
class Meshes:
    def __init__(self, buffers, materials, short_vertex_indices = False,
                 prune_vertex_colors = True, leaf_instancing = False):
        self._buffers = buffers
        self._materials = materials
        self._short_vertex_indices = short_vertex_indices
        self._prune_vertex_colors = prune_vertex_colors
        self._leaf_instancing = leaf_instancing
        self._meshes = {}	# Map Drawing to Mesh.
        self._mesh_specs = []	# List of all mesh specifications

    def has_mesh(self, drawing):
        if drawing.vertices is None or drawing.triangles is None or len(drawing.triangles) == 0:
            return False
        return True
    
    def mesh_index(self, drawing, instance_color):
        mesh = self._meshes.get(drawing)
        if mesh is None:
            mesh = Mesh(drawing, self._buffers, self._materials,
                        self._short_vertex_indices, self._prune_vertex_colors,
                        self._leaf_instancing)
            self._meshes[drawing] = mesh
        mi = len(self._mesh_specs)
        spec = mesh.specification(instance_color)
        self._mesh_specs.append(spec)
        return mi

    @property
    def mesh_specs(self):
        return self._mesh_specs
    
# -----------------------------------------------------------------------------
#
class Mesh:
    def __init__(self, drawing, buffers, materials,
                 short_vertex_indices = False, prune_vertex_colors = True,
                 leaf_instancing = False):
        self._drawing = drawing
        self._buffers = buffers
        self._materials = materials
        self._short_vertex_indices = short_vertex_indices
        self._prune_vertex_colors = prune_vertex_colors
        self._leaf_instancing = leaf_instancing
        
        self._primitives = None
        self._geom_buffers = None
        self._texture_images = []
        self._converted_vertex_to_texture_colors = False
        self._has_vertex_colors = (drawing.vertex_colors is not None)

    # -----------------------------------------------------------------------------
    #
    def specification(self, instance_color):
        prims = self._geometry_primitives()
        if not self._has_vertex_colors:
            # Need to copy the geometry with the correct material color.
            from copy import deepcopy
            prims = deepcopy(prims)
            if self._converted_vertex_to_texture_colors:
                instance_color = (255,255,255,255)
        else:
            instance_color = (255,255,255,255)  # Modulated by vertex colors
        materials = self._materials
        d = self._drawing
        transparent = d.showing_transparent(include_children = False)
        twosided_lighting = (d.multitexture is not None or d.texture is not None)
        tex_images = self._texture_images
        for p, prim in enumerate(prims):
            texture_image = tex_images[p % len(tex_images)] if tex_images else None
            material_color = prim.pop('single_vertex_color', instance_color)
            material = materials.material(material_color, texture_image,
                                          transparent = transparent,
                                          twosided_lighting = twosided_lighting)
            prim['material'] = material.index
        mesh = {'primitives': prims}
        return mesh

    # -----------------------------------------------------------------------------
    #
    def _geometry_primitives(self):
        prims = self._primitives
        if prims is not None:
            return prims
        
        self._primitives = prims = []
        for vi,ni,ci,tci,ti,mode,single_vertex_color in self._geometry_buffers():
            attr = {'POSITION': vi}
            prim = {'attributes': attr,
                    'indices': ti,
                    'mode': mode}
            if single_vertex_color is not None:
                prim['single_vertex_color'] = single_vertex_color
            if ni is not None:
                attr['NORMAL'] = ni
            if ci is not None:
                attr['COLOR_0'] = ci
            if tci is not None:
                attr['TEXCOORD_0'] = tci
            prims.append(prim)
            
        return prims

    # -----------------------------------------------------------------------------
    #
    def _geometry_buffers(self):
        geom_bufs = self._geom_buffers
        if geom_bufs is not None:
            return geom_bufs
        
        d = self._drawing
        va, na, vc, tc = (d.vertices, d.normals, d.vertex_colors, d.texture_coordinates)

        # Get triangles, lines or points
        if d.display_style == d.Solid:
            ta = d.masked_triangles
        else:
            ta = d._draw_shape.elements	# Lines or points
            
        # Collect textures
        self._texture_images = _read_texture_images(d)
        if len(self._texture_images) == 0:
            tc = None

        # Combine instances into a single triangle set.
        if not self._leaf_instancing:
            positions = d.get_positions(displayed_only = True)
            if not positions.is_identity():
                instance_colors = d.get_colors(displayed_only = True)
                va,na,vc,tc,ta = combine_instance_geometry(va, na, vc, tc, ta,
                                                           positions, instance_colors)

        # Convert vertex colors to texture colors
        if self._materials._convert_vertex_to_texture_colors:
            if vc is not None:
                from chimerax.surface.texture import has_single_color_triangles
                if has_single_color_triangles(ta, vc):
                    from chimerax.surface.texture import vertex_colors_to_texture
                    tex_coords, tex_image = vertex_colors_to_texture(vc)
                    tc = tex_coords
                    self._texture_images = [tex_image]
                    vc = None
                    self._converted_vertex_to_texture_colors = True

        geom = [(va, na, vc, tc, ta)]

        # Split multitexture into one geometry per texture
        ntex = len(self._texture_images)
        if ntex > 1:
            geom = _split_multitexture_geometry(va, na, vc, tc, ta, ntex)
                
        # Some implementations only allow 16-bit unsigned vertex indices.
        if self._short_vertex_indices:
            geom = limit_vertex_count(geom)
            
        self._geom_buffers = geom_bufs = [self._make_buffers(pva,pna,pvc,ptc,pta)
                                          for pva,pna,pvc,ptc,pta in geom]
        return geom_bufs

    # -----------------------------------------------------------------------------
    #
    def _make_buffers(self, va, na, vc, tc, ta):
        b = self._buffers
        mat = self._materials
        from numpy import float32, uint32, uint16
        
        vi = b.add_array(va.astype(float32, copy=False), bounds=True, target=b.GLTF_ARRAY_BUFFER)
        ni = b.add_array(na, target=b.GLTF_ARRAY_BUFFER) if na is not None and not self._materials.flat_lighting else None
        ci = None
        single_vertex_color = None
        if vc is not None:
            if not mat._preserve_transparency:
                vc = vc[:,:3]
            if mat._float_vertex_colors:
                vc = vc.astype(float32)
                vc /= 255
            if self._prune_vertex_colors:
                single_vertex_color = _single_vertex_color(vc)
            if single_vertex_color is None:
                ci = b.add_array(vc, normalized = not mat._float_vertex_colors, target=b.GLTF_ARRAY_BUFFER)
        tci = b.add_array(tc) if tc is not None else None
        ne = len(ta)
        etype = uint16 if self._short_vertex_indices else uint32
        ea = ta.astype(etype, copy=False).reshape((ta.size,))
        ti = b.add_array(ea, target=b.GLTF_ELEMENT_ARRAY_BUFFER)
        mode = _mesh_style(ta)
        return (vi,ni,ci,tci,ti,mode,single_vertex_color)
    
# -----------------------------------------------------------------------------
#
def _single_vertex_color(vertex_colors):
    if len(vertex_colors) > 0:
        color = vertex_colors[0]
        if (vertex_colors == color).all():
            return tuple(color)
    return None
    
# -----------------------------------------------------------------------------
#
def _mesh_style(triangle_array):
    ndim = triangle_array.ndim
    esize = triangle_array.shape[1] if ndim == 2 else 1
    if esize == 3:
        mode = GLTF_TRIANGLES
    elif esize == 2:
        mode = GLTF_LINES
    else:
        mode = GLTF_POINTS
    return mode
    
# -----------------------------------------------------------------------------
#
def _element_size_for_gltf_mode(gltf_mode):
    if gltf_mode is None or gltf_mode == GLTF_TRIANGLES:
        element_size = 3
    elif gltf_mode == GLTF_LINES:
        element_size = 2
    elif gltf_mode == GLTF_POINTS:
        element_size = 1
    else:
        raise glTFError('glTF reader only handles triangles, lines, and points, got mode %d' % p['mode'])
    return element_size
    
# -----------------------------------------------------------------------------
#
def _read_texture_images(drawing):
    d = drawing
    if d.texture and d.texture.dimension == 2:
        d._opengl_context.make_current()
        ti = [d.texture.read_texture_data()]
        d._opengl_context.done_current()
    elif d.multitexture:
        d._opengl_context.make_current()
        ti = [t.read_texture_data() for t in d.multitexture]
        d._opengl_context.done_current()
    else:
        ti = []

    return ti

# -----------------------------------------------------------------------------
#
def _split_multitexture_geometry(va, na, vc, tc, ta, ntex):
    tpt = len(ta)//ntex
    geom = []
    for i in range(ntex):
        tai = ta[i*tpt:(i+1)*tpt]
        vi_min, vi_max = tai.min(), tai.max()
        vai = va[vi_min:vi_max+1]
        nai = na[vi_min:vi_max+1] if na is not None else None
        vci = vc[vi_min:vi_max+1] if vc is not None else None
        tci = tc[vi_min:vi_max+1]
        geom.append((vai, nai, vci, tci, tai - vi_min))
    return geom

# -----------------------------------------------------------------------------
#
def combine_instance_geometry(va, na, vc, tc, ta, places, instance_colors):
    v = []
    n = []
    c = []
    u = []
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
        if tc:
            u.append(tc)
        t.append(ta+offset)
        offset += len(va)

    from numpy import concatenate
    ctc = concatenate(u) if tc else None

    cva, cna, cca, cta = concatenate(v), concatenate(n), concatenate(c), concatenate(t)

    # Instanced geometry often is scaled so normals need renormalizing.
    from chimerax.geometry import normalize_vectors
    normalize_vectors(cna)
    
    return cva, cna, cca, ctc, cta

# -----------------------------------------------------------------------------
#
def single_vertex_color(n, color):
    from numpy import empty, uint8
    vc = empty((n,4), uint8)
    vc[:] = color
    return vc

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
                esize = len(tv)
                if len(vmap) > vmax - esize or ti == nt-1:
                    sva = va[vi]
                    sna = None if na is None else na[vi]
                    svc = None if vc is None else vc[vi]
                    from numpy import array
                    sta = array([vmap[v] for tv in ta[ti0:ti+1] for v in tv])
                    sta = sta.reshape((len(sta)//esize,esize))
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
    def add_array(self, array, bounds=False, normalized=False, target=None):

        a = {}
        a['count'] = array.shape[0]
        if len(array.shape) == 1:
            t = 'SCALAR'
        elif array.shape[1] == 2:
            t = 'VEC2'
        elif array.shape[1] == 3:
            t = 'VEC3'
        elif array.shape[1] == 4:
            t = 'VEC4'
        else:
            raise glTFError('glTF buffer shape %s not allowed, must be 1 or 2 dimensional with second dimension size 1, 2, 3, or 4' % repr(tuple(array.shape)))
        a['type'] = t
        a['componentType'] = self.value_types[array.dtype.type]
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
        a['bufferView'] = self.add_buffer(b, target=target)

        return len(self.accessors) - 1

    # -----------------------------------------------------------------------------
    # Possible bufferView targets.
    # This is not required by the GLTF spec, but BabylonJS warns when target is not
    # specified.
    #
    GLTF_ARRAY_BUFFER = 34962
    GLTF_ELEMENT_ARRAY_BUFFER = 34963
    
    # -----------------------------------------------------------------------------
    #
    def add_buffer(self, bytes, target=None):
        bvi = len(self.buffer_views)
        nb = len(bytes)
        if nb % 4 != 0:
            bytes += b'\0' * (4 - (nb%4))  # byteOffset is required to be multiple of 4
        self.buffer_bytes.append(bytes)
        bv = {"byteLength": nb, "byteOffset": self.nbytes, "buffer": 0}
        if target is not None:
            bv["target"] = target
        self.buffer_views.append(bv)
        self.nbytes += len(bytes)
        return bvi

    # -----------------------------------------------------------------------------
    #
    def chunk_bytes(self):
        return b''.join(self.buffer_bytes)

# -----------------------------------------------------------------------------
#
class Materials:
    def __init__(self, buffers, preserve_transparency = True, float_vertex_colors = False,
                 convert_vertex_to_texture_colors = False,
                 metallic_factor = 0, roughness_factor = 1,
                 flat_lighting = False, backface_culling = True):
        self._materials = []
        self._preserve_transparency = preserve_transparency
        self._float_vertex_colors = float_vertex_colors
        self._convert_vertex_to_texture_colors = convert_vertex_to_texture_colors
        self._metallic_factor = metallic_factor;
        self._roughness_factor = roughness_factor;
        self.flat_lighting = flat_lighting
        self._backface_culling = backface_culling
        self.textures = Textures(buffers)

        self._single_color_materials = {}	# (rgba, transparent, twosided) -> Material, reuse these
        
    def material(self, color, texture_image = None,
                 transparent = False, twosided_lighting = False):
        r,g,b,a = color
        if not self._preserve_transparency:
            a = 255
        c = (r,g,b,a)
        if not self._backface_culling:
            twosided_lighting = True
        if texture_image is None:
            ctt = (c, transparent, twosided_lighting)
            m = self._single_color_materials.get(ctt, None)
            if m:
                return m

        mi = len(self._materials)
        ti = self.textures.add_texture(texture_image) if texture_image is not None else None
        m = Material(mi, c, texture_index = ti,
                     transparent = (transparent and self._preserve_transparency),
                     metallic_factor = self._metallic_factor,
                     roughness_factor = self._roughness_factor,
                     twosided_lighting = twosided_lighting)
        self._materials.append(m)
        if texture_image is None:
            self._single_color_materials[ctt] = m

        return m

    @property
    def material_specs(self):
        return [m.specification for m in self._materials]

# -----------------------------------------------------------------------------
#
class Material:
    def __init__(self, material_index, base_color8, texture_index = None,
                 transparent = False, metallic_factor = 0, roughness_factor = 1,
                 twosided_lighting = False):
        self._index = material_index
        self._base_color8 = base_color8
        self._texture_index = texture_index
        self._transparent = transparent
        self._metallic_factor = metallic_factor
        self._roughness_factor = roughness_factor
        self._twosided_lighting = twosided_lighting

    @property
    def index(self):
        return self._index
    
    @property
    def specification(self):
        from chimerax.core.colors import rgba8_to_rgba
        color = rgba8_to_rgba(self._base_color8)
        pbr = {'baseColorFactor': color,
               'metallicFactor': self._metallic_factor,
               'roughnessFactor': self._roughness_factor,
               }
        if self._texture_index is not None:
            pbr['baseColorTexture'] = {'index': self._texture_index}
        spec = {'pbrMetallicRoughness': pbr}
        if self._transparent:
            spec['alphaMode'] = 'BLEND'
        if self._twosided_lighting:
            spec['doubleSided'] = True
        return spec

# -----------------------------------------------------------------------------
#
class Textures:
    def __init__(self, buffers):
        self._buffers = buffers
        self._texture_buffers = []	# gltf buffer ids for each image
        self._array_image_id = {}	# id(rgba_array) -> image buffer id

    def __len__(self):
        return len(self._texture_buffers)
    
    def add_texture(self, rgba_array):
        a_id = id(rgba_array)
        if a_id in self._array_image_id:
            return self._array_image_id[a_id]
        bv = self._texture_buffer(rgba_array)
        self._texture_buffers.append(bv)
        im_id = len(self._texture_buffers)-1
        self._array_image_id[a_id] = im_id
        return im_id

    def _texture_buffer(self, rgba_array):
        '''Make a PNG image and save as a buffer.'''
        # Ut oh, Texture does not keep data after filling OpenGL texture, to save memory
        # for large volume data.
        from PIL import Image
        pi = Image.fromarray(rgba_array)
        from io import BytesIO
        image_bytes = BytesIO()
        pi.save(image_bytes, format='PNG')
        bvi = self._buffers.add_buffer(image_bytes.getvalue())
        return bvi
    
    @property
    def texture_specs(self):
        textures = [{'source': i} for i in range(len(self._texture_buffers))]
        images = [{'bufferView': bv, 'mimeType':'image/png'}
                  for bv in self._texture_buffers]
        samplers = [{}]
        return {'textures': textures,
                'images': images,
                'samplers': samplers,
        }

