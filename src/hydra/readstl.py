# -----------------------------------------------------------------------------
# Read in an STL file and create a surface.
# 
def read_stl(path, color = (.7,.7,.7,1)):

    f = open(path, 'rb')
    stl_data = f.read()
#    comment, va, na, ta = parse_stl(file)
    f.close()
    from . import _image3d
    comment, va, na, ta = _image3d.parse_stl(stl_data)

    s = STL_Surface(path)
    p = s.newPiece()
    p.geometry = va, ta
    p.normals = na
    p.color = color

    return s

def parse_stl(f):

    # First read 80 byte comment line.
    comment = f.read(80)

    # Next read uint32 triangle count.
    from numpy import fromstring, uint32, empty, float32
    tc = fromstring(f.read(4), uint32)        # triangle count

    # Next read 50 bytes per triangle containing float32 normal vector
    # followed three float32 vertices, followed by two "attribute bytes"
    # sometimes used to hold color information, but ignored by this reader.
    nv = empty((tc,12), float32)
    for t in range(tc):
        nt = f.read(12*4 + 2)
        nv[t,:] = fromstring(nt[:48], float32)

    va, na, ta = parse_stl_geometry(nv)

    return comment, va, na, ta

# -----------------------------------------------------------------------------
# 50 bytes per triangle containing float32 normal vector
# followed three float32 vertices, followed by two "attribute bytes"
# sometimes used to hold color information, but ignored by this reader.
#
# Eliminate identical vertices and average their normals.
#
def parse_stl_geometry(nv):

    tc = nv.shape[0]

    # Assign numbers to vertices.
    from numpy import empty, int32, uint32, float32, zeros
    tri = empty((tc, 3), int32)
    vnum = {}
    for t in range(tc):
        v0, v1, v2 = nv[t,3:6], nv[t,6:9], nv[t,9:12]
        for a, v in enumerate((v0, v1, v2)):
            tri[t,a] = vnum.setdefault(tuple(v), len(vnum))

    # Make vertex coordinate array.
    vc = len(vnum)
    vert = empty((vc,3), float32)
    for v, vn in vnum.items():
        vert[vn,:] = v

    # Make average normals array.
    normals = zeros((vc,3), float32)
    for t,tvi in enumerate(tri):
        for i in tvi:
            normals[i,:] += nv[t,0:3]
    from . import vector
    vector.normalize_vectors(normals)

    return vert, normals, tri

# -----------------------------------------------------------------------------
# Make special surface class for restoring sessions
#
from .surface import Surface
class STL_Surface(Surface):

    def __init__(self, path):

        Surface.__init__(self)
        self.path = path
        from os.path import basename
        self.name = basename(path)

    def session_state(self):
        p = self.plist[0]
        s = {'path':self.path,
             'place':self.place.matrix,
             'color':p.color}
        if p.copies:
            s['copies'] = p.copies
        return s

# -----------------------------------------------------------------------------
#
def save_stl_surfaces(file, viewer):

    slist = [m for m in viewer.models if isinstance(m, STL_Surface)]
    if slist:
        file.write("'stl surfaces':(\n")
        from .SessionUtil import objecttree
        for s in slist:
            st = s.session_state()
            objecttree.write_basic_tree(st, file, indent = ' ')
            file.write(',\n')
        file.write('),\n')

# -----------------------------------------------------------------------------
#
def restore_stl_surfaces(d, viewer):

    surfs = d.get('stl surfaces')
    if surfs is None:
        return
    from .place import Place
    for st in surfs:
        s = read_stl(st['path'])
        s.place = Place(st['place'])
        p = s.plist[0]
        p.color = st['color']
        if 'copies' in st:
            p.copies = [Place(c) for c in st['copies']]
        viewer.add_model(s)
