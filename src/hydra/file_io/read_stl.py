# -----------------------------------------------------------------------------
# 
def read_stl(path, color = (.7,.7,.7,1)):
    '''
    Read a STL (Stereo Lithography) surface file and create a surface.
    '''

    f = open(path, 'rb')
    stl_data = f.read()
#    comment, va, na, ta = parse_stl(file)
    f.close()
    from .. import _image3d
    comment, va, na, ta = _image3d.parse_stl(stl_data)

    s = STL_Surface(path)
    p = s.new_piece()
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
    from ..geometry import vector
    vector.normalize_vectors(normals)

    return vert, normals, tri

# -----------------------------------------------------------------------------
# Make special surface class for restoring sessions
#
from ..surface import Surface
class STL_Surface(Surface):

    def __init__(self, path):

        from os.path import basename
        name = basename(path)
        Surface.__init__(self, name)

        self.path = path

    def session_state(self):
        p = self.plist[0]
        s = {'id':self.id,
             'path':self.path,
             'displayed': self.displayed,
             'place':self.place.matrix,
             'color':p.color}
        if p.copies:
            s['copies'] = tuple(c.matrix for c in p.copies)
        return s

# -----------------------------------------------------------------------------
#
def restore_stl_surfaces(surfs, session, attributes_only = False):

    if attributes_only:
        models = session.model_list()
        sids = dict((m.id,m) for m in models if isinstance(m, STL_Surface))
    from ..geometry.place import Place
    for st in surfs:
        if attributes_only:
            sid = st['id']
            if sid in sids:
                s = sids[sid]
            else:
                continue
        else:
            s = read_stl(st['path'])
            s.id = st['id']
        s.displayed = st['displayed']
        s.place = Place(st['place'])
        p = s.plist[0]
        p.color = st['color']
        if 'copies' in st:
            p.copies = [Place(c) for c in st['copies']]
        session.add_model(s)
