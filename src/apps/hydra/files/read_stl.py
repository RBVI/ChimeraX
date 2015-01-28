# -----------------------------------------------------------------------------
# 
def read_stl(path, session, color = (178,178,178,255)):
    '''
    Read a STL (Stereo Lithography) surface file and create a surface.
    '''

    f = open(path, 'rb')
    stl_data = f.read()
#    comment, va, na, ta = parse_stl(file)
    f.close()
    from ..surface.surface_cpp import parse_stl
    comment, va, na, ta = parse_stl(stl_data)

    s = STL_Surface(path)
    s.geometry = va, ta
    s.normals = na
    s.color = color

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
from ..models import Model
class STL_Surface(Model):

    def __init__(self, path):

        from os.path import basename
        name = basename(path)
        Model.__init__(self, name)

        self.path = path

    def session_state(self):
        s = {'id':self.id,
             'path':self.path,
             'display': self.display,
             'positions': self.positions.array(),
             'colors':self.colors}
        return s

# -----------------------------------------------------------------------------
#
def restore_stl_surfaces(surfs, session, file_paths, attributes_only = False):

    if attributes_only:
        sids = dict((m.id,m) for m in session.model_list() if isinstance(m, STL_Surface))
    from ..geometry.place import Places
    for st in surfs:
        if attributes_only:
            sid = st['id']
            if sid in sids:
                s = sids[sid]
            else:
                continue
        else:
            p = file_paths.find(st['path'])
            if p is None:
                continue
            s = read_stl(p, session)
            s.id = st['id']
        if 'displayed' in st:
            st['display'] = st['displayed']     # Fix old session files
        s.display = st['display']
        if 'positions' in st:
            s.positions = Places(place_array = st['positions'])
        if 'copies' in st:
            # Old session files.
            s.positions = Places(place_array = st['copies'])
        if 'color' in st:
            s.color = tuple(int(255*c) for c in st['color'])
        if 'colors' in st:
            s.colors = st['colors']
        if not attributes_only:
            session.add_model(s)
