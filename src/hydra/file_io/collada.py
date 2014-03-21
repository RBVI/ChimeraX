def read_collada_surfaces(path, session, color = (.7,.7,.7,1)):

    from os.path import basename
    from ..surface import Surface
    s = Surface(basename(path))

    from collada import Collada, polylist, triangleset
    c = Collada(path)
    from ..geometry.place import Place
    # TODO: For instanced collada objects don't want to use scene.objects() since this creates
    # BoundGeometry objects which has BoundPrimitive objects that have applied the positioning matrix
    # to the geometry.
    for g in c.scene.objects('geometry'):
        m = g.matrix        # 4 x 4 transformation matrix
        pl = Place(m[:3,:])
#        pl = Place(m[:,:3].transpose())
        if pl.is_identity():
            pl = None
        pl = None
        for p in g.primitives():
            if isinstance(p, polylist.BoundPolylist):
                p = p.triangleset()
            if not isinstance(p, triangleset.BoundTriangleSet):
                continue        # Skip line sets.
            t = p.vertex_index            # N by 3 array of vertex indices for triangles
            v = p.vertex                  # M by 3 array of floats for vertex positions
            ni = p.normal_index           # N by 3 array of normal indices for triangles
            n = p.normal		  # M by 3 array of floats for vertex normals
            # Collada allows different normals on the same vertex in different triangles,
            # but Hydra only allows one normal per vertex.
            from numpy import empty
            vn = empty(v.shape, n.dtype)
            vn[t.ravel(),:] = n[ni.ravel(),:]
            sp = s.new_piece()
            sp.geometry = v, t
            sp.normals = vn
            mat = p.material
            c = None
            if not mat is None:
                e = mat.effect
                if not e is None:
                    c = e.diffuse
            sp.color = color if c is None else c
            if not pl is None:
                sp.copies = [pl]

# Ignore texture coordinates for now.            
#            p.texcoord_indexset
#            p.texcoordset[t]

    return s
