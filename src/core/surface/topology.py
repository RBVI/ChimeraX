def check_surface_topology(triangles, name = ''):
    t = triangles
    print ('Checking surface topology', name, 'triangle count', len(t))
#    print ('triangles', t.shape, t[-3:,:])

    de = (t[:,0] == t[:,1]).sum() + (t[:,1] == t[:,2]).sum() + (t[:,2] == t[:,0]).sum()
    print ('degenerate edges', de, '(two vertices of triangle edge are the same)')
    
    edges = set()
    dupe = []
    for v1,v2 in zip(tuple(t[:,0]) + tuple(t[:,1]) + tuple(t[:,2]),
                     tuple(t[:,1]) + tuple(t[:,2]) + tuple(t[:,0])):
        if (v1,v2) in edges:
            dupe.append((v1,v2))
        edges.add((v1,v2))
    print('duplicate edges', len(dupe), '(more than one identical directed edge)')

    be = {}
    seexit = []
    for v1,v2 in edges:
        if not (v2,v1) in edges:
            if v1 in be:
                seexit.append(v1)
            be[v1] = v2
    print ('number of boundary edges', len(be))
    print ('two boundary edges exit same vertex, count', seexit)

    vert = set(tuple(t[:,0]) + tuple(t[:,1]) + tuple(t[:,2]))
    nonloop = []
    noout = []
    found = set()
    for v in vert:
        if v in found:
            continue
        vp = v
        while vp in be:
            found.add(vp)
            vn = be[vp]
            del be[vp]
            vp = vn
        if vp != v:
            nonloop.append((v, vp))
    print ('Boundaries that do not close', nonloop)
