# Display field lines for an electrostatic potential map.
# A specified number of field lines are computed with lines starting and
# ending above the specified minimum potential, and number of lines starting
# from near a charge proportional to the charge.
#
def show_field_lines(v, num_lines = 1000, min_potential = 10, step = 0.5,
                     color = (.7,.7,.7,1), line_width = 1, tube_radius = None,
                     circle_subdivisions = 12, markers = False, model_id = None):
                     
    if min_potential is None:
        min_potential = v.surface_levels[-1] if v.surface_levels else 10
        
    start_points, cutoff = find_starting_points(v, num_lines, min_potential)

    n = len(start_points)
    lines = []
    for i, xyz in enumerate(start_points):
        if i % 100 == 0:
            from chimera.replyobj import status
            status('Computing field line %d of %d' % (i,n))
        line = trace_field_line(v, xyz, cutoff, min_potential, step)
        lines.append(line)
     
    if markers:
        draw_marker_lines(lines, v, color, tube_radius, model_id)
    elif tube_radius is None:
        draw_mesh_lines(lines, v, color, line_width, model_id)
    else:
        draw_tube_lines(lines, v, color, tube_radius, circle_subdivisions, model_id)

# Computing starting points for field lines near charges with the number
# of lines starting around a charge proportional to the charge.  This
# assumes a q/r potential and works by comparing the potential to the
# magnitude of the gradient.
#
def find_starting_points_slow(v, num_lines, min_potential, step = 1):

    sxyz = []
    d = v.data
    isz, jsz, ksz = d.size
    for k in range(0,ksz,step):
        from chimera.replyobj import status
        status('plane %d' % k)
        for j in range(0,jsz,step):
            for i in range(0,isz,step):
                xyz = d.ijk_to_xyz((i,j,k))
                ua = abs(v.interpolated_values([xyz])[0])
                g = v.interpolated_gradients([xyz])[0]
                g2 = g[0]*g[0] + g[1]*g[1] + g[2]*g[2]
                if ua >= min_potential:
                    sxyz.append((g2/ua,xyz))
    sxyz.sort()
    sxyz = sxyz[-num_lines:]
    start_points = tuple(xyz for s,xyz in sxyz)
    c = sxyz[0][0]

    return start_points, c

# Faster version of find_starting_points() using numpy
def find_starting_points(v, num_lines, min_potential):

    points = v.grid_points(v.openState.xform.inverse())
    u = v.full_matrix().flat
    g, outside = v.interpolated_gradients(points, out_of_bounds_list = True)
    g2 = g[:,0]*g[:,0] + g[:,1]*g[:,1] + g[:,2]*g[:,2]
    g2[outside,:] = 0
    import numpy
    ua = numpy.abs(u)
    g2u = g2 / numpy.maximum(ua, min_potential)
    g2u[ua < min_potential] = 0
    si = g2u.argsort()[-num_lines:]
    start_points = points[si,:]
    c = g2u[si[0]]

    return start_points, c

# Follow gradient from given starting point.
def trace_field_line(v, xyz, c, min_potential, step = 0.5, skip = 1):

    d = v.data
    max_length = int(sum(d.size)/step)
    vstep = step*max(d.step)
    u = v.interpolated_values([xyz])[0]
    dir = -1 if u > 0 else 1

    line = [tuple(xyz)]
    from math import sqrt
    gp = None
    for i in range(1, max_length):
        u = v.interpolated_values([xyz])[0]
        g = v.interpolated_gradients([xyz])[0]
        g2 = g[0]*g[0] + g[1]*g[1] + g[2]*g[2]
        gn = sqrt(g2)
        ip = 1 if gp is None else g[0]*gp[0]+g[1]*gp[1]+g[2]*gp[2]
        if gn == 0 or ip < 0 or (g2 > c*abs(u) and u*dir > min_potential):
            if skip > 1 and i % skip != 1:
                line.append(xyz)
            break
        gp = g
        s = dir*vstep/gn
        xyz = tuple(xyz[a] + s*g[a] for a in (0,1,2))
        if i % skip == 0:
            line.append(xyz)

    return line 

# Place markers along each field line.
def draw_marker_lines(lines, v, color = (.7,.7,.7,1), radius = None,
                      model_id = None):

    from VolumePath import Marker_Set, Link
    mset = Marker_Set('%s field lines' % v.name)
    mset.marker_molecule(model_id).openState.xform = v.openState.xform
    if radius is None:
        radius = 0.5*max(v.data.step)
    for i,line in enumerate(lines):
        if i % 100 == 0:
            from chimera.replyobj import status
            status('Drawing line %d of %d' % (i,len(lines)))
        mprev = None
        for xyz in line:
            m = mset.place_marker(xyz, color, radius)
            if mprev:
                l = Link(m, mprev, color, radius)
                l.bond.halfbond = True
            mprev = m
    return mset

# Create mesh for field lines.
def draw_mesh_lines(lines, v, color = (.7,.7,.7,1), line_width = 1,
                    model_id = None):

    va, ta = line_mesh(lines)
    p = create_surface(va, ta, color, v, model_id)
    p.displayStyle = p.Mesh
    p.useLighting = False
    p.lineThickness = line_width
    return p.model

def create_surface(va, ta, color, v, model_id):

    import _surface
    s = _surface.SurfaceModel()
    s.name = '%s field lines' % v.name
    p = s.addPiece(va, ta, color)
    p.save_in_session = True

    from chimera import openModels, OpenModels
    id, subid = ((OpenModels.Default, OpenModels.Default)
                 if model_id is None else model_id)
    openModels.add([s], baseId = id, subid = subid)
    s.openState.xform = v.openState.xform
    return p

# Create zero area triangles that trace out lines.
def line_mesh(lines):

    np = sum(len(line) for line in lines)
    from numpy import empty, float32, int32
    va = empty((np,3), float32)
    nt = np - len(lines)
    ta = empty((nt,3), int32)
    p = t = 0
    for line in lines:
        lp = len(line)
        va[p:p+lp,:] = line
        tla = ta[t:t+lp-1,:]
        tla[:,0]= range(p, p+lp-1)
        tla[:,1] = tla[:,2] = tla[:,0]+1
        p += lp
        t += lp-1
    return va, ta

# Create tubes for field lines.
def draw_tube_lines(lines, v, color = (.7,.7,.7,1), radius = 1,
                    circle_subdivisions = 12, model_id = None):

    va, ta = tube_mesh(lines, radius, circle_subdivisions)
    p = create_surface(va, ta, color, v, model_id)
    return p.model

def tube_mesh(lines, radius, circle_subdivisions, end_caps = True,
              segment_subdivisions = 0):

    from VolumePath import tube, spline
    
    lv = []
    lt = []
    vc = 0
    for line in lines:
        ptlist = spline.natural_cubic_spline(line, segment_subdivisions,
                                             return_tangents = True)
        t = tube.Tube(ptlist, None, radius, circle_subdivisions, end_caps)
        lva, lta = t.geometry()
        lv.append(lva)
        lt.append(lta)
        lta += vc
        vc += len(lva)
    from numpy import concatenate
    va = concatenate(lv)
    ta = concatenate(lt)
    return va, ta
