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

# -----------------------------------------------------------------------------
#
polygon_colors = {
    3: (128,255,0,255),
    4: (255,0,128,255),
    5: (255,128,0,255),
    6: (0,128,255,255),
    7: (128,0,255,255),
    }
polygon_names = {
    3: 'triangle',
    4: 'square',
    5: 'pentagon',
    6: 'hexagon',
    7: 'septagon',
    }

# -----------------------------------------------------------------------------
#
def attach_polygons(session, edges, n, edge_length = 1, edge_thickness = 0.2,
                    edge_inset = 0.1, color = None, vertex_degree = None):

    if edges == 'selected':
        edges = selected_edges(session)

    if color is None:
        global polygon_colors
        color = polygon_colors[n] if n in polygon_colors else (.7,.7,.7,1)
            
    from math import pi, sin
    radius = 0.5*edge_length / sin(pi/n)
    if len(edges) == 0:
        p = Polygon(session, n, radius, edge_thickness, edge_inset, color = color)
        polygons =  [p]
    else:
        polygons = []
        for e in edges:
            if joined_edge(e) is None:
                p = Polygon(session, n, radius, edge_thickness, edge_inset,
                            color = color, marker_set = e.structure)
                polygons.append(p)
                join_edges(p.edges[0], e, vertex_degree)
    select_polygons(session, polygons)
    return polygons

# -----------------------------------------------------------------------------
#
def select_polygons(session, polygons):

    session.selection.clear()
    for p in polygons:
        for v in p.vertices:
            v.selected = True
        for e in p.edges:
            e.selected = True

# -----------------------------------------------------------------------------
#
def ordered_link_markers(link):

    m0, m1 = link.atoms
    idiff = m0.residue.number - m1.residue.number
    if idiff == 1 or idiff < -1:
        m0, m1 = m1, m0
    return m0, m1
    
# -----------------------------------------------------------------------------
#
def ordered_vertex_edges(marker):

    links = marker.bonds
    if len(links) != 2:
        return links
    l0, l1 = links
    if l0.atoms[1] is marker:
        return l0,l1
    return l1,l0
    
# -----------------------------------------------------------------------------
#
def connected_markers(m):

    mlist = [m]
    reached = set(mlist)
    i = 0
    while i < len(mlist):
        mi = mlist[i]
        for ml in mi.neighbors:
            if not ml in reached:
                mlist.append(ml)
                reached.add(ml)
        i += 1
    return mlist

# -----------------------------------------------------------------------------
#
class Polygon:

    def __init__(self, session, n, radius = 1.0, edge_thickness = 0.2, inset = 0.1,
                 color = (.7,.7,.7,1), marker_set = None, markers = None, edges = None):
 
        self.n = n
        self.radius = radius
        self.thickness = edge_thickness
        self.inset = inset
        self.using_inset = True
        self.marker_set = marker_set

        self._create_markers(session, color, markers, edges)

        if self.vertices:
            # Add placement method to get positions for placing molecule
            # copies on cage polygons.
            mset = self.marker_set
            if not hasattr(mset, 'placements'):
                p = lambda n,mset=mset: parse_placements(n, marker_set=mset)
                mset.placements = p

    # -------------------------------------------------------------------------
    #
    def _create_markers(self, session, color, markers = None, edges = None):

        initial_view = False
        if self.marker_set is None:
            msets = cage_marker_sets(session)
            if msets:
                self.marker_set = msets[0]
            else:
                initial_view = session.models.empty()
                self.marker_set = Cage(session)
                session.models.add([self.marker_set])

        if markers is None:
            mlist = []
            r = self.radius - self.inset
            edge_radius = 0.5*self.thickness
            from math import pi, cos, sin
            for a in range(self.n):
                angle = a*2*pi/self.n
                p = (r*cos(angle), r*sin(angle), 0)
                m = self.marker_set.create_marker(p, color, edge_radius)
                mlist.append(m)
        else:
            im = {a.residue.number:a for a in self.marker_set.atoms}
            mlist = [im[i] for i in markers]
            
        self.vertices = mlist
        for v in mlist:
            v.polygon = self

        if edges is None:
            elist = [add_link(m, mlist[(i+1)%self.n], color, edge_radius)
                     for i,m in enumerate(mlist)]
        else:
            ie = {tuple(a.residue.number for a in b.atoms):b for b in self.marker_set.bonds}
            elist = [ie[e] for e in edges]
            
        self.edges = elist
        for e in elist:
            e.polygon = self

        if initial_view:
            session.main_view.initial_camera_view()

    # -------------------------------------------------------------------------
    #
    def center(self):

        return marker_center(self.vertices)

    # -------------------------------------------------------------------------
    #
    def normal(self):

        x0,x1,x2 = [m.coord for m in self.vertices[:3]]
        from chimerax.geometry import normalize_vector, cross_product
        n = normalize_vector(cross_product(x1-x0, x2-x1))
        return n
        
    # -------------------------------------------------------------------------
    #
    def edge_length(self):

        from math import pi, sin
        edge_length = 2*self.radius*sin(pi/self.n)
        return edge_length
        
    # -------------------------------------------------------------------------
    #
    def use_inset(self, use):

        use = bool(use)
        if self.using_inset == use:
            return

        self.using_inset = use
        self.reposition_vertices()

    # -------------------------------------------------------------------------
    #
    def reposition_vertices(self):

        r = self.radius
        ri = (r - self.inset) if self.using_inset else r
        c = self.center()
        from math import sqrt
        for m in self.vertices:
            d = m.coord-c
            f = ri/sqrt((d*d).sum())
            m.coord = c + f*d

    # -------------------------------------------------------------------------
    #
    def inset_scale(self):

        r = self.radius
        s = (r - self.inset) / r if self.using_inset else 1.0
        return s
        
    # -------------------------------------------------------------------------
    #
    def neighbor_polygons(self):

        plist = []
        reached = set([self])
        for l0 in self.edges:
            l1 = joined_edge(l0)
            if l1:
                pn = l1.polygon
                if not pn in reached:
                    reached.add(pn)
                    plist.append(pn)
        return plist

    # -------------------------------------------------------------------------
    # Marker position without inset.
    #
    def vertex_xyz(self, m):

        c = self.center()
        xyz = c + (m.coord - c)/self.inset_scale()
        return xyz
    
    # -------------------------------------------------------------------------
    #
    def resize(self, edge_length, edge_thickness, edge_inset):

        from math import pi, sin
        radius = 0.5*edge_length / sin(pi/self.n)
        if radius != self.radius or edge_inset != self.inset:
            self.radius = radius
            self.inset = edge_inset
            self.reposition_vertices()
            c = self.center()
            for m in self.vertices:
                xyz = c + (m.coord - c)/self.inset_scale()

        self.thickness = edge_thickness
        r = 0.5*edge_thickness
        for m in self.vertices:
            m.radius = r
        for e in self.edges:
            e.radius = r
        
    # -------------------------------------------------------------------------
    #
    def delete(self):

        elist = [joined_edge(e) for e in self.edges if joined_edge(e)]
        unjoin_edges(elist)
        for v in self.vertices:
            v.delete()

    # -------------------------------------------------------------------------
    #
    def polygon_state(self):
        data = {attr:getattr(self, attr) for attr in ('n', 'radius', 'thickness', 'inset')}
        data['markers'] = [a.residue.number for a in self.vertices]
        data['edges'] = [tuple(a.residue.number for a in b.atoms) for b in self.edges]
        data['version'] = 1
        return data

    # -------------------------------------------------------------------------
    #
    @staticmethod
    def polygon_from_state(state, marker_set):
        p = Polygon(marker_set.session, state['n'], radius = state['radius'],
                    edge_thickness = state['thickness'], inset = state['inset'],
                    marker_set = marker_set, markers = state['markers'], edges = state['edges'])
        return p

# -----------------------------------------------------------------------------
#
def marker_center(markers):

    from numpy import sum
    c = sum([m.coord for m in markers], axis=0) / len(markers)
    return c
        
# -----------------------------------------------------------------------------
#
def joined_edge(e):

    return getattr(e, 'joined_edge', None)
    
# -----------------------------------------------------------------------------
#
def join_edges(l0, l1, vertex_degree = None):

    join(l0, l1)
    join(l1, l0)

    optimize_placement(l0.polygon, vertex_degree)

    if not vertex_degree is None:
        # Join edges to achieve given vertex degree
        m1, m2 = l0.atoms
        vertex_join_edges(m1, vertex_degree)
        vertex_join_edges(m2, vertex_degree)

    return True

# -----------------------------------------------------------------------------
#
def join(l0, l1):

    l0j = joined_edge(l0)
    if l0j:
        unjoin_edges([l0j])
    l0.joined_edge = l1
    l0.color = lighten_color(l0.color)

# -----------------------------------------------------------------------------
# If degree polygons are joined around a vertex but one pair of edges is not
# joined, then join that pair of edges.
#
def vertex_join_edges(m, degree):

    llist = ordered_vertex_edges(m)
    if len(llist) != 2:
        return          # Broken polygon.
    l0, l1 = llist

    e1, d1 = vertex_next_unpaired_edge(l1, 1)
    e0, d0 = vertex_next_unpaired_edge(l0, -1)
    if e0 and e1 and 1 + d0 + d1 == degree:
        join_edges(e0, e1, degree)

# -----------------------------------------------------------------------------
#
def vertex_next_unpaired_edge(edge, direction):

    e = edge
    d = 0
    while True:
        ej = joined_edge(e)
        if ej is None:
            break
        en = next_polygon_edge(ej, direction)
        if en is edge or en is None:
            e = None   # Vertex surrounded by joined polygons or polygon broken
            break
        e = en
        d += 1
    return e, d

# -----------------------------------------------------------------------------
#
def next_polygon_edge(edge, direction):

    m0, m1 = ordered_link_markers(edge)
    
    s = set(m1.bonds if direction == 1 else m0.bonds)
    s.remove(edge)
    if len(s) == 1:
        return s.pop()
    return None

# -----------------------------------------------------------------------------
#
def unjoin_edges(edges):

    lset = set(edges)
    for link in lset:
        lj = joined_edge(link)
        if lj:
            delattr(lj, 'joined_edge')
            lj.color = darken_color(lj.color)
            delattr(link, 'joined_edge')
            link.color = darken_color(link.color)

# -----------------------------------------------------------------------------
#
def delete_polygons(plist):

    for p in plist:
        p.delete()
        
# -----------------------------------------------------------------------------
#
def optimize_placement(polygon, vertex_degree):

    p = optimized_placement(polygon, vertex_degree)
    for a in polygon.vertices:
        a.coord = p * a.coord

# -----------------------------------------------------------------------------
#
def optimized_placement(polygon, vertex_degree):

    lpairs = []
    for l0 in polygon.edges:
        l0j = joined_edge(l0)
        if l0j:
            lpairs.append((l0,l0j))

    from chimerax.geometry import Place, align_points
    if len(lpairs) == 0:
        tf = Place()
    elif len(lpairs) == 1:
        l0,l1 = lpairs[0]
        tf = edge_join_transform(l0, l1, vertex_degree)
    else:
        xyz0 = []
        xyz1 = []
        for l0, l1 in lpairs:
            xyz0.extend(edge_alignment_points(l0))
            xyz1.extend(reversed(edge_alignment_points(l1)))
        from numpy import array
        tf, rms = align_points(array(xyz0), array(xyz1))
    return tf

# -----------------------------------------------------------------------------
#
def edge_alignment_points(edge):

    p = edge.polygon
    c = p.center()
    s = 1.0/p.inset_scale()
    pts = [c + s*(m.coord-c) for m in ordered_link_markers(edge)]
    return pts

# -----------------------------------------------------------------------------
# Calculate transform aligning edge of one polygon with edge of another polygon.
# For hexagons put them in the same plane abutting each other.
# For pentagons make them non-coplanar so that optimization works without
# requiring symmetry breaking.
#
def edge_join_transform(link, rlink, vertex_degree):

    f0 = edge_coordinate_frame(rlink)
    f1 = edge_coordinate_frame(link)
    from chimerax.geometry import Place
    r = Place(((-1,0,0,0),
               (0,-1,0,0),
               (0,0,1,0)))     # Rotate 180 degrees about z.

    if vertex_degree is None:
        ea = 0
    else:
        a = 180-360.0/link.polygon.n
        ra = 180-360.0/rlink.polygon.n
        ea = vertex_degree*(a + ra) - 720
    if ea != 0:
        from math import sin, cos, pi
        a = -pi/6 if ea < 0 else pi/6
        rx = Place(((1,0,0,0),
                    (0,cos(a),sin(a),0),
                    (0,-sin(a),cos(a),0)))
        r = rx * r
        
    tf = f0 * r * f1.inverse()
    
    return tf

# -----------------------------------------------------------------------------
# 3 by 4 matrix mapping x,y,z coordinates to center of edge with x along edge,
# and y directed away from center of the polygon, and z perpendicular to the
# plane of the polygon.
#
def edge_coordinate_frame(edge):

    x0, x1 = edge_alignment_points(edge)
    p = edge.polygon
    c = p.center()
    c01 = 0.5 * (x0+x1)
    from chimerax.geometry import cross_product, normalize_vector, Place
    xa = normalize_vector(x1 - x0)
    za = normalize_vector(cross_product(xa, c01-c))
    ya = cross_product(za, xa)
    tf = Place(((xa[0],ya[0],za[0],c01[0]),
                (xa[1],ya[1],za[1],c01[1]),
                (xa[2],ya[2],za[2],c01[2])))
    return tf

# -----------------------------------------------------------------------------
#
def optimize_shape(session, fixedpolys = set(), vertex_degree = None):

    links = selected_edges(session)
    if len(links) == 0:
        links = sum([list(mset.bonds) for mset in cage_marker_sets(session)], [])
    reached = set([link.polygon for link in links])
    plist = list(reached)
    i = 0
    while i < len(plist):
        p = plist[i]
        if not p in fixedpolys:
            optimize_placement(p, vertex_degree)
        for pn in p.neighbor_polygons():
            if not pn in reached:
                reached.add(pn)
                plist.append(pn)
        i += 1
    
# -----------------------------------------------------------------------------
#
def polygon_counts(plist):

    c = {}
    for p in plist:
        c[p.n] = c.get(p.n,0) + 1
    pctext = ', '.join(['%d %s%s' % (cnt, polygon_names.get(n,'ngon'),
                                     's' if cnt > 1 else '')
                        for n, cnt in sorted(c.items())])
    if len(c) > 1:
        pctext += ', %d polygons' % len(plist)
    return pctext

# -----------------------------------------------------------------------------
#
def expand(polygons, distance):

    for p in polygons:
        d = distance * p.normal()
        for m in p.vertices:
            m.coord = d + m.coord

# -----------------------------------------------------------------------------
#
def align_molecule():

    # TODO: Not ported.
    from chimera import selection
    atoms = selection.currentAtoms(ordered = True)
    mols = set([a.molecule for a in atoms])
    if len(mols) != 1:
        return
    mol = mols.pop()
    molxf = mol.openState.xform
    from Molecule import atom_positions
    axyz = atom_positions(atoms, molxf)
    from numpy import roll, float32, float64
    from Matrix import xform_matrix, xform_points
    from chimera.match import matchPositions
    xflist = []
    for mset in cage_marker_sets():
        for p in polygons(mset):
            if p.n == len(atoms):
                c = p.center()
                vxyz = [p.vertex_xyz(m) for m in p.vertices]
                exyz = (0.5*(vxyz + roll(vxyz, 1, axis = 0))).astype(float32)
                xform_points(exyz, mset.transform(), molxf)
                xf, rms = matchPositions(exyz.astype(float64),
                                         axyz.astype(float64))
                xflist.append(xf)

    molxf.multiply(xflist[0])
    mol.openState.xform = molxf

    import MultiScale
    mm = MultiScale.multiscale_manager()
    tflist = [xform_matrix(xf) for xf in xflist]
    mm.molecule_multimer(mol, tflist)

# -----------------------------------------------------------------------------
#
def polygons(mset):

    return set([m.polygon for m in mset.atoms])

# -----------------------------------------------------------------------------
#
def selected_edges(session):

    from chimerax.atomic import selected_bonds
    bonds = selected_bonds(session)
    from numpy import array
    cbonds = bonds.filter(array([hasattr(b, 'polygon') for b in bonds], bool))
    return cbonds

# -----------------------------------------------------------------------------
#
def selected_vertices(session):

    from chimerax.atomic import selected_atoms
    atoms = selected_atoms(session)
    from numpy import array
    catoms = atoms.filter(array([hasattr(a, 'polygon') for a in atoms], bool))
    return catoms

# -----------------------------------------------------------------------------
#
def selected_polygons(session, full_cages = False, none_implies_all = False):

    plist = set([e.polygon for e in selected_edges(session)] +
                [v.polygon for v in selected_vertices(session)])
    if full_cages:
        cages = set([p.marker_set for p in plist])
        plist = sum([cage_polygons(c) for c in cages], [])
    if none_implies_all and len(plist) == 0:
        plist = sum([cage_polygons(c) for c in cage_marker_sets()], [])
    return plist

# -----------------------------------------------------------------------------
#
def cage_polygons(marker_set):

    plist = list(set([m.polygon for m in marker_set.atoms
                      if hasattr(m, 'polygon')]))
    return plist

# -----------------------------------------------------------------------------
#
def selected_cages():

    plist = selected_polygons()
    mslist = list(set([p.marker_set for p in plist if p.vertices]))
    return mslist

# -----------------------------------------------------------------------------
#
from chimerax.markers import MarkerSet
class Cage(MarkerSet):
    def __init__(self, session, name = 'Cage'):
        MarkerSet.__init__(self, session, name = name)
    def take_snapshot(self, session, flags):
        ei = {b:tuple(a.residue.number for a in b.atoms) for b in self.bonds}
        joined_edges = [(ei[b], ei[b.joined_edge]) for b in self.bonds if hasattr(b, 'joined_edge')]
        polygons = set([a.polygon for a in self.atoms if hasattr(a, 'polygon')] +
                       [b.polygon for b in self.bonds if hasattr(b, 'polygon')])
        pstates = [p.polygon_state() for p in polygons]
        data = {'marker set state': MarkerSet.take_snapshot(self, session, flags),
                'joined edges': joined_edges,
                'polygons': pstates,
                'version': 1}
        return data
    @staticmethod
    def restore_snapshot(session, data):
        s = MarkerSet.restore_snapshot(session, data['marker set state'])
        ei = {tuple(a.residue.number for a in b.atoms):b for b in s.bonds}
        for ei1,ei2 in data['joined edges']:
            e1,e2 = ei[ei1],ei[ei2]
            e1.joined_edge = e2
            e2.joined_edge = e1
        for pstate in data['polygons']:
            Polygon.polygon_from_state(pstate, s)
        return s
        
    
# -----------------------------------------------------------------------------
#
def add_link(a1, a2, color, radius):
    from chimerax.markers import create_link
    return create_link(a1, a2, rgba = color, radius = radius)

# -----------------------------------------------------------------------------
#
def cage_marker_sets(session):
    from chimerax.markers import MarkerSet
    return [m for m in session.models.list(type = MarkerSet) if m.name == 'Cage']
    
# -----------------------------------------------------------------------------
#
def lighten_color(rgba):
    r,g,b,a = rgba
    return (r + (255-r)//2, g + (255-g)//2, b + (255-b)//2, a)
def darken_color(rgba):
    r,g,b,a = rgba
    return (max(0, r - (255-r)), max(0, g - (255-g)), max(0, b - (255-b)), a)

# -----------------------------------------------------------------------------
#
def toggle_inset():

    clist = selected_cages()
    if len(clist) == 0:
        clist = cage_marker_sets()
    for c in clist:
        for p in polygons(c):
            p.use_inset(not p.using_inset)

# -----------------------------------------------------------------------------
#
def use_inset(use = True):

    clist = selected_cages()
    if len(clist) == 0:
        clist = cage_marker_sets()
    for c in clist:
        for p in polygons(c):
            p.use_inset(use)

# -----------------------------------------------------------------------------
#
def scale(plist, f):

    mlist = sum([p.vertices for p in plist],[])
    c = marker_center(mlist)
    for p in plist:
        shift = (f-1.0)*(p.center() - c)
        for m in p.vertices:
            m.coord = shift + m.coord
        p.resize(f*p.edge_length(), f*p.thickness, f*p.inset)
    
# -----------------------------------------------------------------------------
#
def make_mesh(color = (.7,.7,.7,1), edge_thickness = 0.4):

    clist = selected_cages()
    if len(clist) == 0:
        clist = cage_marker_sets()
    for c in clist:
        make_polygon_mesh(polygons(c), color, edge_thickness)

# -----------------------------------------------------------------------------
#
def make_polygon_mesh(plist, color, edge_thickness):

    # Find sets of joined vertices.
    mt = {}
    for p in plist:
        for e in p.edges:
            m0, m1 = ordered_link_markers(e)
            for m in (m0, m1):
                if not m in mt:
                    mt[m] = set([m])
            je = joined_edge(e)
            if je:
                jm1, jm0 = ordered_link_markers(je)
                mt[m0].add(jm0)
                mt[m1].add(jm1)

    # Create markers at average postions of joined markers.
    from VolumePath import Marker_Set, Link
    mset = Marker_Set('Mesh')
    mm = {}
    r = 0.5*edge_thickness
    from numpy import mean
    for m, mg in mt.items():
        if not m in mm:
            xyz = mean([me.polygon.vertex_xyz(me) for me in mg], axis=0)
            mc = mset.place_marker(xyz, color, r)
            for me in mg:
                mm[me] = mc

    # Create links between markers.
    et = {}
    for p in plist:
        for e in p.edges:
            em1, em2 = e.atoms
            m1, m2 = mm[em1], mm[em2]
            if not (m1,m2) in et:
                et[(m1,m2)] = et[(m2,m1)] = Link(m1, m2, color, r)

    return mset

# -----------------------------------------------------------------------------
# Return list of transforms from origin to each polygon standard reference
# frame for polygons with n sides for placing molecule copies on cage.
#
def parse_placements(name, marker_set):

    if name.startswith('pn'):
        ns = name[2:]
        each_edge = True
    elif name.startswith('p'):
        ns = name[1:]
        each_edge = False
    else:
        from chimerax.core.errors import UserError
        raise UserError('Symmetry placement must start with "p" or "pn", got "%s"' % name)

    try:
        n = int(ns)
    except Exception:
        from chimerax.core.errors import UserError
        raise UserError('Symmetry placement must be "p" or "pn" followed by an integer, got "%s"' % name)

    return placements(n, marker_set, each_edge)

# -----------------------------------------------------------------------------
# Return list of transforms from origin to each polygon standard reference
# frame for polygons with n sides for placing molecule copies on cage.
#
def placements(n, marker_set, each_edge = False):

    plist = polygons(marker_set)
    from chimerax.geometry import Places
    tflist = Places([polygon_coordinate_frame(p) for p in plist if p.n == n])

    if each_edge:
        from chimerax import geometry
        tflist = tflist * geometry.cyclic_symmetry_matrices(n)

    return tflist

# -----------------------------------------------------------------------------
# Map (0,0,0) to polygon center, (0,0,1) to polygon normal and (1,0,0) to
# normalized axis from center to the first polygon marker.
#    
def polygon_coordinate_frame(polygon):

    p = polygon
    c = p.center()
    za = p.normal()
    xa = p.vertices[0].coord - c
    from chimerax.geometry import normalize_vector, cross_product, Place
    ya = normalize_vector(cross_product(za, xa))
    xa = normalize_vector(cross_product(ya, za))
    tf = [(xa[a], ya[a], za[a], c[a]) for a in (0,1,2)]
    return Place(tf)
