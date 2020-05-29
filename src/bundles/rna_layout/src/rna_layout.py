# -----------------------------------------------------------------------------
# Layout single strand RNA with specified base pairing.
#

# Series of stems and loops.
class Circuit:
    def __init__(self, stems_and_loops):
        self.stems_and_loops = stems_and_loops

    # -------------------------------------------------------------------------
    # Compute 3-d layout for circuit with stems and loops placed on a circle.
    # The layout leaves space for a stem at the bottom (negative y axis) of the
    # circle.
    #
    # The returned coordinates are a dictionary mapping residue number to a
    # transform (3 by 4 matrix) which takes (0,0,0) to the N1 or N9 base atom
    # which joins the sugar and the x-axis is approximately in the direction
    # from phosphorous toward this base atom.
    #
    def layout(self, params, pattern = 'circle'):

        # Compute stem and loop segment layouts.
        sl = self.stems_and_loops
        n = len(sl)
        segs = []
        for e in sl:
            segs.extend(e.layout(params, n))

        if pattern == 'circle':
            coords = self.circle_layout(segs, params)
        elif pattern == 'line':
            coords = self.straight_layout(segs, params)
        elif pattern == 'helix':
            curve = HelixCurve(params.helix_radius, params.helix_rise)
            coords = self.curve_layout(segs, params, curve)
        elif pattern == 'sphere':
            curve = SphereSpiral(params.sphere_radius, params.sphere_turns)
            coords = self.curve_layout(segs, params, curve)
            
        return coords

    def circle_layout(self, segs, params):

        coords = {}
        p = params

        # Compute circle radius
        seg_lengths = [seg.width for seg in segs] + [seg.pad for seg in segs]  + [p.pair_width, p.loop_spacing]
        wc = value_counts(seg_lengths)
        radius = polygon_radius(list(wc.items()))

        # Layout loops and stems on a circle
        stem_angle_step = circle_angle(p.pair_width, radius)
        gap_step = circle_angle(p.loop_spacing, radius)
        angle = 0.5 * stem_angle_step + gap_step
        from math import pi, sin, cos
        a = 0.5*stem_angle_step*pi/180
        from random import random
        from chimerax.geometry import translation, rotation
        stf = translation((radius*sin(a), -radius+radius*cos(a), 0))
        for seg in segs:
            angle_step = circle_angle(seg.width, radius)
            rtf = rotation((0,0,1), 180 - 0.5*angle_step)
            if p.branch_tilt != 0:
                btf = rotation((1,0,0), p.branch_tilt * (1-2*random()))
                rtf = rtf * btf
            ptf = rotation((0,0,1), -angle, center = (0,radius,0))
            ctf = stf * ptf * rtf
            for b, tf in seg.placements.items():
                coords[b] = ctf * tf
            # Next position along circle.
            gap_step = circle_angle(seg.pad, radius)
            angle += angle_step + gap_step

        return coords

    def straight_layout(self, segs, params):

        coords = {}
        s = 0
        from chimerax.geometry import translation, rotation
        for i, seg in enumerate(segs):
            rtf = rotation((1,0,0), params.branch_twist * i)
            stf = translation((s, 0, 0))
            ptf = stf * rtf
            for b, tf in seg.placements.items():
                coords[b] = ptf * tf
            # Next position along line
            s += seg.width + seg.pad

        return coords

    def curve_layout(self, segs, params, curve):

        seg_lengths = [(seg.width, seg.pad) for seg in segs]
        placements = segments_on_curve(curve, seg_lengths)

        coords = {}
        from random import random
        from chimerax.geometry import rotation
        random_tilt = params.branch_tilt
        for seg, place in zip(segs, placements):
            tf = place
            if random_tilt != 0:
                tf = tf * rotation((1,0,0), random_tilt * (1-2*random()))
            for b, stf in seg.placements.items():
                coords[b] = tf * stf

        return coords

# Duplex RNA segment.
class Stem:
    def __init__(self, base5p, base3p, length, circuit):
        self.base5p = base5p
        self.base3p = base3p
        self.length = length
        self.circuit = circuit  # Circuit at far end of stem

    def layout(self, params, circuit_segments):

        p = params
        coords = {}
        from chimerax.geometry import translation, rotation, Place
        rtf = rotation((0,1,0), p.stem_twist,
                       center = (0.5*p.pair_width,0,-p.pair_off_axis))
        ttf = translation((0,p.pair_spacing,0))
        stf = ttf * rtf
        tf1 = Place()
        tf2 = Place(matrix = ((-1,0,0,p.pair_width),(0,-1,0,0),(0,0,1,0)))
        for i in range(self.length):
            coords[self.base5p+i] = tf1
            coords[self.base3p-i] = tf2
            tf1 = stf * tf1
            tf2 = stf * tf2

        # Added circuit at end of stem.
        ccoords = self.circuit.layout(p)

        # Motion to end of stem assumes the stem end x-axis points
        # to base pair partner.
        stf = coords[self.base5p + self.length - 1]
        for b, tf in ccoords.items():
            coords[b] = stf * tf

        return [Segment(coords, p.pair_width, p.loop_spacing)]

# Single strand RNA segment.
class Loop:
    def __init__(self, base5p, length):
        self.base5p = base5p
        self.length = length

    def layout(self, params, circuit_segments):
#        segs = self.horseshoe_segments(params, circuit_segments)
        segs = self.helix_segments(params, circuit_segments)
        return segs

    def helix_segments(self, params, circuit_segments):
        n = self.length
        b = self.base5p
        spacing = params.loop_spacing

        # Nucleotide orientation
        from chimerax.geometry import Place
        r90 = Place(matrix = ((0,1,0,0),(-1,0,0,0),(0,0,1,0)))
        
        helix_length = params.helix_loop_size
        if n < helix_length:
            # Make each nucleotide a segment.  Not enough nucleotides to make a helix.
            segs = singleton_segments(b, n, spacing, r90)
        else:
            segs = []
            nh = n // helix_length
            ne = n % helix_length
            base_index = b
            start_pad = ne//2
            if start_pad > 0:
                segs.extend(singleton_segments(base_index, start_pad, spacing, r90))
                base_index += start_pad
            nleft = n - start_pad
            while nleft >= helix_length:
                hlen = 3*helix_length
                if hlen > nleft: hlen = 2*helix_length
                if hlen > nleft: hlen = helix_length
                seg = self.helix_segment(base_index, hlen, spacing, params.helix_loop_rise)
                base_index += hlen
                segs.append(seg)
                nleft -= hlen
            if nleft > 0:
                segs.extend(singleton_segments(base_index, nleft, spacing, r90))

        return segs

    def helix_segment(self, base_index, count, spacing, rise):
        from math import pi, sin
        angle = 2*pi / count
        angle_deg = angle*180/pi
        radius = 0.5 * spacing / sin(angle/2)
        from chimerax.geometry import translation, rotation, Place
        orient = rotation((0,1,0), 25) * rotation((1,0,0), 230)
        step = translation((rise/count, 0, 0)) * rotation((1,0,0), angle_deg, center=(0,radius,0))
        p = Place()
        place = {}
        for i in range(count):
            place[base_index + i] = p * orient
            p = step * p
        return Segment(place, rise, pad = 0)
    
    def horseshoe_segments(self, params, circuit_segments):    
        n = self.length
        b = self.base5p
        spacing = params.loop_spacing
        
        # Nucleotide orientation
        from chimerax.geometry import Place
        r90 = Place(matrix = ((0,1,0,0),(-1,0,0,0),(0,0,1,0)))

        curve_length = params.min_lobe_size  # Number of nucleotides in curved part of horseshoe
        if n < curve_length:
            # Make each nucleotide a segment.  Not enough nucleotides to make a horseshoe.
            segs = singleton_segments(b, n, spacing, r90)
        else:
            # Create as many horseshoe segments as needed.
            mxs = params.max_lobe_size
            lsp = params.lobe_spacing_for_stem_loop() if circuit_segments == 1 else params.lobe_spacing
            # Start with lsp single nucleotide segments for spacing
            segs = singleton_segments(b, lsp, spacing, r90)
            nl = n-lsp  # number of remaining nucleotides
            bl = b+lsp	# first nucleotide index
            while nl >= curve_length+lsp:
                c = (nl+mxs+lsp-1) // (mxs+lsp)  # Number of horseshoes
                ls = nl//c - lsp  # Nucleotides per horseshoe + spacer
                if (ls - curve_length) % 2 == 1:
                    ls -= 1  # Make sure have even number of nucleotides to make two sides of horseshoe
                side_length = (ls - curve_length) // 2  # Number of nucleotides in one horsehoe side.
                # Create horseshoe segment.
                seg = self.horseshoe_segment(side_length, curve_length, spacing, bl)
                segs.append(seg)
                ll = 2*side_length + curve_length  # Nucleotides used by two sides and curve of horseshoe.
                bl += ll  # Increment current nucleotide index
                nl -= ll  # Decrement number of nucleotides remaining
                # Add lsp single nucleotide segments for spacing
                segs.extend(singleton_segments(bl, lsp, spacing, r90))
                bl += lsp
                nl -= lsp
            # Any leftover nucleotides with too few to make a horseshoe, make each its own segment.
            segs.extend(singleton_segments(bl, nl, spacing, r90))
        return segs

    def horseshoe_segment(self, side_length, semicircle_length, spacing, start_index):
        # Make a horseshoe segment with two sides each with side_length nucleotides
        # parallel the y axis, curved top has semicircle_length nucleotides, spacing of
        # nucleotides is spacing.  Starting nucleotide index is start_index.
        from chimerax.geometry import Place, rotation
        c = {}
        rtf = rotation((0,0,1), 40)	# Reduce backbone connection bond lengths.
        rtf = rotation((1,0,0), -35)*rtf	# Reduce backbone connection bond lengths.
        # Horseshoe side going up in y.
        ns = side_length
        b = start_index
        ls = spacing
        for i in range(ns):
            c[b+i] = Place(matrix = ((1,0,0,0),(0,1,0,i*ls),(0,0,1,0))) * rtf
        from math import pi, cos, sin
        nc = semicircle_length
        r = 0.5*ls/sin(0.5*pi/(nc-1))
        for i in range(nc):
            a = i*pi/(nc-1)
            ca, sa = cos(a), sin(a)
            # Horeshoe curve
            c[b+ns+i] = Place(matrix = ((ca,sa,0,r*(1-ca)),(-sa,ca,0,r*sa+ns*ls),(0,0,1,0))) * rtf
        # Horseshoe side going down in y.
        for i in range(ns):
            c[b+ns+nc+i] = Place(matrix = ((-1,0,0,2*r),(0,-1,0,(ns-1-i)*ls),(0,0,1,0))) * rtf
        return Segment(c, 2*r, spacing)

class Segment:
    '''
    Placements for a stretch of one or more contiguous nucleotides.
    Width is the distance (Angstroms) between first and last nucleotides.
    Pad is space from last nucleotide to the first of the next segment.
    '''
    def __init__(self, placements, width, pad):
        self.width = width
        self.pad = pad
        self.placements = placements  # Map nucleotide index to Place
    @property
    def count(self):
        return len(self.placements)
    
def singleton_segments(base_index, count, spacing, placement):
    return [Segment({base_index + i: placement}, width = 0, pad = spacing) for i in range(count)]
        
class Layout_Parameters:

    def __init__(self, loop_spacing = 6.5,
                 min_lobe_size = 8, max_lobe_size = 28, lobe_spacing = 0,
                 pair_spacing = 3, pair_width = 10, pair_off_axis = 2,
                 stem_twist = 36, branch_twist = 145, branch_tilt = 0):
        self.loop_spacing = loop_spacing	# Spacing of nucleotides in horseshoe, Angstroms.
        self.min_lobe_size = min_lobe_size	# Number of nucleotides in horseshoe curved part
        self.max_lobe_size = max_lobe_size	# Max nucleotides in horseshoe sides + curved part
        self.lobe_spacing = lobe_spacing	# Number of nucleotides to place between 2 horseshoes
        self.pair_spacing = pair_spacing	# Spacing of one base pair to next base pair, Angstroms.
        self.pair_width = pair_width		# Phophorous to phosporous distance in base pair, Angstroms.
        self.pair_off_axis = pair_off_axis
        self.stem_twist = stem_twist		# Twist per base-pair in a stem (degrees).
        self.branch_twist = branch_twist	# Twist per loop or stem in straight pattern (degrees).
        self.branch_tilt = branch_tilt		# Random tilt magnitude (degrees) for circle pattern

        self.helix_radius = 300			# Radius for helix layout, Angstroms
        self.helix_rise = 50			# Rise per turn for helix layout, Angstroms

        self.sphere_radius = 370		# Spiral sphere layout radius, Angstroms
        self.sphere_turns = 16			# Spiral sphere turns from top to botton, count.

        self.helix_loop_size = 8		# Number of nucleotides in a loop layed out as a helix
        self.helix_loop_rise = 20		# Rise in loop helix over one turn, Angstroms

    def lobe_spacing_for_stem_loop(self):
        '''
        A stem with a large loop can make a horseshoe for the loop where
        the ends of the horseshoe are much further apart than the ends of the stem.
        That creates a long bond between the stem and the loop.  To avoid that
        put some of the loop into spacers (single nucleotide segments) by increasing
        the lobe spacing (number of nucleotides between two horseshoes) to the
        value computed by this routine.

        Lobe spacing is set so that the spacers on each side of the horseshoe
        when attached to the stem span a width equal to the width of the horseshoe.
        '''
        from math import pi, ceil
        gap = pi*self.lobe_radius() - (self.pair_width + 2*self.loop_spacing)
        els = max(0, int(ceil(0.5*gap / self.loop_spacing)))
        return els

    def lobe_radius(self):
        from math import pi, sin
        r = 0.5*self.loop_spacing/sin(0.5*pi/(self.min_lobe_size-1))
        return r

# -----------------------------------------------------------------------------
# Create a Circuit object representing RNA topology from a list of 
# (start, end, length) base paired segments.
#
def circuit(pair_map, start, end):

    sl = []
    s = start
    while s <= end:
        if s in pair_map and pair_map[s] <= end:
            e = pair_map[s]
            l = 1
            while pair_map.get(s+l) == e-l:
                l += 1
            sl.append(Stem(s, e, l, circuit(pair_map, s+l, e-l)))
            s = e + 1
        else:
            l = 1
            while s+l not in pair_map and s+l <= end:
                l += 1
            sl.append(Loop(s,l))
            s += l
    c = Circuit(sl)
    return c

# -----------------------------------------------------------------------------
#
def polygon_radius(lnlist):

    from math import pi, asin
    l = sum(n*l for l,n in lnlist)
    r0 = l/(2*pi)
    r1 = l/2
    while r1-r0 > 1e-5 * l:
        rm = 0.5*(r0+r1)
        if sum(n*asin(min(1.0,l/(2*rm))) for l,n in lnlist) > pi:
            r0 = rm
        else:
            r1 = rm
    return rm

# -----------------------------------------------------------------------------
#
def circle_angle(side_length, radius):

    from math import asin, pi
    return 2*asin(min(1.0,0.5*side_length/radius)) * (180/pi)

# -----------------------------------------------------------------------------
#
def value_counts(values):

    vc = {}
    for v in values:
        if v in vc:
            vc[v] += 1
        else:
            vc[v] = 1
    return vc

# -----------------------------------------------------------------------------
#
class HelixCurve:
    def __init__(self, radius, rise):
        self._radius = radius
        self._rise = rise  # rise per turn
    def position(self, t):
        r = self._radius
        from math import pi, cos, sin
        dz = self._rise / (2*pi)
        from numpy import array, float64
        return array((r*cos(t), r*sin(t), dz*t), float64)
    def velocity(self, t):
        r = self._radius
        from math import pi, cos, sin
        dz = self._rise / (2*pi)
        from numpy import array, float64
        return array((-r*sin(t), r*cos(t), dz), float64)
    def normal(self, t):
        from math import cos, sin
        from numpy import array, float64
        return array((cos(t), sin(t), 0), float64)

# -----------------------------------------------------------------------------
#
class SphereSpiral:
    def __init__(self, radius, turns):
        self._radius = radius
        self._turns = turns
    def position(self, t):
        r = self._radius
        wt = 2 * self._turns * t
        # r * (sin(t) * cos(w*t), sin(t) * sin(w*t), cos(t))
        from math import cos, sin
        rs = r*sin(t)
        from numpy import array, float64
        p = array((rs * cos(wt), rs * sin(wt), r * cos(t)), float64)
        return p
    def velocity(self, t):
        r = self._radius
        w = 2 * self._turns
        wt = w * t
        from math import cos, sin
        ct, st, cwt, swt = cos(t), sin(t), cos(wt), sin(wt)
        from numpy import array, float64
        v = r * array((ct*cwt - w*st*swt, ct*swt + w*st*cwt, -st), float64)
        return v
    def normal(self, t):
        return self.position(t) / self._radius
    
# -----------------------------------------------------------------------------
#
def segments_on_curve(curve, seg_lengths):
    placements = []
    t = 0
    for length, pad in seg_lengths:
        t1 = curve_segment_end(curve, t, length)
        p = curve_segment_placement(curve, t, t1)
        placements.append(p)
        t = curve_segment_end(curve, t1, pad)
    return placements

# -----------------------------------------------------------------------------
#
def curve_segment_end(curve, t, length, tolerance = 1e-3, max_steps = 100):
    if length == 0:
        return t
    xyz0 = curve.position(t)
    v = curve.velocity(t)
    frac = 0.5
    from chimerax.geometry import norm, inner_product
    t1 = t + frac * length / norm(v)
    for step in range(max_steps):
        xyz1 = curve.position(t1)
        d = norm(xyz1-xyz0)
        if abs(d-length) < tolerance * length:
            return t1
        v1 = curve.velocity(t1)
        # Want |xyz1 + v1*dt - xyz0| = length
        delta = xyz1 - xyz0
        a,b,c = (inner_product(v1,v1),
                 2*inner_product(v1,delta),
                 inner_product(delta,delta) - length*length)
        dt1, dt2 = quadratic_roots(a, b, c)
        if dt1 is None:
            # No point in tangent line is within target length.
            # Go to closest approach
            dt1 = dt2 = -b / 2*a
        dt_min = dt1 if abs(dt1) < abs(dt2) else dt2
        t1 += frac * dt_min
    return None

# -----------------------------------------------------------------------------
#
def quadratic_roots(a, b, c):
    if a == 0:
        if b != 0:
            return -c/b, -c/b
        else:
            return None, None
    d2 = b*b-4*a*c
    if d2 < 0:
        return None, None
    from math import sqrt
    d = sqrt(d2)
    return (-b + d) / (2*a), (-b - d) / (2*a)

# -----------------------------------------------------------------------------
#
def curve_segment_placement(curve, t0, t1):
    from chimerax.geometry import normalize_vector, cross_product, Place
    if t1 > t0:
        xyz0, xyz1 = curve.position(t0), curve.position(t1)
        x_axis = normalize_vector(xyz1 - xyz0)
        center = xyz0
    else:
        x_axis = normalize_vector(curve.velocity(t0))
        center = curve.position(t0)
    y_axis = normalize_vector(curve.normal(0.5*(t0+t1)))  # May not be normal to x_axis
    z_axis = normalize_vector(cross_product(x_axis, y_axis))
    y_axis = normalize_vector(cross_product(z_axis, x_axis))
    p = Place(axes = (x_axis, y_axis, z_axis), origin = center)
    return p
        
# -----------------------------------------------------------------------------
# Base pair file is 3 integer columns start, end, length.  Each line indicates
# a base paired region of specified length with paired sequence positions
# (s,e), (s+1,e-1), ... (s+length-1,e-(length-1)).
#
# This is the format of the supplementary material from the Kevin Weeks
# HIV RNA secondary structure paper.
#
def read_base_pairs(path):

    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    pairs = []
    for line in lines:
        s, e, l = [int(f) for f in line.split()]
        pairs.append((s,e,l))
    return pairs

# -----------------------------------------------------------------------------
# Detect whether base pairing forms a tree of branched stems and loops, or
# whether there are cycles.  The layout algorithm here only handles a tree.
# The Kevin Weeks HIV RNA secondary structure has no cycles.
#
def check_interleaves(pairs):

    for s,e,l in pairs:
        b = i = 0
        for s2,e2,l2 in pairs:
            b1 = (s2 > s and s2 < e)
            b2 = (e2 > s and e2 < e)
            if b1 and b2:
                b += 1
            elif (b1 and not b2) or (not b1 and b2):
                i += 1
        if i > 0:
            print('%5d %5d %3d %3d %3d' % (s,e,l,b,i))
        else:
            print('%5d %5d %3d %3d' % (s,e,l,b))

# -----------------------------------------------------------------------------
# Pairs is a list of triplets of residue positions p1, p2 and length where
# p1 is base paired with p2, p1+1 is base paired with p2-1, ..., and
# p1+len-1 is base paired with p2-(len-1).
#
# The returned pair map maps a base position to its paired position.
# Positions which are not paired do not appear in the map.
#
def pair_map(pairs):

    p = {}
    for s,e,l in pairs:
        if e-s+1 < 2*l:
            from chimerax.core.errors import UserError
            raise UserError('Stem start %d, end %d, length %d is too long' % (s,e,l))
        for i in range(l):
            p[s+i] = e-i
            p[e-i] = s+i
    return p
    
# -----------------------------------------------------------------------------
#
def place_markers(session, coords, radius = 0.5, color = (180,180,180,255), name = 'path',
                  pair_map = {}):

    from chimerax.markers import MarkerSet, create_link
    mset = MarkerSet(session, name)

    # Create markers.
    mmap = {}
    btf = sorted(coords.items())
    for b, tf in btf:
        xyz = tf.origin()
        mmap[b] = m = mset.create_marker(xyz, color, radius, id=b)
        m.extra_attributes = e = {'base_placement':tf}
        if b in pair_map:
            e['paired_with'] = pair_map[b]

    # Link consecutive markers.
    rl = 0.5*radius
    for b, tf in btf:
        if b+1 in mmap:
            create_link(mmap[b], mmap[b+1], color, rl)

    # Link base pairs.
    for b1,b2 in pair_map.items():
        if b1 < b2 and b1 in mmap and b2 in mmap:
            create_link(mmap[b1], mmap[b2], color, rl)
        
    return mset

# -----------------------------------------------------------------------------
# Adjust coordinate frame for each residue so y-axis points from one residue
# to next.  Have old x-axis lie in xy plane of new coordinate frame.
# Origin of frames is not changed.
#
def path_coordinates(coords):

    c = {}
    from chimerax.geometry import orthonormal_frame, Place
    for p, tf in coords.items():
        if p+1 in coords:
            tfn = coords[p+1]
            o = [tf[a][3] for a in (0,1,2)]
            y = [tfn[a][3] - o[a] for a in (0,1,2)]
            x = [tf[a][0] for a in (0,1,2)]
            za,xa,ya = orthonormal_frame(y, x)
            c[p] = Place(matrix = ((xa[0],ya[0],za[0],o[0]),
                                   (xa[1],ya[1],za[1],o[1]),
                                   (xa[2],ya[2],za[2],o[2])))
    return c
    
# -----------------------------------------------------------------------------
#
def place_residues(session, coords, seq, residue_templates_molecule):

    res = dict((r.name, r) for r in residue_templates_molecule.residues)
    if 'U' in res:
        res['T'] = res['U']

    from chimerax.atomic import AtomicStructure
    m = AtomicStructure(session, name = 'RNA')
    n = len(seq)
    rlist = []
    for p, tf in sorted(coords.items()):
        if p <= n:
            t = seq[p-1]
            if t in res:
                rt = res[t]
                r = copy_residue(rt, p, tf, m)
                rlist.append(r)

    # Join consecutive residues
    rprev = rlist[0]
    for r in rlist[1:]:
        if r.number == 1 + rprev.number:
            a2 = r.find_atom('P')
            a1 = rprev.find_atom("O3'")
            if a1 and a2:
                m.new_bond(a1,a2)
        rprev = r
                
    session.models.add([m])
    return m

# -----------------------------------------------------------------------------
#
def copy_residue(r, p, tf, m):

    cr = m.new_residue(r.name, r.chain_id, p)
    amap = {}
    for a in r.atoms:
        amap[a] = ca = m.new_atom(a.name, a.element)
        ca.coord = tf * a.coord
        cr.add_atom(ca)

    bonds = set(sum((a.bonds for a in r.atoms), []))
    for b in bonds:
        a1, a2 = b.atoms
        if a1 in amap and a2 in amap:
            m.new_bond(amap[a1], amap[a2])
    return cr

# -----------------------------------------------------------------------------
#
def minimize_rna_backbone(mol,
                          chunk_size = 10,
                          gradient_steps = 100,
                          conjugate_gradient_steps = 100,
                          update_interval = 10,
                          nogui = True):

    base_atoms = ('N1,C2,N2,O2,N3,C4,N4,C5,C6,N6,O6,N7,C8,N9,'
                  'H1,H21,H22,H2,H41,H42,H5,H6,H61,H62,H8')
    ng = 'true' if nogui else 'false'
    opt = ('cache false nogui %s nsteps %d cgsteps %d interval %d'
           % (ng, gradient_steps, conjugate_gradient_steps, update_interval))
    prep = True

    # Find ranges of residues for each chain id.
    cr = {}
    for r in mol.residues:
        cid = r.chain_id
        p = r.number
        pmin, pmax = cr.get(cid, (p,p))
        cr[cid] = (min(p,pmin), max(p,pmax))

    # Minimize backbone atoms in blocks of chunk_size residues.
    mid = mol.oslIdent()
    from chimera import runCommand
    for cid, (pmin, pmax) in cr.items():
        c = '' if cid.isspace() else '.' + cid
        for i in range(pmin, pmax, chunk_size):
            i1 = i-1 if i > pmin else i
            i2 = i + chunk_size
            if i2+1 == pmax:
                i2 = pmax       # Always minimize at least 2 nucleotides.
            prep = '' if prep is True else 'prep false'
            cmd = 'minimize spec %s:%d-%d%s fragment true freeze %s:%d%s:%d-%d%s@%s %s %s' % (mid, i1, i2, c, mid, i1, c, i, i2, c, base_atoms, prep, opt)
            print(cmd)
            runCommand(cmd)
            
# -----------------------------------------------------------------------------
# Rotate nucleotides in loops about x-axis by random amount.
#
def random_loop_orientations(coords, pmap, angle = 90, center = (0,0,0)):

    from random import random
    from chimerax.geometry import rotation
    for b, tf in tuple(coords.items()):
        if not b in pmap:
            a = (2*random()-1)*angle
            rx = rotation((1,0,0), a, center)
            coords[b] = tf * rx
            
# -----------------------------------------------------------------------------
#
def place_coordinate_frames(coords, radius):

    r = radius
    t = .2*r
    from VolumePath import Marker_Set, Marker, Link
    mset = Marker_Set('nucleotide orientations')
    for b, tf in coords.items():
        p0 = (tf[0][3],tf[1][3],tf[2][3])
        m0 = Marker(mset, 4*b, p0, (.5,.5,.5,1), t)
        p1 = (tf[0][3]+r*tf[0][0],tf[1][3]+r*tf[1][0],tf[2][3]+r*tf[2][0])
        m1 = Marker(mset, 4*b+1, p1, (1,.5,.5,1), t)
        Link(m0, m1, (1,.5,.5,1), t)
        p2 = (tf[0][3]+r*tf[0][1],tf[1][3]+r*tf[1][1],tf[2][3]+r*tf[2][1])
        m2 = Marker(mset, 4*b+2, p2, (.5,1,.5,1), t)
        Link(m0, m2, (.5,1,.5,1), t)
        p3 = (tf[0][3]+r*tf[0][2],tf[1][3]+r*tf[1][2],tf[2][3]+r*tf[2][2])
        m3 = Marker(mset, 4*b+3, p3, (.5,.5,1,1), t)
        Link(m0, m3, (.5,.5,1,1), t)
    return mset

# -----------------------------------------------------------------------------
#
def color_rna_path(markers, seq):

    colors = {'A': (1,.5,.5,1),
              'C': (1,1,.5,1),
              'G': (.5,1,.5,1),
              'T': (.5,.5,1,1),
              'U': (.5,.5,1,1),
              }
    n = len(seq)
    for m in markers:
        i = m.residue.number - 1
        if i < n:
            color = colors.get(seq[i])
            if color:
                m.set_rgba(color)

# -----------------------------------------------------------------------------
#
def read_fasta(fasta_path):

    f = open(fasta_path, 'r')
    f.readline()        # header
    seq = f.read()
    f.close()
    seq = seq.replace('\n', '')
    return seq

# -----------------------------------------------------------------------------
#
def color_path_regions(markers, reg_path, seq_start):

    f = open(reg_path)
    lines = f.readlines()
    f.close()

    c3 = [line.split('\t')[:3] for line in lines]
    from chimera.colorTable import getColorByName
    regions = [(int(i1), int(i2), getColorByName(cname).rgba())
               for i1,i2,cname in c3]

    color = {}
    for i1,i2,rgba in regions:
        for i in range(i1,i2+1):
            color[i] = rgba

    for m in markers:
        i = seq_start + m.residue.number - 1
        if i in color:
            m.set_rgba(color[i])
    
# -----------------------------------------------------------------------------
#
def rna_path(session, sequence_length, pair_map, pattern = 'line', marker_radius = 2,
             random_branch_tilt = 0, name = 'rna layout'):

    # Compute layout.
    c = circuit(pair_map, 1, sequence_length)
    params = Layout_Parameters(
        branch_tilt = random_branch_tilt    # Random branch tilt range
        )
    coords = c.layout(params, pattern = pattern)
    #random_loop_orientations(coords, pair_map, 90, center = (0,8,0))

    # Place a marker at each nucleotide position.
    if name is None:
        return coords

    mset = place_markers(session, coords, radius = marker_radius, name = name,
                         pair_map = pair_map)

    # Place coordinate frames for debugging.
    #place_coordinate_frames(coords, marker_radius)

    return mset, coords

# -----------------------------------------------------------------------------
# Color loops and stems.
#
def color_path(markers, pair_map, loop_color, stem_color):

    for m in markers:
        m_id = m.residue.number
        m.color = (stem_color if m_id in pair_map else loop_color)

# -----------------------------------------------------------------------------
#
def rna_atomic_model(session, sequence, base_placements, name = 'RNA',
                     nucleotide_templates = 'rna-templates.pdb'):

    # Place residues at each position to build an atomic model.
    templates = template_molecule(session, nucleotide_templates)
    mol = place_residues(session, base_placements, sequence, templates)
    mol.name = name
    return mol

# -----------------------------------------------------------------------------
#
template_mol = {}
def template_molecule(session, nucleotide_templates):

    from os.path import join, dirname
    path = join(dirname(__file__), nucleotide_templates)

    global template_mol
    if path in template_mol:
        return template_mol[path]
    stream = open(path, 'r')
    
    name = nucleotide_templates
    from chimerax.atomic.pdb import open_pdb
    mols, msg = open_pdb(session, stream, name)
    stream.close()
    tmol = mols[0]
    template_mol[path] = tmol

    return tmol
    
# -----------------------------------------------------------------------------
#
def color_stems_and_loops(mol, pair_map, loop_color, stem_color):

    for a in mol.atoms:
        a.color = (stem_color if a.residue.number in pair_map else loop_color)
    for r in mol.residues:
        r.ribbon_color = (stem_color if r.number in pair_map else loop_color)
    
# -----------------------------------------------------------------------------
#
def make_hiv_rna():

    sequence_length = 9254
    
    # Read secondary structure.
    hivdir = '/usr/local/src/staff/goddard/presentations/hiv-rna-may2011'
    from os.path import join, dirname
    pairs_file = join(hivdir, 'pairings.txt')
    pairs = read_base_pairs(path)
    #check_interleaves(pairs)
    pmap = pair_map(pairs)
    print('%d stems involving %d nucleotides' % (len(pmap), len(pairs)))

    sequence_file = join(hivdir, 'hiv-pNL4-3.fasta')
    sequence_start = 455
    
    mset, coords = rna_path(sequence_length, pmap,
                            pattern = 'line',
#                            random_branch_tilt = 45,
                            )

    # Set placement matrices for use with sym command each 10 nucleotides.
    sym_spacing = 10
    mm = mset.marker_molecule()
    tfplace = [tf for b, tf in sorted(coords.items())[::sym_spacing]]
    mm.placements = lambda name,tfplace=tfplace: tfplace

    # Color nucleotides by type (A = red, C = yellow, G = green, T = U = blue).
    color_rna_path(mset.markers(), sequence)
    color_path(mset.markers(), pmap,
               loop_color = (.5,.5,.5,1), stem_color = (1,1,0,1))

    # Color segments listed in a text file.
    #regions_file = join(hivdir, 'regions.txt')
    #color_path_regions(mset.markers(), regions_file, sequence_start)

    # Read nucleotide sequence.
    seq = read_fasta(sequence_file)[sequence_start-1:]

    mol = rna_atomic_model(seq, coords)
    minimize_rna_backbone(mol)
