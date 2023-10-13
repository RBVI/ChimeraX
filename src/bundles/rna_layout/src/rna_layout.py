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
    # circle with specified gap width.  The gap is positioned on the x-axis
    # with left end of the gap at the origin.
    #
    # The returned coordinates are a dictionary mapping residue number to a
    # transform (3 by 4 matrix) which takes (0,0,0) to the nucleotide P atom
    # maps the x-axis in the direction of the next nucleotide P atom,
    # and has the position of a basepaired nucleotide P atom in the mapped
    # xy plane (y-axis maps to approximately the direction of the base).
    #
    def layout(self, params, pattern = 'circle', gap = None):

        # Compute stem and loop segment layouts.
        sl = self.stems_and_loops
        n = len(sl)
        segs = []
        for e in sl:
            segs.extend(e.layout(params, n))

        if pattern == 'circle':
            # Layout on circle with a gap at the bottom, progressing
            # clockwise from bottom left gap end point which is at the origin.
            if gap is None:
                gap = params.pair_width
            coords = self.circle_layout(segs, params, gap)
        elif pattern == 'line':
            # Layout on x-axis
            coords = self.straight_layout(segs, params)
        elif pattern == 'helix':
            # Helix along z axis
            curve = HelixCurve(params.helix_radius, params.helix_rise)
            coords = self.curve_layout(segs, params, curve)
        elif pattern == 'sphere':
            # Spiral path on sphere from pole to pole.
            if params.sphere_radius is None or params.sphere_turns is None:
                length = sum([(seg.width + seg.pad) for seg in segs], 0)
                params.set_sphere_defaults(length)
                print ('RNA sphere radius %.1f' % params.sphere_radius)
            curve = SphereSpiral(params.sphere_radius, params.sphere_turns)
            coords = self.curve_layout(segs, params, curve)
            
        return coords

    def circle_layout(self, segs, params, gap):
        '''
        Layout on circle with a gap at the bottom, progressing
        clockwise from bottom left gap end point which is at the origin.
        This is used both for laying out a top level circle pattern
        and also for laying out the loops and stems attached in a
        circle to the end of a stem.
        '''

        # Compute circle radius that exactly fits the segments
        # with segment end points lying on the circle.
        seg_lengths = [seg.width for seg in segs] + [seg.pad for seg in segs]  + [gap]
        wc = value_counts(seg_lengths)
        radius = polygon_radius(list(wc.items()))

        # Translate from origin at bottom of circle to origin
        # at lower left gap end point.
        gap_angle_step = circle_angle(gap, radius)
        from math import pi, sin, cos
        a = 0.5*gap_angle_step*pi/180
        from chimerax.geometry import translation, rotation
        stf = translation((radius*sin(a), -radius+radius*cos(a), 0))

        # Layout segments on the circle
        coords = {}
        angle = 0.5 * gap_angle_step
        for seg in segs:
            angle_step = circle_angle(seg.width, radius)
            rtf = rotation((0,0,1), 180 - 0.5*angle_step)
            if params.branch_tilt != 0:
                from random import random
                btf = rotation((1,0,0), params.branch_tilt * (1-2*random()))
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
        '''Layout on x-axis starting at the origin, proceeding in positive x direction.'''

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
        '''
        Layout along a curve which is a class instance such as HelixCurve
        or SphereSpiral defining position(t), velocity(t), normal(t) methods.
        Layout starts at t = 0 and goes toward increasing t values until
        all segments are layed out.
        '''
        
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
    '''
    Double helical RNA segment.
    Lays out two strands with nucleotides oriented for base pairing.
    '''
    
    def __init__(self, base5p, base3p, length, circuit):
        self.base5p = base5p
        self.base3p = base3p
        self.length = length
        self.circuit = circuit  # Circuit at far end of stem

    def layout(self, params, circuit_segments):

        p = params
        from chimerax.geometry import translation, rotation, Place, vector_rotation, norm
        from math import pi, cos, sin

        # Compute helix axis direction.
        # The helix rotation axis is does not make a 90 degree angle with
        # the P-P basepair line.  It makes an angle of approximately 110 degrees.
        pa = p.pair_tilt * pi / 180
        from numpy import array
        axis = array((cos(pa), sin(pa), 0))

        # Compute screw motion that advances basepair along helix.
        # Initial basepair P-P is spans (0,0,0) to (pair_width,0,0).
        rtf = rotation(axis, p.stem_twist,
                       center = (0.5*p.pair_width,0,-p.pair_off_axis))
        ttf = translation(p.pair_spacing * axis)
        stf = ttf * rtf

        # Specify initial orientations of basepair nucleotides
        # so P-atoms are on helix and paired P atom is in the
        # oriented xy plane.
        p0, p1 = (0,0,0), (p.pair_width,0,0)
        tf1 = self.stem_residue_orientation(p0, stf*p0, p1)
        p2 = stf.inverse()*p1
        tf2 = self.stem_residue_orientation(p1, p2, p0)

        # Keep track of basepair P-P orientation used to
        # orient the loop at the end of the stem.
        ctf = Place()

        # Layout nucleotides for both strands of double helix.
        coords = {}
        for i in range(self.length):
            coords[self.base5p+i] = tf1
            coords[self.base3p-i] = tf2
            tf1 = stf * tf1
            tf2 = stf * tf2
            ctf = stf * ctf

        # The next segment after the stem should put its first
        # P-atom at a position on the helix so that the last
        # nucleotide of the stem has the right orientation to
        # make a base pair.  To achieve this the initial basepair
        # P-P does not lie on the x-axis.  Instead we tilt the
        # entire helix so the paired P is advanced up the helix
        # by one position.
        width = norm(p2)
        ttf = vector_rotation(p2, p1)
        for b, tf in coords.items():
            coords[b] = ttf * coords[b]

        # Added circuit at end of stem.
        ccoords = self.circuit.layout(p, gap = width)

        # Position the end circuit using the P-P orientation at
        # the end of the stem and taking account of the tilt of
        # the entire helix.
        ctf = ttf * ctf * ttf.inverse()
        for b, tf in ccoords.items():
            coords[b] = ctf * tf

        # Use segment width for the tilted helix.
        seg = [Segment(coords, width, 0)]
        
        return seg

    def stem_residue_orientation(self, p, next_p, pair_p):
        from chimerax.geometry import cross_product, orthonormal_frame
        from numpy import array
        x, y = next_p - p, array(pair_p) - p
        z = cross_product(x, y)
        tf = orthonormal_frame(z, xdir = x, origin = p)
        return tf
    
# Single strand RNA segment.
class Loop:
    '''
    Layout a single stranded series of nucleotides in a pattern
    more compact than a straight line, such as a helix or a series
    of horseshoes.
    '''
    def __init__(self, base5p, length):
        self.base5p = base5p	# Index of first base.
        self.length = length	# Number of bases

    def layout(self, params, circuit_segments):
        pattern = params.loop_pattern
        if pattern == 'helix':
            segs = self.helix_segments(params, circuit_segments)
        elif pattern == 'horseshoe':
            segs = self.horseshoe_segments(params, circuit_segments)

        self.rotate_nucleotides(segs, params.loop_twist)

        return segs

    def helix_segments(self, params, circuit_segments):
        '''
        Layout nucleotides along helical segments with a specified
        spacing, helix radius and rise per turn.  Multiple single
        turn helix segments are created so that the helix segments
        can be layed out to follow a curve such as a circle.
        '''
        n = self.length
        b = self.base5p
        spacing = params.loop_spacing

        # One turn helix length is minimum.  If shorter lay out in straight line.
        helix_length = params.helix_loop_size
        if n < helix_length:
            # Make each nucleotide a segment.  Not enough nucleotides to make a helix.
            segs = singleton_segments(b, n, spacing)
        else:
            segs = []

            # Distribute extra nucleotides that won't go into helix
            # equally at start and end.
            nh = n // helix_length
            ne = n % helix_length
            base_index = b
            start_pad = ne//2
            if start_pad > 0:
                segs.extend(singleton_segments(base_index, start_pad, spacing))
                base_index += start_pad
            nleft = n - start_pad

            # Make helix turns of 3, 2, or 1 times the desired length.
            while nleft >= helix_length:
                hlen = 3*helix_length
                if hlen > nleft: hlen = 2*helix_length
                if hlen > nleft: hlen = helix_length
                seg = self.helix_segment(base_index, hlen, spacing, params.helix_loop_rise)
                base_index += hlen
                segs.append(seg)
                nleft -= hlen

            # Added final nucleotides in a straight segment.
            if nleft > 0:
                segs.extend(singleton_segments(base_index, nleft, spacing))

        return segs

    def helix_segment(self, base_index, count, spacing, rise):
        '''
        Lay out a single turn helix on x axis.
        Radius is calculated to fit the specified number of nucleotides.
        '''
        # Choose radius so one helix turn has desired length.
        # Helix arc length = s, radius = r, rise = h: s**2 = r**2 + (h/(2*pi))**2
        length = spacing * count
        from math import sqrt, pi
        radius = sqrt((length/(2*pi))**2 - (rise/(2*pi))**2)

        # Compute screw motion to advance along helix
        angle = 2*pi / count
        angle_deg = angle*180/pi
        from chimerax.geometry import translation, rotation, vector_rotation, Place
        step = translation((rise/count, 0, 0)) * rotation((1,0,0), angle_deg, center=(0,radius,0))

        # Compute first nucleotide so P-P lies on helix.
        orient = vector_rotation((1,0,0), step*(0,0,0))

        # Place nucleotides on helix.
        place = {}
        p = Place()
        for i in range(count):
            place[base_index + i] = p * orient
            p = step * p
            
        return Segment(place, rise, pad = 0)
    
    def horseshoe_segments(self, params, circuit_segments):
        '''
        Layout nucleotides in a sequence of horseshoe shapes.
        A horseshoe has two straight sides parallel y-axis and
        a half-circle cap at the top.  Horseshoes are connected
        side by side along the x-axis.
        '''
        n = self.length
        b = self.base5p
        spacing = params.loop_spacing
        
        curve_length = params.horseshoe_curve_size  # Number of nucleotides in curved part of horseshoe
        if n < curve_length:
            # Make each nucleotide a segment.  Not enough nucleotides to make a horseshoe.
            segs = singleton_segments(b, n, spacing)
        else:
            # Create as many horseshoe segments as needed.
            mxs = params.horseshoe_curve_size + 2*params.horseshoe_side_size
            if circuit_segments == 1:
                lsp = params.horseshoe_spacing_for_stem_loop()
            else:
                lsp = params.horseshoe_spacing
            # Start with lsp single nucleotide segments for spacing
            segs = singleton_segments(b, lsp, spacing)
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
                segs.extend(singleton_segments(bl, lsp, spacing))
                bl += lsp
                nl -= lsp
            # Any leftover nucleotides with too few to make a horseshoe, make each its own segment.
            segs.extend(singleton_segments(bl, nl, spacing))
        return segs

    def horseshoe_segment(self, side_length, semicircle_length, spacing, start_index):
        '''
        Make a horseshoe segment with two sides each with side_length nucleotides
        parallel the y axis, curved top has semicircle_length nucleotides, spacing of
        nucleotides is spacing.  Starting nucleotide index is start_index.
        '''
        from chimerax.geometry import translation, rotation
        c = {}
        # Horseshoe side going up in y.
        ns = side_length
        b = start_index
        ls = spacing
        for i in range(ns):
            c[b+i] = translation((0,i*ls,0)) * rotation((0,0,1), 90)
        from math import pi, cos, sin
        nc = semicircle_length
        r = 0.5*ls/sin(0.5*pi/(nc-1))
        for i in range(nc):
            a = i*pi/(nc-1)
            a_deg = a*180/pi
            ca, sa = cos(a), sin(a)
            # Horeshoe curve
            c[b+ns+i] = translation((r*(1-ca),r*sa+ns*ls,0)) * rotation((0,0,1),90-a_deg)
        # Horseshoe side going down in y.
        for i in range(ns):
            c[b+ns+nc+i] = translation((2*r,(ns-1-i)*ls,0)) * rotation((0,0,1), -90)

        return Segment(c, 2*r, spacing)

    def rotate_nucleotides(self, segs, max_rotation):
        '''Rotate each nucleotide a random angle around P-P backbone.'''
        if max_rotation == 0:
            return

        from chimerax.geometry import rotation
        from random import random
        for seg in segs:
            for b, place in seg.placements.items():
                a = (2*random()-1) * max_rotation
                seg.placements[b] = place * rotation((1,0,0), a)

class Segment:
    '''
    Nucleotide placements for a sequence of one or more nucleotides
    such as a double helical stem or a portion of a loop.
    Width is the distance (Angstroms) between first and last nucleotide P atoms.
    Pad is space from last nucleotide P to the first P of the next segment.
    '''
    def __init__(self, placements, width, pad):
        self.width = width
        self.pad = pad
        self.placements = placements  # Map nucleotide index to Place
    @property
    def count(self):
        '''Number of nucleotides placed.'''
        return len(self.placements)
    
def singleton_segments(base_index, count, spacing, placement = None):
    '''Create one nucleotide Segments with the same orientation and width.'''
    if placement is None:
        from chimerax.geometry import Place
        placement = Place()
    return [Segment({base_index + i: placement}, width = spacing, pad = 0) for i in range(count)]
        
class LayoutParameters:
    '''Parameters controlling the nucleotide backbone P atom path.'''
    
    def __init__(self, loop_pattern = 'helix', loop_spacing = 6.5, loop_twist = 0,
                 helix_loop_size = 8, helix_loop_rise = 20,
                 horseshoe_curve_size = 8, horseshoe_side_size = 10, horseshoe_spacing = 1,
                 pair_spacing = 2.55, pair_width = 18.3, pair_off_axis = 3.4, pair_tilt = 110.7,
                 stem_twist = 31.5, branch_twist = 0, branch_tilt = 0,
                 helix_radius = 300, helix_rise = 50,
                 sphere_radius = None, sphere_turns = None, sphere_turn_spacing = 60):

        # Loop layout parameters
        self.loop_pattern = loop_pattern	# Loop layout "helix" or "horseshoe".
        self.loop_spacing = loop_spacing	# Spacing of nucleotides in horseshoe or helix, Angstroms.
        self.loop_twist = loop_twist		# Random nucleotide rotation magnitude about P-P, degrees

        # Loop layout parameters for helical loops
        self.helix_loop_size = helix_loop_size	# Number of nucleotides in a loop layed out as a helix
        self.helix_loop_rise = helix_loop_rise	# Rise in loop helix over one turn, Angstroms

        # Loop layout parameters for horseshoe loops
        self.horseshoe_curve_size = horseshoe_curve_size # Number of nucleotides in horseshoe curved part
        self.horseshoe_side_size = horseshoe_side_size	# Max nucleotides in one horseshoe side
        self.horseshoe_spacing = horseshoe_spacing	# Number of nucleotides to place between 2 horseshoes

        # Segment orientation parameters
        self.branch_twist = branch_twist	# Twist per loop or stem in straight pattern (degrees).
        self.branch_tilt = branch_tilt		# Random tilt magnitude (degrees) for circle pattern

        # Double helix stem parameters
        self.pair_spacing = pair_spacing	# Spacing of one base pair to next base pair, Angstroms.
        self.pair_width = pair_width		# P-P basepair spacing between double helix strands.
        self.pair_off_axis = pair_off_axis	# P-P center point distance from double helix axis.
        self.pair_tilt = pair_tilt		# Angle helix axis makes with basepair P-P line.
        self.stem_twist = stem_twist		# Twist per base-pair in a stem (degrees).

        # Global path layout parameters
        self.helix_radius = helix_radius	# Radius for helix layout, Angstroms
        self.helix_rise = helix_rise		# Rise per turn for helix layout, Angstroms

        self.sphere_radius = sphere_radius	# Spiral sphere layout radius, Angstroms
        self.sphere_turns = sphere_turns	# Spiral sphere turns from top to bottom, count.
        self.sphere_turn_spacing = sphere_turn_spacing	# Spacing between turns, Angstroms

    def set_sphere_defaults(self, path_length):
        '''
        Choose sphere radius and number of turns to accomodate path length.
        Spiral length l = 4*r*t, spacing s = pi*r/t, where r = radius, t = num turns.
        '''
        spacing = self.sphere_turn_spacing
        turns = self.sphere_turns
        radius = self.sphere_radius
        from math import pi, sqrt
        if radius is None and turns is None:
            turns = max(1, sqrt((pi/4)*path_length/spacing))
            radius = turns * spacing / pi
        elif radius is None:
            radius = turns * spacing / pi
        elif turns is None:
            turns = max(1, pi*radius/spacing)
        self.sphere_radius = radius
        self.sphere_turns = turns

    def horseshoe_spacing_for_stem_loop(self):
        '''
        A stem with a large loop can make a horseshoe for the loop where
        the ends of the horseshoe are much further apart than the ends of the stem.
        That creates a long bond between the stem and the loop.  To avoid that
        put some of the loop into spacers (single nucleotide segments) by increasing
        the horseshoe spacing (number of nucleotides between two horseshoes) to the
        value computed by this routine.

        Horseshoe spacing is set so that the spacers on each side of the horseshoe
        when attached to the stem span a width equal to the width of the horseshoe.
        '''
        from math import pi, ceil
        gap = pi*self.horseshoe_radius() - (self.pair_width + 2*self.loop_spacing)
        els = max(0, int(ceil(0.5*gap / self.loop_spacing)))
        return els

    def horseshoe_radius(self):
        from math import pi, sin
        r = 0.5*self.loop_spacing/sin(0.5*pi/(self.horseshoe_curve_size-1))
        return r

# -----------------------------------------------------------------------------
#
def circuit(pair_map, start, end):
    '''
    Create a Circuit object and Stem and Loop objects representing
    RNA topology from a base pair.  The base pair map maps a nucleotide
    index to a paired nucleotide index.  The entire sequence ranges from
    nucleotide index start to end.  Unpaired nucleotides are in loops.
    '''

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
def polygon_radius(edge_lengths):
    '''
    Determine the radius of a circle that exactly fits several
    straight segments placed with ends on the circle end-to-end.
    The edge_lengths are pairs of a length and number of times
    that length should be repeated.
    '''
    from math import pi, asin
    l = sum(n*l for l,n in edge_lengths)
    r0 = l/(2*pi)
    r1 = l/2
    while r1-r0 > 1e-5 * l:
        rm = 0.5*(r0+r1)
        if sum(n*asin(min(1.0,l/(2*rm))) for l,n in edge_lengths) > pi:
            r0 = rm
        else:
            r1 = rm
    return rm

# -----------------------------------------------------------------------------
#
def circle_angle(side_length, radius):
    '''
    Angle spanned by edge with given length where ends are
    on a circle of specified radius.
    '''
    from math import asin, pi
    return 2*asin(min(1.0,0.5*side_length/radius)) * (180/pi)

# -----------------------------------------------------------------------------
#
def value_counts(values):
    '''
    Return a dictionary mapping value to a count of how
    many times that value occurs in sequence values.
    '''
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
    '''
    Define a helical curve with axis on z axis and given radius and rise per turn.
    '''
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
    '''
    Define a spiral on a sphere from pole to pole, poles on z-axis,
    sphere radius specified, number of turns of spiral given.
    '''
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
    '''
    Return a list of placements (Place instances) that map segments of
    specified lengths one after another so their end-points are on the
    specified curve. A Place maps the segment from (0,0,0) to (length,0,0)
    to its position with end-points on the curve.
    '''
    placements = []
    t = 0
    for length, pad in seg_lengths:
        t1 = curve_segment_end(curve, t, length)
        if t1 is None:
            from chimerax.core.errors import UserError
            raise UserError('rna: Layout curve turns too quickly to place segment'
                            ' of length %.4g at curve parameter %.4g' % (length, t))
        p = curve_segment_placement(curve, t, t1)
        placements.append(p)
        t = curve_segment_end(curve, t1, pad)
    return placements

# -----------------------------------------------------------------------------
#
def curve_segment_end(curve, t, length, tolerance = 1e-3, max_steps = 100):
    '''
    Compute a point on the curve beyond curve parameter value t which is
    a distance length from the point at position t.  Can return None if
    convergence fails.  This iteratively takes linear steps based on the
    curve velocity vector to find the desired end point.
    '''
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
    '''
    Return real roots of quadratic equation a*x*x + b*x + c = 0.
    If roots are complex returns None, None.
    '''
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
    '''
    Create a Place instance mapping segment (0,0,0), (0,0,length) to
    curve points at parameter value t0 and t1.
    '''
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
#
def read_base_pairs(path):
    ''''
    Base pair file has 3 columns start, end, length, each integers.
    Each line indicates a base paired region of specified length
    with paired sequence positions (s,e), (s+1,e-1), ... (s+length-1,e-(length-1)).

    This is the format of the supplementary material from the Kevin Weeks
    HIV RNA secondary structure paper.
    '''
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    pairs = []
    for line in lines:
        s, e, l = [int(f) for f in line.split()]
        pairs.append((s,e,l))
    return pairs

# -----------------------------------------------------------------------------
#
def check_interleaves(pairs):
    '''
    Detect whether base pairing forms a tree of branched stems and loops, or
    whether there are cycles.  The layout algorithm here only handles a tree.
    The Kevin Weeks HIV RNA secondary structure has no cycles.
    Pseudoknot RNA structures do not form a tree.
    '''
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
#
def pair_map(pairs):
    '''
    Pairs is a list of triplets of residue positions p1, p2 and length where
    p1 is base paired with p2, p1+1 is base paired with p2-1, ..., and
    p1+len-1 is base paired with p2-(len-1).

    The returned pair map maps a base position to its paired position.
    Positions which are not paired do not appear in the map.
    '''

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
def place_markers(session, coords, radius = 0.5, color = (180,180,180,255),
                  name = 'path', pair_map = {}):
    '''
    Create a MarkerSet with markers at positions specified by coords which maps
    an integer index to a Place instance.  The origin of the Place i.e. where
    it maps (0,0,0) is the location of the marker.  Markers with consecutive
    integer indices are connected with links, and pairs of markers specified
    in pair_map (mapping index to index) are also connected by links.  Markers
    have residue number equal to their index.  The MarkerSet is returned.
    '''
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
#
def place_residues(session, coords, sequence, residue_templates,
                   status_interval = 1000):
    '''
    Use placements specified by coords (map of index to Place instance) to
    position nucleotides with given sequence.  The nucleotide single letter
    code (A,C,G,U) maps to a Residue given by map residue_templates.
    Index i corresponds to sequence position i-1.
    '''
    from chimerax.atomic import AtomicStructure
    m = AtomicStructure(session, name = 'RNA', auto_style = False)
    n = len(sequence)
    rlist = []
    for p, tf in sorted(coords.items()):
        if p <= n:
            t = sequence[p-1].upper()
            if t in residue_templates:
                rt = residue_templates[t]
                r = copy_residue(rt, p, tf, m)
                rlist.append(r)
                if status_interval and len(rlist) % status_interval == 0:
                    session.logger.status('Created %d of %d residues'
                                          % (len(rlist), len(coords)))

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
    '''
    Create a copy of Residue r in molecule m with coordinates transformed
    by Place tf, and residue number p.
    '''
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
    '''
    TODO: This Chimera code has not been ported to ChimeraX.
    '''
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
#
def random_loop_orientations(coords, pmap, angle = 90, center = (0,0,0)):
    '''
    Rotate nucleotides in loops about x-axis by random amount.
    '''
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
    '''
    TODO: This Chimera code has not been ported to ChimeraX.
    '''
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
    '''
    TODO: This Chimera code has not been ported to ChimeraX.
    '''
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
    '''
    TODO: This Chimera code has not been ported to ChimeraX.
    '''
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
             layout_parameters = None, name = 'rna layout'):

    # Compute layout.
    c = circuit(pair_map, 1, sequence_length)

    if layout_parameters is None:
        layout_parameters = LayoutParameters()

    coords = c.layout(layout_parameters, pattern = pattern)

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
def rna_atomic_model(session, sequence, base_placements, name = 'RNA'):

    # Place residues at each position to build an atomic model.
    res_templates = rna_nucleotide_templates(session)
    mol = place_residues(session, base_placements, sequence, res_templates)
    mol.name = name
    return mol

# -----------------------------------------------------------------------------
#
def rna_nucleotide_templates(session, file = 'rna_templates_6pj6.cif', chain_id = 'I',
                             residue_numbers = {'A': 2097, 'C': 2096, 'G': 2193, 'U': 2192}):
    '''
    Return residues A,G,C,U with P at (0,0,0) and next residue P at (x,0,0) with x>0
    and rotated so that a basepaired residue P is in the xy-plane with y > 0.
    These are used for building an atomic model from a P-atom backbone trace.
    '''

    # Read template residues from mmCIF file
    from os.path import join, dirname
    path = join(dirname(__file__), file)
    from chimerax.mmcif import open_mmcif
    mols, msg = open_mmcif(session, path)
    m = mols[0]

    # Calculate transforms to standard coordinates.
    res_tf = []
    pair_name = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
    from chimerax.geometry import cross_product, orthonormal_frame
    for resname in ('A','C','G','U'):
        r = m.find_residue(chain_id, residue_numbers[resname])
        rnext = m.find_residue(chain_id, residue_numbers[resname]+1)
        rpair = m.find_residue(chain_id, residue_numbers[pair_name[resname]])
        a0,a1,a2 = r.find_atom('P'), rnext.find_atom('P'), rpair.find_atom('P')
        o,x,y = a0.coord, a1.coord, a2.coord
        z = cross_product(x-o, y-o)
        tf = orthonormal_frame(z, xdir = x-o, origin = o).inverse()
        res_tf.append((resname, r, tf))

    # Transform template residues to standard coordinate frame.
    res = {}
    for name, r, tf in res_tf:
        ratoms = r.atoms
        ratoms.coords = tf * ratoms.coords
        res[name] = r

    res['T'] = res['U']

    return res
# -----------------------------------------------------------------------------
#
def color_stems_and_loops(mol, pair_map, loop_color, stem_color, p_color = None):

    for a in mol.atoms:
        if p_color and a.name == 'P':
            a.color = p_color
        elif a.residue.number in pair_map:
            a.color = stem_color
        else:
            a.color = loop_color
    for r in mol.residues:
        r.ribbon_color = (stem_color if r.number in pair_map else loop_color)
    
# -----------------------------------------------------------------------------
#
def make_hiv_rna():
    '''
    TODO: This Chimera code has not been ported to ChimeraX.
    '''
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
