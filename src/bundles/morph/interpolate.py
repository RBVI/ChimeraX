from time import time
smt = stt = rit = rst = rsit = 0
from time import time
def interpolate(coordset0, coordset1, residue_groups, residue_interpolators,
                method, rate_method, frames, log):
        # coordset0     initial coordinates indexed by atom index
        # coordset1     final coordinates indexed by atom index
        # residue_groups      list of collections of residues defining groups moved semi-rigidly
        # residue_interpolators   maps residue to an interpolator instance
        # method        interpolation method name
        # rate_method   spacing method for interpolation steps: "linear", "sinusoidal", "ramp up", "ramp down"
        # frames        number of frames to generate in trajectory
        # log           Logger for reporting progress messages

        # Get transform for each rigid segment
        t0 = time()
        seg_info = []
        from .util import segment_alignment_atoms
        from chimerax.core.geometry import align_points
        from chimerax.core.atomic import Atoms
        for rList in residue_groups:
                raindices = Atoms(segment_alignment_atoms(rList)).coord_indices
                cList0 = coordset0[raindices]
                c0 = cList0.mean(axis = 0)
                cList1 = coordset1[raindices]
                c1 = cList1.mean(axis = 0)
                xform, rmsd = align_points(cList0, cList1)
                seg_info.append((rList, rList.atoms.coord_indices, c0, c1, xform))
        t1 = time()
        global stt
        stt += t1-t0

        # Calculate interpolated coordinates for each frame.
        coordsets = []
        segment_motion = SegmentMotionMethods[method]
        rateFunction = RateMap[rate_method]
        rate = rateFunction(frames)  # Compute fractional steps controlling speed of motion.
        nc = len(coordset0)
        c0s = coordset0.copy()
        c1s = coordset1.copy()
        for i, f in enumerate(rate):
                coordset = coordset0.copy()
                for rList, atom_indices, c0, c1, xform in seg_info:
                        t0 = time()
                        xf0, xf1 = segment_motion(xform, c0, c1, f)
                        t1 = time()
                        global rit
                        rit += t1-t0
                        t0 = time()
                        # Apply rigid segment motion
                        c0s[atom_indices] = xf0 * coordset0[atom_indices]
                        c1s[atom_indices] = xf1 * coordset1[atom_indices]
                        t1 = time()
                        global smt
                        smt += t1 - t0
                        t0 = time()
                        for r in rList:
                                residue_interpolators[r](c0s, c1s, f, coordset)
                        t1 = time()
                        global rst
                        rst += t1-t0

                coordsets.append(coordset)
                if log and False:
                        log.status("Trajectory frame %d generated" % i)

        # Add last frame with coordinates equal to final position.
        coordsets.append(coordset1.copy())

        return coordsets

def interpolateCorkscrew(xf, c0, c1, f):
        """Interpolate by splitting the transformation into a rotation
        and a translation along the axis of rotation."""
        from chimerax.core import geometry
        dc = c1 - c0
        vr, a = xf.rotation_axis_and_angle()        # a is in degrees
        tra = dc * vr                                # magnitude of translation
                                                # along rotation axis
        vt = dc - tra * vr                        # where c1 would end up if
                                                # only rotation is used
        cm = c0 + vt / 2
        v0 = geometry.cross_product(vr, vt)
        if geometry.norm(v0) <= 0.0:
                ident = geometry.identity()
                return ident, ident
        v0 = geometry.normalize_vector(v0)
        if a != 0.0:
                import math
                l = geometry.norm(vt) / 2 / math.tan(math.radians(a / 2))
                cr = cm + v0 * l
        else:
                import numpy
                cr = numpy.array((0.0, 0.0, 0.0))

        Tinv = geometry.translation(-cr)
        R0 = geometry.rotation(vr, a * f)
        R1 = geometry.rotation(vr, -a * (1 - f))
        X0 = geometry.translation(cr + vr * (f * tra)) * R0 * Tinv
        X1 = geometry.translation(cr - vr * ((1 - f) * tra)) * R1 * Tinv
        return X0, X1

def interpolateIndependent(xf, c0, c1, f):
        """Interpolate by splitting the transformation into a rotation
        and a translation."""
        from chimerax.core import geometry
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        xt = xf.translation()
        Tinv = geometry.translation(-xt)
        T = geometry.translation(xt * f)
        X0 = T * geometry.rotation(vr, a * f)
        X1 = T * geometry.rotation(vr, -a * (1 - f)) * Tinv
        return X0, X1

def interpolateLinear(xf, c0, c1, f):
        """Interpolate by translating c1 to c0 linearly along with
        rotation about translation point."""
        from chimerax.core import geometry
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        Tinv0 = geometry.translation(-c0)
        Tinv1 = geometry.translation(-c1)
        dt = c1 - c0
        R0 = geometry.rotation(vr, a * f)
        R1 = geometry.rotation(vr, -a * (1 - f))
        T = geometry.translation(c0v + dt * f)
        X0 = T * R0 * Tinv0
        X1 = T * R1 * Tinv1
        return X0, X1

SegmentMotionMethods = {
        "corkscrew": interpolateCorkscrew,
        "independent": interpolateIndependent,
        "linear": interpolateLinear,
}

def rateLinear(frames):
        "Generate fractions from 0 to 1 linearly (excluding start/end)"
        return [ float(s) / frames for s in range(1, frames) ]

def rateSinusoidal(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (slow at beginning, fast in middle, slow at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = math.pi + s * math.pi
                v = math.cos(a)
                r = (v + 1) / 2
                rate.append(r)
        return rate

def rateRampUp(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (slow at beginning, fast at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = math.pi + s * piOverTwo
                v = math.cos(a)
                r = v + 1
                rate.append(r)
        return rate

def rateRampDown(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (fast at beginning, slow at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = s * piOverTwo
                r = math.sin(a)
                rate.append(r)
        return rate

RateMap = {
        "linear": rateLinear,
        "sinusoidal": rateSinusoidal,
        "ramp up": rateRampUp,
        "ramp down": rateRampDown,
}

def residue_interpolators(residues, cartesian):
        # Create interpolation function for each residue.
        from .interp_residue import internal_residue_interpolator, cartesian_residue_interpolator
        residue_interpolator = cartesian_residue_interpolator if cartesian else internal_residue_interpolator
        res_interp = {}
        t0 = time()
        for r in residues:
                res_interp[r] = residue_interpolator(r)
        t1 = time()
        global rsit
        rsit += t1-t0
        return res_interp
