def interpolate(mol, molXform, segments, equivAtoms, method, rMethod,
                        frames, cartesian, cb):
        # mol                molecule where new coordinate sets are added
        #                should already have first frame in place
        # molXform        transform to convert atoms from their own
        #                local coordinate system into "mol" local coordinates
        #                (usually [inverse transform of trajectory model] x
        #                [transform of target model])
        # segments        list of 2-tuples of matching residue lists
        # equivAtoms        dictionary of equivalent atoms
        #                key is atom from first frame
        #                value is 2-tuple of atoms from last frame and "mol"
        # method        interpolation method name
        # rMethod        rate profile, e.g., "linear"
        # frames        number of frames to generate in trajectory
        # cartesian        use cartesian coordinate interpolation?
        # cb                function called for every frame generated 

        from . import interp_residue
        from .util import getAtomList
        from numpy import mean
        from chimerax.core.geometry import align_points
        from chimerax.core.atomic import Atoms

        interpolateFunction = InterpolationMap[method]
        rateFunction = RateMap[rMethod]
        if cartesian:
                planFunction = interp_residue.planCartesian
        else:
                planFunction = interp_residue.planInternal

        rate = rateFunction(frames)
        numFrames = len(rate) + 1
        csSize = mol.num_atoms
        baseCS = max(mol.coordset_ids) + 1
        segMap = {}
        plan = {}
        for seg in segments:
                rList0, rList1 = seg
                aList0 = Atoms(getAtomList(rList0))
                aList1 = Atoms([ equivAtoms[a0] for a0 in aList0 ])
                c1 = aList1.coords.mean(axis = 0)
                segMap[seg] = (aList0, aList1, molXform * c1)
                for r in rList0:
                        plan[r] = planFunction(r)

        lo = 0.0
        interval = 1.0
        for i in range(len(rate)):
                f = (rate[i] - lo) / interval
                lo = rate[i]
                interval = 1.0 - lo
                from numpy import zeros, float32
                cs = zeros((csSize,3), float32)
                for seg in segments:
                        aList0, aList1, c1 = segMap[seg]
                        cList0 = aList0.coords
                        c = cList0.mean(axis = 0)
                        cList1 = molXform * aList1.coords
                        xform, rmsd = align_points(cList0, cList1)
                        xf, xf1 = interpolateFunction(xform, c, c1, f)
                        rList0, rList1 = seg
                        for r in rList0:
                                interp_residue.applyPlan(plan[r], r, cs, f,
                                                equivAtoms, xf, xf1 * molXform)
                cs_id = baseCS + i
                mol.add_coordset(cs_id, cs)
                mol.active_coordset_id = cs_id
                if cb:
                        cb(mol)
        from numpy import zeros, float32
        cs = zeros((csSize,3), float32)
        for a0, a1 in equivAtoms.items():
                cs[a0.coord_index] = molXform * a1.coord
        cs_id = baseCS + len(rate)
        mol.add_coordset(cs_id, cs)
        mol.active_coordset_id = cs_id

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

InterpolationMap = {
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
