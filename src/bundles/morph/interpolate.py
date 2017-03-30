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

from time import time
stt = rit = rst = rsit = 0
from time import time
def interpolate(coordset0, coordset1, segment_interpolator, residue_interpolator,
                rate_method, frames, log):
        '''
        coordset0              initial coordinates indexed by atom index
        coordset1              final coordinates indexed by atom index
        segment_interpolator   function to interpolate groups of residues rigidly
        residue_interpolator   function to interpolate residue conformations
        rate_method            spacing method for interpolation steps:
                                 "linear", "sinusoidal", "ramp up", "ramp down"
        frames                 number of frames to generate in trajectory
        log                    Logger for reporting progress messages
        '''

        # Calculate interpolated coordinates for each frame.
        coordsets = []
        rateFunction = RateMap[rate_method]
        rate = rateFunction(frames)  # Compute fractional steps controlling speed of motion.

        c1s = coordset1.copy()
        segment_interpolator.reverse_motion(c1s)

        for i, f in enumerate(rate):
                coordset = coordset0.copy()
                # Interpolate residue conformations
                t0 = time()
                residue_interpolator.interpolate(coordset0, c1s, f, coordset)
                t1 = time()
                global rst
                rst += t1-t0

                # Interplate segment motions
                segment_interpolator.interpolate(f, coordset)

                coordsets.append(coordset)
                if log and False:
                        log.status("Trajectory frame %d generated" % i)

        # Add last frame with coordinates equal to final position.
        coordsets.append(coordset1.copy())

        return coordsets

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

class ResidueInterpolator:
        def __init__(self, residues, cartesian):
                # Create interpolation function for each residue.
                from .interp_residue import internal_residue_interpolator, cartesian_residue_interpolator
                residue_interpolator = cartesian_residue_interpolator if cartesian else internal_residue_interpolator
                res_interp = {}
                t0 = time()
                cartesian_atoms = []
                dihedral_atoms = []
                for r in residues:
                        residue_interpolator(r, cartesian_atoms, dihedral_atoms)
                t1 = time()
                global rsit
                rsit += t1-t0
                from chimerax.core.atomic import Atoms
                self.cartesian_atom_indices = Atoms(cartesian_atoms).coord_indices
                self.dihedral_atom_indices = Atoms(dihedral_atoms).coord_indices
                
        def interpolate(self,coords0, coords1, f, coord_set):
                from .interp_residue import interpolate_linear, interpolate_dihedrals
                interpolate_linear(self.cartesian_atom_indices, coords0, coords1, f, coord_set)
                interpolate_dihedrals(self.dihedral_atom_indices, coords0, coords1, f, coord_set)

class SegmentInterpolator:
        def __init__(self, residue_groups, method, coordset0, coordset1):
                # Get transform for each rigid segment
                t0 = time()
                calc_motion_parameters = segment_motion_methods[method]
                self.motion_parameters = motion_params = []
                self.motion_transforms = motion_transforms = []
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
                        atom_indices = rList.atoms.coord_indices
                        motion_transforms.append((atom_indices, xform))
                        axis, angle, center, shift = calc_motion_parameters(xform, c0, c1)
                        motion_params.append((atom_indices, axis, angle, center, shift))
                t1 = time()
                global stt
                stt += t1-t0

        def reverse_motion(self, coordset):
                for atom_indices, xform in self.motion_transforms:
                        ca = coordset[atom_indices]	# Copies array
                        xform.inverse().move(ca)
                        coordset[atom_indices] = ca

        def interpolate(self, f, coordset):
                global rit
                t0 = time()
                for atom_indices, axis, angle, center, shift in self.motion_parameters:
                        apply_rigid_motion(coordset, atom_indices, axis, angle, center, shift, f)
                t1 = time()
                rit += t1-t0

def apply_rigid_motion_py(coordset, atom_indices, axis, angle, center, shift, f):
        from chimerax.core.geometry import rotation, translation
        xf = translation(f*shift) * rotation(axis, f*angle, center)
        ca = coordset[atom_indices]	# Copies array
        xf.move(ca)  # Apply rigid segment motion
        coordset[atom_indices] = ca
# Use C++ optimized version
from ._morph import apply_rigid_motion

def interpolate_corkscrew(xf, c0, c1):
        'Decompose transform as a rotation about an axis and shift along that axis.'
        from chimerax.core.geometry import inner_product, cross_product, identity, norm
        from chimerax.core.geometry import normalize_vector
        dc = c1 - c0
        vr, a = xf.rotation_axis_and_angle()        # a is in degrees
        tra = inner_product(dc, vr)                                # magnitude of translation
                                                # along rotation axis
        vt = dc - tra * vr                        # where c1 would end up if
                                                # only rotation is used
        cm = c0 + vt / 2
        v0 = cross_product(vr, vt)
        if norm(v0) <= 0.0:
                ident = identity()
                return ident, ident
        v0 = normalize_vector(v0)
        if a != 0.0:
                import math
                l = norm(vt) / 2 / math.tan(math.radians(a / 2))
                cr = cm + v0 * l
        else:
                import numpy
                cr = numpy.array((0.0, 0.0, 0.0))

        return vr, a, cr, tra*vr

def interpolate_linear(xf, c0, c1):
        'Rotate about center c0 and translate c0 linearly to c1.'
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        return vr, a, c0, c1-c0

def interpolate_independent(xf, c0, c1):
        'Rotate about 0,0,0 and translate that point linearly to destination.'
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        shift = xf.translation()
        from numpy import zeros, float64
        center = zeros((3,), float64)
        return vr, a, center, shift

segment_motion_methods = {
        "corkscrew": interpolate_corkscrew,
        "independent": interpolate_independent,
        "linear": interpolate_linear,
        }
