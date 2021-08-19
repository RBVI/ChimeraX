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
                if log and (i+1)%100 == 0:
                        log.status("Trajectory frame %d generated" % (i+1))

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
        def __init__(self, residues, cartesian, log = None):
                # Create interpolation function for each residue.
                from .interp_residue import internal_residue_interpolator, cartesian_residue_interpolator
                residue_interpolator = cartesian_residue_interpolator if cartesian else internal_residue_interpolator
                res_interp = {}
                t0 = time()
                cartesian_atoms = []
                dihedral_atoms = []
                nr = len(residues)
                for i,r in enumerate(residues):
                        if log and i%100 == 0:
                                log.status('Making interpolator for residue %d of %d' % (i, nr))
                        residue_interpolator(r, cartesian_atoms, dihedral_atoms)
                t1 = time()
                global rsit
                rsit += t1-t0
                from chimerax.atomic import Atoms
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
                from chimerax.geometry import align_points
                from chimerax.atomic import Atoms
                for rList in residue_groups:
                        aatoms = Atoms(segment_alignment_atoms(rList))
                        raindices = aatoms.coord_indices
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
                        xform.inverse().transform_points(ca, in_place = True)
                        coordset[atom_indices] = ca

        def interpolate(self, f, coordset):
                global rit
                t0 = time()
                for atom_indices, axis, angle, center, shift in self.motion_parameters:
                        apply_rigid_motion(coordset, atom_indices, axis, angle, center, shift, f)
                t1 = time()
                rit += t1-t0

def apply_rigid_motion_py(coordset, atom_indices, axis, angle, center, shift, f):
        from chimerax.geometry import rotation, translation
        xf = translation(f*shift) * rotation(axis, f*angle, center)
        ca = coordset[atom_indices]	# Copies array
        xf.transform_points(ca, in_place = True)  # Apply rigid segment motion
        coordset[atom_indices] = ca
# Use C++ optimized version
from .morph_cpp import apply_rigid_motion

def interpolate_corkscrew(xf, c0, c1, minimum_rotation = 0.1):
        '''
        Rotate and move along a circular arc perpendicular to the rotation axis and
        translate parallel the rotation axis.  This makes the initial geometric center c0
        traverse a helical path to the final center c1.  The circular arc spans an angle
        equal to the rotation angle so it is nearly a straight segment for small angles,
        and for the largest possible rotation angle of 180 degrees it is a half circle
        centered half-way between c0 and c1 in the plane perpendicular to the rotation axis.
        Rotations less than the minimum (degrees) are treated as no rotation.
        '''
        from chimerax.geometry import normalize_vector
        dc = c1 - c0
        axis, angle = xf.rotation_axis_and_angle()      # a is in degrees.
        if abs(angle) < minimum_rotation:
                # Use linear instead of corkscrew interpolation to
                # avoid numerical precision problems at small rotation angles.
                # ChimeraX bug #2928.
                center = c0
                shift = dc
        else:
                from chimerax.geometry import inner_product, cross_product, norm
                tra = inner_product(dc, axis)           # Magnitude of translation along rotation axis.
                shift = tra*axis
                vt = dc - tra * axis                    # Translation perpendicular to rotation axis.
                v0 = cross_product(axis, vt)
                if norm(v0) == 0 or angle == 0:
                        center = c0
                else:
                        import math
                        l = 0.5*norm(vt) / math.tan(math.radians(angle / 2))
                        center = c0 + 0.5*vt + l*normalize_vector(v0)

        return axis, angle, center, shift

def interpolate_linear(xf, c0, c1):
        'Rotate about center c0 and translate c0 linearly to c1.'
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        return vr, a, c0, c1-c0

def interpolate_independent(xf, c0, c1):
        '''
        Rotate about 0,0,0 and translate c0 to c1.  This makes little sense
        since 0,0,0 is an arbitrary point in the structure coordinate system,
        often the origin of the crystal cell.
        '''
        vr, a = xf.rotation_axis_and_angle()                # a is in degrees
        shift = xf.translation()
        from numpy import zeros, float64
        center = zeros((3,), float64)
        return vr, a, center, shift

segment_motion_methods = {
        "corkscrew": interpolate_corkscrew,
#        "independent": interpolate_independent,
        "linear": interpolate_linear,
        }
