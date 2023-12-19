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

def compute_morph(mols, log, method = 'corkscrew', rate = 'linear', frames = 20,
                  cartesian = False, match_same = False, core_fraction = 0.5, min_hinge_spacing = 6,
                  color_segments = False, color_core = None):
        from time import time
        t0 = time()
        motion = MolecularMotion(mols[0], method = method, rate = rate, frames = frames,
                                 match_same = match_same, core_fraction = core_fraction,
                                 min_hinge_spacing = min_hinge_spacing, log = log)
        traj = motion.trajectory()
        res_interp = None
        for i, mol in enumerate(mols[1:]):
                log.status("Computing interpolation %d\n" % (i+1))
                res_groups, atom_map, trimmed = motion.segment_and_pair_atoms(mol)
                if res_interp is None or trimmed:
                        from .interpolate import ResidueInterpolator
                        res_interp = ResidueInterpolator(traj.residues, cartesian, log)
                motion.interpolate(res_groups, atom_map, res_interp)
                if color_segments and i == 0:
                        from random import seed, randint
                        seed(1)
                        for rg in res_groups:
                                c = (randint(128,255), randint(128,255), randint(128,255), 255)
                                rg.ribbon_colors = c
                                rg.atoms.colors = c
                if color_core and i == 0:
                        rgba = color_core.uint8x4()
                        for r in traj.residues:
                                if getattr(r, '_in_morph_core', False):
                                        r.ribbon_color = rgba
                                        r.atoms.colors = rgba
        traj.active_coordset_id = 1	# Start at initial trajectory frame.
        t1 = time()
        log.status('Computed morph %d frames in %.3g seconds' % (traj.num_coordsets, t1-t0))
        return traj

class MolecularMotion:

        def __init__(self, m, method = "corkscrew", rate = "linear", frames = 20,
                     match_same = False, core_fraction = 0.5, min_hinge_spacing = 6, log = None):
                """
                Compute a trajectory that starting from molecule m conformation.
                Subsequent calls to interpolate must supply molecules
                that have the exact same set of atoms as 'm'.
                Currently support keyword options are:

                        method                string, default "corkscrew"
                                        Use interpolation method 'method'.
                                        Known methods are "corkscrew",
                                        "independent" and "linear".
                        rate                string, default "linear"
                                        Interpolate frames so that the
                                        trajectory motion appears to be
                                        "rate": either "linear" (constant
                                        motion throughout) or "sinusoidal"
                                        (fast in middle, slow at ends).
                        frames                integer, default 20
                                        Number of intermediate frames to
                                        generate in trajectory
                        match_same      Whether to match atoms with same chain id,
                                        same residue number and same atom name.
                        core_fraction   Fraction of atoms in chain that align best
                                        to move rigidly.
                	min_hinge_spacing  Minimum length of consecutive residue segment
                			   to move rigidly.
                        log             Logger for providing status messages
                """

                # Make a copy of the molecule to hold the computed trajectory
                from .util import copyMolecule
                tmol, atomMapMol, residueMapMol = copyMolecule(m)
                tmol.name = "Morph - %s" % m.name
                self.mol = tmol

                self.method = method
                self.rate = rate
                self.frames = frames
                self.match_same = match_same
                self.core_fraction = core_fraction
                self.min_hinge_spacing = min_hinge_spacing
                self.log = log

        def segment_and_pair_atoms(self, m):
                """Divide into residue segments and pair atoms."""

                #
                # Find matching set of residues.  First try for
                # one-to-one residue match, and, if that fails,
                # then finding a common set of residues.
                #
                sm = self.mol
                cf = self.core_fraction
                mhs = self.min_hinge_spacing
                log = self.log
                from .segment import segmentHingeSame, segmentHingeExact, segmentHingeApproximate
                if self.match_same:
                        results = segmentHingeSame(sm, m, cf, mhs, log=log)
                else:
                        from .segment import AtomPairingError
                        try:
                                results = segmentHingeExact(sm, m, cf, mhs, log=log)
                        except AtomPairingError:
                                try:
                                        results = segmentHingeApproximate(sm, m, cf, mhs, log=log)
                                except AtomPairingError as e:
                                        from chimerax.core.errors import UserError
                                        raise UserError(str(e))

                segments, atom_map = results
                trimmed = (len(atom_map) < sm.num_atoms)
                if trimmed:
                        from chimerax.atomic import Atoms
                        paired_atoms = Atoms(tuple(atom_map.keys()))
                        unpaired_atoms = sm.atoms.subtract(paired_atoms)
                        unpaired_atoms.delete()

                if sm.deleted:
                        from chimerax.core.errors import UserError
                        raise UserError('No atoms matched')

                from chimerax.atomic import Residues
                res_groups = [Residues(r0) for r0,r1 in segments]
                return res_groups, atom_map, trimmed

        def interpolate(self, res_groups, atom_map, res_interp):
                '''
                Interpolate between current conformation in trajectory
                and new conformation using atom map for atom pairing.
                '''

                # Make coordinate set arrays for starting and final coordinates
                sm = self.mol
                nc = sm.coordset_size
                matoms = sm.atoms
                maindices = matoms.coord_indices
                from numpy import float64, empty
                coords0 = empty((nc,3), float64)
                coords0[:] = -10000
                coords0[maindices] = matoms.coords
                coords1 = empty((nc,3), float64)
                coords1[:] = -10000
                # Convert to trajectory local coordinates.
                xform = sm.scene_position.inverse()
                from chimerax.atomic import Atoms
                coords1[maindices] = xform * Atoms([atom_map[a] for a in matoms]).scene_coords
                from .interpolate import SegmentInterpolator
                seg_interp = SegmentInterpolator(res_groups, self.method, coords0, coords1)

                from .interpolate import interpolate
                coordsets = interpolate(coords0, coords1, seg_interp, res_interp,
                                        self.rate, self.frames, sm.session.logger)
                base_id = max(sm.coordset_ids) + 1
                for i, cs in enumerate(coordsets):
                        sm.add_coordset(base_id + i, cs)
                sm.active_coordset_id = base_id + i

        def trajectory(self):
                return self.mol
