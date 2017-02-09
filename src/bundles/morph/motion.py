def compute_morph(mols, log, method = 'corkscrew', rate = 'linear', frames = 20, cartesian = False):
        motion = MolecularMotion(mols[0], method = method, rate = rate, frames = frames, cartesian = cartesian)
        for i, mol in enumerate(mols[1:]):
                log.status("Computing interpolation %d\n" % (i+1))
                motion.interpolate(mol)
        traj = motion.trajectory()
        return traj

class MolecularMotion:

        def __init__(self, m, method = "corkscrew", rate = "linear", frames = 20, cartesian = False):
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
                """

                # Make a copy of the molecule to hold the computed trajectory
                from .util import copyMolecule
                tmol, atomMapMol, residueMapMol = copyMolecule(m)
                tmol.name = "Morph - %s" % m.name
                self.mol = tmol

                self.method = method
                self.rate = rate
                self.frames = frames
                self.cartesian = cartesian

        def interpolate(self, m):
                """Interpolate to new conformation 'm'."""

                #
                # Find matching set of residues.  First try for
                # one-to-one residue match, and, if that fails,
                # then finding a common set of residues.
                #
                from . import segment
                sm = self.mol
                try:
                        results = segment.segmentHingeExact(sm, m)
                except ValueError:
                        results = segment.segmentHingeApproximate(sm, m)
                segments, atomMap, unusedResidues, unusedAtoms = results
                unusedResidues.delete()
                unusedAtoms.delete()

                if sm.deleted:
                        from chimerax.core.errors import UserError
                        raise UserError('No atoms matched')
                #
                # Interpolate between last conformation in trajectory
                # and new conformations
                #
                sm.active_coordset_id = max(sm.coordset_ids)
                from .interpolate import interpolate
                combinedXform = sm.scene_position.inverse() * m.scene_position
                interpolate(sm, combinedXform,
                                segments, atomMap, self.method,
                                self.rate, self.frames,
                                self.cartesian, self.report_progress)

        def report_progress(self, mol):
                log = mol.session.logger
                log.status("Trajectory frame %d generated" % mol.num_coord_sets)

        def trajectory(self):
                return self.mol
