# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.commands import register, CmdDesc, AtomSpecArg
from chimerax.core.commands import BoolArg, FloatArg, IntArg, EnumOf, Or

def initialize(command_name):
    register("findclash", findclash_desc, findclash)

def findclash(session, spec=None, make_pseudobonds=False, log=True,
              naming_style="command", overlap_cutoff=0.6,
              hbond_allowance=0.4, bond_separation=4, test="other",
              intra_residue=False):
    from chimerax.core.errors import LimitationError
    from chimerax.core.atomic import Atoms
    from chimerax.core.commands import atomspec
    from chimerax.core.core_settings import settings
    if test != "self":
        raise LimitationError("findclash test \"%s\" not implemented" % test)
    if hbond_allowance != 0:
        session.logger.warning("Hydrogen bond finding is not implemented.  "
                               "Setting hbond_allowance to 0.0.")
        hbond_allowance = 0
    if naming_style != "command" and naming_style != "command-line":
        raise LimitationError("findclash naming style \"%s\" not implemented" %
                              naming_style)
    if make_pseudobonds:
        raise LimitationError("findclash make pseudobonds not implemented")
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    atoms = results.atoms
    neighbors = _find_neighbors(atoms, bond_separation)
    clashes = []
    for i in range(len(atoms)-1):
        a = atoms[i]
        if intra_residue:
            # Want intra-residue contacts, so check all atoms
            others = atoms[i+1:]
        else:
            # Do not want intra-residue contacts, so remove atoms from residue
            from numpy import logical_not
            others = atoms[i+1:]
            residues = others.residues
            # Generate mask for atoms from same residue
            mask = others.residues.mask(Atoms([a.residue]))
            others = others.filter(logical_not(mask))
        # Remove atoms within "separation" bonds of this atom
        others -= neighbors[a]
        if len(others) > 0:
            clashes.extend(_find_clash_self(a, others,
                                            overlap_cutoff, hbond_allowance))
    if log:
        session.logger.info("Allowed overlap: %g" % overlap_cutoff)
        session.logger.info("H-bond overlap reduction: %g" % hbond_allowance)
        session.logger.info("Ignored contact between atoms separated by %d "
                            "bonds or less" % bond_separation)
        session.logger.info("%d contacts" % len(clashes))
        session.logger.info("atom1\tatom2\toverlap\tdistance")
        save = settings.atomspec_contents
        settings.atomspec_contents = "command-line specifier"
        msgs = ["%s\t%s\t%.3f\t%.3f" % c for c in clashes]
        settings.atomspec_contents = save
        session.logger.info('\n'.join(msgs))
        session.logger.info("%d contacts" % len(clashes))
findclash_desc = CmdDesc(required=[("spec", AtomSpecArg)],
                         keyword=[("make_pseudobonds", BoolArg),
                                  ("log", BoolArg),
                                  ("naming_style", EnumOf(["simple",
                                                           "command",
                                                           "command-line",
                                                           "serial",
                                                           "serialnumber"])),
                                  ("overlap_cutoff", FloatArg),
                                  ("hbond_allowance", FloatArg),
                                  ("bond_separation", IntArg),
                                  ("test", Or(EnumOf(["self",
                                                      "other",
                                                      "model"]),
                                              AtomSpecArg)),
                                  ("intra_residue", BoolArg),
                                 ],
                       synopsis='find clashes between atom pairs')

def _find_neighbors(atoms, separation):
    # Find neighboring atoms within "separation" bonds
    # Return as a dictionary of Atoms
    from chimerax.core.atomic import Atoms
    m = {}
    for a in atoms:
        neighbors = set()
        check_list = set([a])
        for i in range(separation):
            new_check_list = set()
            while check_list:
                ca = check_list.pop()
                atoms = ca.neighbors
                new_check_list.update(atoms)
                neighbors.update(atoms)
            check_list = new_check_list
        m[a] = Atoms(list(neighbors))
    return m


def _find_clash_self(a, others, cutoff, allowance):
    from numpy import where
    from numpy.linalg import norm
    dist = norm(others.scene_coords - a.scene_coord, axis=1)
    # TODO: Only subtract allowance for potentially hydrogen-bonded pairs
    overlap = (others.radii + a.radius) - dist - allowance
    return [(a, others[i], overlap[i], dist[i])
            for i in where(overlap >= cutoff)[0]]
