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

from chimerax.core.commands import register, CmdDesc, EnumOf, AtomSpecArg

def initialize(command_name, logger):
    register(command_name, dumpspec_desc, dumpspec, logger=logger)

def dumpspec(session, atoms=None):
    if atoms is None:
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    msgs = []

    models = session.models.list()
    models.sort(key=lambda m: m.id)
    msgs.append("Sorted session model order:")
    for m in models:
        msgs.append("  %s" % m)

    msgs.append("\"%s\" matches:" % atoms)
    models = results.models
    msgs.append("  %d models" % len(models))
    for m in models:
        msgs.append("    %s" % m)

    instances = results.model_instances
    msgs.append("  %d model instances" % len(instances))
    for mi in instances:
        msgs.append("    %s" % mi)

    atoms = results.atoms
    msgs.append("  %d atoms (should match %d)" % (len(atoms), results.num_atoms))
    for a in atoms:
        msgs.append("    %s" % a)
    session.logger.info('\n'.join(msgs))
dumpspec_desc = CmdDesc(required=[("atoms", AtomSpecArg),],
                        synopsis='report what an atom specifier matches')
