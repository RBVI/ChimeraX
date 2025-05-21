# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import register, CmdDesc, EnumOf, AtomSpecArg, atomspec

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
