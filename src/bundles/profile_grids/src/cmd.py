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

from chimerax.core.commands.cli import RegisteredCommandInfo

registry = RegisteredCommandInfo()

# all the commands use the trick that the run() function
# temporarily puts a copy of the Profile Grid instance into
# the global namespace as '_pg'

def label_cmd(session, residues=None, *, height=1.5, offset=(0,0,3), on_top=False, palette=None, range=None):
    from chimerax.core.errors import UserError
    alignment = _pg.alignment
    assoc_residues = alignment.associated_residues()
    if not assoc_residues:
        raise UserError("No chains are associated with the alignment")
    if residues is None:
        label_residues = assoc_residues
    else:
        assoc_set = set(assoc_residues)
        label_residues = [r for r in residues if r in assoc_set]
        if not label_residues:
            raise UserError("None of the specified residues are associated with the alignment")
    #TODO: similar code to "mutationscores label"

from chimerax.core.commands import CmdDesc, register
from chimerax.core.commands import Or, EmptyArg, FloatArg, Color8Arg, Float3Arg, BoolArg, ColormapArg
from chimerax.core.commands import ColormapRangeArg
from chimerax.atomic import ResiduesArg

register("label",
    CmdDesc(
        required=[('residues', Or(ResiduesArg, EmptyArg))],
        keyword=[('height', FloatArg), ('offset', Float2Arg), ('on_top', BoolArg), ('palette', ColormapArg),
            ('range', ColormapRangeArg)],
        synopsis='Label residues with grid data'),
    label_cmd, registry=registry)

def run(session, pg, text):
    from chimerax.core.commands import Command
    cmd = Command(session, registry=registry)
    global _pg
    _pg = pg
    try:
        cmd.run(text, log=False)
    finally:
        _pg = None
