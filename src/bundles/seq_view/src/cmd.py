# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

from chimerax.core.commands import register
from chimerax.core.commands.cli import RegisteredCommandInfo

registry = RegisteredCommandInfo()

# all the commands use the trick that the run() function
# temporarily puts a copy of the seq_view instance into
# the global namespace as '_sv'
def scf_shim(session, file_name, color_structures=True):
    _sv.load_scf_file(file_name, color_structures=color_structures)
from chimerax.core.commands import CmdDesc
from chimerax.core.commands import OpenFileNameArg, BoolArg

class OpenScfFileNameArg(OpenFileNameArg):
    name_filter = "SCF files (*.scf *.seqsel)"

register("scfLoad",
    CmdDesc(
        required=[('file_name', OpenScfFileNameArg)],
        keyword=[('color_structures', BoolArg)],
        synopsis='Load SCF file'),
    scf_shim, registry=registry)

def run(session, sv, text):
    from chimerax.core.commands import Command
    cmd = Command(session, registry=registry)
    global _sv
    _sv = sv
    try:
        cmd.run(text, log=False)
    finally:
        _sv = None
