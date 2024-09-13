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

from chimerax.core.errors import LimitationError
#import pyKVFinder

def cmd_kvfinder(session, structures=None):
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    from .prep import get_struct_input
    for s in structures:
        insert_codes = s.residues.insertion_codes
        if len(insert_codes[insert_codes != '']) > 0:
            raise LimitationError(
                "KVFinder cannot handle structures that have residues with insertion codes.")
        struct_input = get_struct_input(s)
    raise NotImplementedError("kvfinder command not fully implemented")

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg
    from chimerax.atomic import AtomicStructuresArg
    kw = { 'required': [('structures', Or(AtomicStructuresArg, EmptyArg))],
        'keyword': [], }
    register(command_name, CmdDesc(**kw, synopsis="Find pockets"), cmd_kvfinder, logger=logger)
