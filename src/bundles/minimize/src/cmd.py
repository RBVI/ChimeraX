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

from chimerax.core.errors import UserError
from chimerax.add_charge import ChargeMethodArg

def cmd_minimize(session, structures, *, charge_method=ChargeMethodArg.default_value, dock_prep=True,
        his_scheme=None):
        args = (session, structures, charge_method, his_scheme)
    if dock_prep:
        #TODO: pass charge_method/his_scheme to right step
        from chimerax.dock_prep import dock_prep_caller
        dock_prep_caller(session, structures, memorize_name="minimization",
            callback=lambda args=args: _minimize(*args))
    else:
        _minimize(*args)

#TODO: def _minimize(...):

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, EnumOf, BoolArg
    from chimerax.atomic import AtomicStructuresArg
    desc = CmdDesc(
        required = [('structures', Or(AtomicStructuresArg, EmptyArg))],
        keyword = [
            ('charge_method', ChargeMethodArg),
            ('dock_prep', BoolArg),
            ('his_scheme', EnumOf(['HIP', 'HIE', 'HID'])),
        ],
        synopsis = 'Minimize structures'
    )
    register("minimize", desc, cmd_minimize, logger=logger)
