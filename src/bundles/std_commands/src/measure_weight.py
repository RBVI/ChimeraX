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

def measure_weight(session, atoms):
    '''
    Report the molecular weight in Daltons of the specified atoms.
    '''
    weight = sum(atoms.elements.masses)
    spec = getattr(atoms, 'spec', '')
    msg = f'Molecular weight of {len(atoms)} atoms {spec} is {"%.0f"%weight} daltons'
    session.logger.status(msg, log = True)
    return weight
            
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', AtomsArg)],
        synopsis = 'compute molecular weight of specified atoms')
    register('measure weight', desc, measure_weight, logger=logger)
