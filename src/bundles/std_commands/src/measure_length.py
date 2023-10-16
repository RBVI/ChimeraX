# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def measure_length(session, bonds):
    '''
    Compute sum of length of bonds.
    '''
    if len(bonds) > 0:
        a1, a2 = bonds.atoms
        v = a1.scene_coords - a2.scene_coords
        from numpy import sqrt
        length = sqrt((v*v).sum(axis=1)).sum()
    else:
        length = 0
    msg = 'Total length of %d bonds = %.5g' % (len(bonds), length)
    session.logger.status(msg, log = True)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import BondsArg
    desc = CmdDesc(
        required = [('bonds', BondsArg)],
        synopsis = 'compute sum of lengths of bonds')
    register('measure length', desc, measure_length, logger=logger)
