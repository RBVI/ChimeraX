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

def time(session, command):
    '''Time a command.'''
    from chimerax.core.commands import run
    from time import time
    t0 = time()
    run(session, command)
    t1 = time()

    session.triggers.add_handler('frame drawn', lambda name,data,s=session,t=t1: _report_draw_time(s,t))

    msg = 'command time %.4g seconds' % (t1-t0)
    log = session.logger
    log.status(msg)
    log.info(msg)

def _report_draw_time(session, tstart):
    from time import time
    tend = time()
    session.logger.info('draw time %.4g seconds' % (tend-tstart))
    from chimerax.core.triggerset import DEREGISTER
    return DEREGISTER

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, RestOfLine
    desc = CmdDesc(required=[('command', RestOfLine)],
                   synopsis='time a command')
    register('time', desc, time, logger=logger)
