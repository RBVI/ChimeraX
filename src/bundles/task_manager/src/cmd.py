# vim: set et sw=4 sts=4:
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
from chimerax.core.session import Session
from chimerax.core.commands import IntArg, CmdDesc, EnumOf
from chimerax.core.errors import UserError

__all__ = ['taskman', 'taskman_desc']

def taskman(session: Session, action: str, job: int = None) -> None:
    if action == 'list':
        if len(session.tasks.list()) == 0:
            session.logger.info("No tasks running")
        else:
            session.logger.info("\n".join([str(task) for task in session.tasks.values()]))
    elif action == 'kill':
        if not job:
            raise UserError("Job keyword required for command 'kill'")
        task = session.tasks[int(job)]
        if not task:
            raise UserError("No such task: %s" % job)
        task.terminate()
    else:
        raise UserError("Unsupported action. Please use one of [list, kill].")


taskman_desc: CmdDesc = CmdDesc(
    required = [("action", EnumOf(["list", "kill"]))],
    optional = [("job", IntArg)],
    synopsis = "Manage tasks on the ChimeraX command line"
)
