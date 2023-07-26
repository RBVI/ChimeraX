# vim: set et sw=4 sts=4:
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
from chimerax.core.session import Session
from chimerax.core.commands import StringArg, CmdDesc
from chimerax.core.errors import UserError

__all__ = ['taskman', 'taskman_desc']

def taskman(session: Session, action: str, job: str = None) -> None:
    if action == 'list':
        if len(session.tasks.list()) == 0:
            session.logger.info("No tasks running")
        else:
            session.logger.info("\n".join([str(task) for task in session.tasks.values()]))
    elif action == 'kill':
        if not job:
            raise UserError("Job keyword required for command 'kill'")
    elif action == 'pause':
        pass
    else:
        raise UserError("Unsupported action. Please use one of [list, kill, pause].")

taskman_desc: CmdDesc = CmdDesc(
    required = [("action", StringArg)],
    optional = [("job", StringArg)],
    synopsis = "Manage tasks on the ChimeraX command line"
)
