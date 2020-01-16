# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Register the meeting command for ChimeraX.
#
def register_command(ci, logger):
    from chimerax.core.commands import CmdDesc, register
    if ci.name == "meeting":
        from chimerax.core.commands import (StringArg, IntArg, ColorArg,
                                            OpenFileNameArg, BoolArg)
        desc = CmdDesc(optional = [('host', StringArg)],
                       keyword = [('port', IntArg),
                                  ('name', StringArg),
                                  ('color', ColorArg),
                                  ('face_image', OpenFileNameArg),
                                  ('copy_scene', BoolArg),
                                  ('relay_commands', BoolArg),
                                  ('update_interval', IntArg)])
        from .meeting import meeting as func
    elif ci.name == "meeting close":
        desc = CmdDesc()
        from .meeting import meeting_close as func
    elif ci.name == "meeting send":
        desc = CmdDesc()
        from .meeting import meeting_send as func
    elif ci.name == "conference":
        from chimerax.core.commands import (StringArg, IntArg, ColorArg,
                                            OpenFileNameArg, BoolArg, EnumOf)
        desc = CmdDesc(required=[("action", EnumOf(["start", "join"])),
                                 ("location", StringArg)],
                       keyword=[("color", ColorArg),
                                ("face_image", OpenFileNameArg),
                                ("copy_scene", BoolArg),
                                ("relay_commands", BoolArg),
                                ("update_interval", IntArg)])
        from .conference import conference as func
    elif ci.name == "conference set":
        from chimerax.core.commands import (StringArg, IntArg, ColorArg,
                                            OpenFileNameArg, BoolArg, EnumOf)
        desc = CmdDesc(keyword=[("color", ColorArg),
                                ("face_image", OpenFileNameArg),
                                ("copy_scene", BoolArg),
                                ("relay_commands", BoolArg),
                                ("update_interval", IntArg)])
        from .conference import conference_set as func
    elif ci.name == "conference close":
        desc = CmdDesc()
        from .conference import conference_close as func
    elif ci.name == "conference send":
        desc = CmdDesc()
        from .conference import conference_send as func
    else:
        raise ValueError("trying to register unknown command: %s" % ci.name)
    if desc.synopsis is None:
        desc.synopsis = ci.synopsis
    register(ci.name, desc, func, logger=logger)
