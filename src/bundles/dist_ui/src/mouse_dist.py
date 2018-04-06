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

from chimerax.ui import MouseMode
class DistMouseMode(MouseMode):
    name = 'distance'
    icon_file = 'distance.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._first_atom = None

    def enable(self):
        self.session.logger.status(
            "Distance mouse mode: right-click on two atoms to show(/hide) distance")

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        message = lambda txt: self.session.logger.status(
            "Distance mouse mode: %s" % txt, color = "red")
        from chimerax.core.atomic import PickedAtom
        if isinstance(pick, PickedAtom):
            if self._first_atom and self._first_atom.deleted:
                self._first_atom = None
            if self._first_atom:
                if pick.atom == self._first_atom:
                    message("same atom picked twice")
                else:
                    pbg = self.session.pb_manager.get_group("distances", create=False)
                    pbs = pbg.pseudobonds if pbg else []
                    for pb in pbs:
                        a1, a2 = pb.atoms
                        if (a1 == self._first_atom and pick.atom == a2) \
                        or (a1 == pick.atom and self._first_atom == a2):
                            command = "~dist %s %s" % (a1.string(style="command line"),
                                a2.string(style="command line"))
                            break
                    else:
                        command = "dist %s %s" % (self._first_atom.string(style="command line"),
                            pick.atom.string(style="command line"))
                    self._first_atom = None
                    from chimerax.core.commands import run
                    run(self.session, command)
            else:
                self._first_atom = pick.atom
        else:
            message("no atom picked by mouse click")
