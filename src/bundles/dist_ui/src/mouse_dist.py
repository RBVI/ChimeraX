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
        self.session.logger.status("Distance mouse mode:"
            " right-click on two atoms to show(/hide) distance", color="green")

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        warning = lambda txt: self.session.logger.status(
            "Distance mouse mode: %s" % txt, color = "red")
        message = self.session.logger.status
        from chimerax.core.atomic import PickedAtom
        if isinstance(pick, PickedAtom):
            if self._first_atom and self._first_atom.deleted:
                self._first_atom = None
            if self._first_atom:
                if pick.atom == self._first_atom:
                    warning("same atom picked twice")
                else:
                    pbg = self.session.pb_manager.get_group("distances", create=False)
                    pbs = pbg.pseudobonds if pbg else []
                    for pb in pbs:
                        a1, a2 = pb.atoms
                        if (a1 == self._first_atom and pick.atom == a2) \
                        or (a1 == pick.atom and self._first_atom == a2):
                            command = "~dist %s %s" % (a1.string(style="command line"),
                                a2.string(style="command line"))
                            message("Removing distance")
                            break
                    else:
                        a1, a2 = self._first_atom, pick.atom
                        command = "dist %s %s" % (a1.string(style="command line"),
                            a2.string(style="command line"))
                        from chimerax.core.geometry import distance
                        message("Distance from %s to %s is %g" % (a1, a2, distance(a1.scene_coord,
                            a2.scene_coord)))
                    self._first_atom = None
                    from chimerax.core.commands import run
                    run(self.session, command)
            else:
                self._first_atom = pick.atom
                message("Distance from %s to..." % pick.atom)
        else:
            warning("no atom picked by mouse click")
