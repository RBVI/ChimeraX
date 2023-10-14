# vim: set expandtab ts=4 sw=4:

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

from chimerax.mouse_modes import MouseMode
class DistMouseMode(MouseMode):
    name = 'distance'
    icon_file = 'distance.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._first_atom = None

    def enable(self):
        self.session.logger.status(
            "Right-click on two atoms to show distance, or on distance to hide", color="green")

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        x,y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        self._picked_object(pick)

    def _picked_object(self, pick):
        warning = lambda txt: self.session.logger.status(
            "Distance mouse mode: %s" % txt, color = "red")
        message = self.session.logger.status
        from chimerax.atomic import PickedAtom, PickedPseudobond
        from chimerax.core.commands import run
        if isinstance(pick, PickedAtom):
            if self._first_atom and self._first_atom.deleted:
                self._first_atom = None
            if self._first_atom:
                if pick.atom == self._first_atom:
                    warning("same atom picked twice")
                else:
                    a1, a2 = self._first_atom, pick.atom
                    command = "dist %s %s" % (a1.string(style="command line"),
                        a2.string(style="command line"))
                    from chimerax.geometry import distance
                    message("Distance from %s to %s is %g" % (a1, a2,
                        distance(a1.scene_coord, a2.scene_coord)))
                    self._first_atom = None
                    run(self.session, command)
            else:
                self._first_atom = pick.atom
                message("Distance from %s to..." % pick.atom)
        elif isinstance(pick, PickedPseudobond):
            if pick.pbond.group.name == "distances":
                a1, a2 = pick.pbond.atoms
                command = "~dist %s %s" % (a1.string(style="command line"),
                        a2.string(style="command line"))
                message("Removing distance")
                run(self.session, command)
            else:
                warning("not a distance")
        else:
            warning("no atom/distance picked by mouse click")

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        self._picked_object(pick)
