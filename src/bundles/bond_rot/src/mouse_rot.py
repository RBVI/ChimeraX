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

from chimerax.mouse_modes import MouseMode
class BondRotationMouseMode(MouseMode):
    name = 'bond rotation'
    icon_file = 'bondrot.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._bond_rot = None
        self._speed_factor = 2
        self._minimum_angle_step = 2	# Degrees.  Only applies to drag with 3d pointer.

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        pick = self._picked_bond(event)
        self._bond_rot = self._bond_rotation(pick)
    
    def mouse_drag(self, event):
        br = self._bond_rot
        if br:
            dx, dy = self.mouse_motion(event)
            br.angle += dy

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        self._delete_bond_rotation()
    
    def wheel(self, event):
        pick = self._picked_bond(event)
        br = self._bond_rotation(pick)
        if br:
            d = event.wheel_value()
            br.angle += d
            self.session.bond_rotations.delete_rotation(br)

    def _picked_bond(self, event):
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        return pick

    def _bond_rotation(self, pick):
        from chimerax.atomic import PickedBond
        if isinstance(pick, PickedBond):
            from .manager import BondRotationError
            try:
                br = self.session.bond_rotations.new_rotation(pick.bond)
                self.session.logger.status('Rotating bond %s' % str(pick.bond))
            except BondRotationError as e:
                self.session.logger.status(str(e))
                br = None
        else:
            br = None
        return br

    def _delete_bond_rotation(self):
        br = self._bond_rot
        if br is not None:
            self.session.bond_rotations.delete_rotation(br)
            self._bond_rot = None

    def laser_click(self, xyz1, xyz2):
        from chimerax.ui.mousemodes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        self._bond_rot = self._bond_rotation(pick)

    def drag_3d(self, position, move, delta_z):
        if move is None:
            self._delete_bond_rotation()
        else:
            br = self._bond_rot
            if br:
                axis, angle = move.rotation_axis_and_angle()
                from chimerax.core.geometry import inner_product
                if inner_product(axis, br.axis) < 0:
                    angle = -angle
                angle_change = self._speed_factor * angle
                if abs(angle_change) < self._minimum_angle_step:
                    return "accumulate drag"
                br.angle += angle_change
