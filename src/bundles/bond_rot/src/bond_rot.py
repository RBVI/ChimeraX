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

from chimerax.core.state import State
class BondRotation(State):
    """A bond rotation, which can have multiple BondRotaters associated.
    If any BondRotaters rotate the bond, BondRotation will update the others.
    
    Should only be created by the bond-rotation manager
    """

    def __init__(self, session, bond):
        self.session = session
        self.bond = bond
        self.rotaters = []

    def delete_rotater(self, rotater):
        self.rotaters.remove(rotater)

    def new_rotater(self, moving_side, one_shot):
        rotater = BondRotater(self.session, self, moving_side, one_shot)
        self.rotaters.append(rotater)
        return rotater

    def _rotater_update(self, initiator, delta):
        for r in self.rotaters:
            if r == initiator:
                continue
            if r.moving_side == initiator.moving_side:
                r._angle += delta
            else:
                r._angle -= delta

    # session methods
    def reset_state(self, session):
        # manager will nuke everything
        pass

    @staticmethod
    def restore_snapshot(session, data):
        return BondRotation(session, data['bond'])

    def take_snapshot(self, session, flags):
        # to avoid circularity, don't save rotaters
        # they will add themselves to self.rotaters
        return {
            'version': 1,

            'bond': self.bond,
        }

class BondRotater(State):
    # instances given to API users; works in conjunction with BondRotation
    def __init__(self, session, rotation, moving_side, one_shot):
        self.session = session
        self.rotation = rotation
        self.moving_side = moving_side
        self.one_shot = one_shot
        self._angle = 0.0
        self._undo_state = None

    def get_angle(self):
        return self._angle

    def set_angle(self, angle):
        if angle == self._angle:
            return
        delta = angle - self._angle
        self._angle = angle
        moving, fixed = self.moving_side.coord, self.bond.other_atom(self.moving_side).coord
        from chimerax.geometry import z_align, rotation
        za = z_align(moving, fixed)
        update = za.inverse() * rotation((0,0,-1), delta) * za
        side_atoms = self.bond.side_atoms(self.moving_side)
        coords = side_atoms.coords
        # avoid a copy...
        update.transform_points(coords, in_place=True)
        if self._undo_state:
            self._undo_state.add(side_atoms, "coords", side_atoms.coords, coords)
        side_atoms.coords = coords
        self.rotation._rotater_update(self, delta)
        # manager listening on changes will fire 'modified' trigger...

    angle = property(get_angle, set_angle)

    @property
    def axis(self):
        moving, fixed = self.moving_side.coord, self.bond.other_atom(self.moving_side).coord
        from chimerax.geometry import normalize_vector
        axis = normalize_vector(moving - fixed)
        return axis

    @property
    def bond(self):
        return self.rotation.bond

    def swap_sides(self):
        self.moving_side = self.bond.other_atom(self.moving_side)
        if not self.one_shot:
            manager = self.session.bond_rotations
            manager.triggers.activate_trigger(manager.REVERSED, self)

    @property
    def undo_state(self):
        return self._undo_state

    @undo_state.setter
    def undo_state(self, undo_state):
        if self._undo_state is not None:
            self.session.undo.register(self._undo_state)
        self._undo_state = undo_state

    # session methods
    def reset_state(self, session):
        # manager will nuke everything
        pass

    @staticmethod
    def restore_snapshot(session, data):
        rotater = BondRotater(session, data['rotation'], data['moving_side'], data['one_shot'])
        # to avoid circularity, BondRotation doesn't save the rotaters,
        # so add the rotater to BondRotation...
        data['rotation'].rotaters.append(rotater)
        return rotater

    def take_snapshot(self, session, flags):
        return {
            'version': 2,

            'rotation': self.rotation,
            'moving_side': self.moving_side,
            'one_shot': self.one_shot,
            'angle': self._angle,
        }

