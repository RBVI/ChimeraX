# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.state import State
#class BondRotation(State):
class BondRotation:
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

    def new_rotater(self, ident, moving_side, one_shot):
        rotater = BondRotater(self.session, self, ident, moving_side, one_shot)
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

#class BondRotater(State):
class BondRotater:
    # instances given to API users; works in conjunction with BondRotation
    def __init__(self, session, rotation, ident, moving_side, one_shot):
        self.session = session
        self.rotation = rotation
        self.ident = ident
        self.moving_side = moving_side
        self.one_shot = one_shot
        self._angle = 0.0

    def get_angle(self):
        return self._angle

    def set_angle(self, angle):
        if angle == self._angle:
            return
        delta = self._angle - angle
        self._angle = angle
        moving, fixed = self.moving_side.coord, self.bond.other_atom(self.moving_side).coord
        from chimerax.core.geometry import z_align, rotation
        za = z_align(moving, fixed)
        update = za.inverse() * rotation((0,0,1), delta) * za
        side_atoms = self.bond.side_atoms(self.moving_side)
        coords = side_atoms.coords
        # avoid a copy...
        update.move(coords)
        side_atoms.coords = coords
        self.rotation._rotater_update(self, delta)
        # manager listening on changes will fire 'modified' trigger...

    angle = property(get_angle, set_angle)

    @property
    def axis(self):
        moving, fixed = self.moving_side.coord, self.bond.other_atom(self.moving_side).coord
        from chimerax.core.geometry import normalize_vector
        axis = normalize_vector(moving - fixed)
        return axis

