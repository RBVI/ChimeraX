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
        self._bond_rot = self._picked_bond_rotation(pick, move_smaller_side = not event.shift_down())
    
    def mouse_drag(self, event):
        br = self._bond_rotation
        if br:
            dx, dy = self.mouse_motion(event)
            br.angle += dy

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        self._log_command()
        self._delete_bond_rotation()

    def _log_command(self):
        br = self._bond_rotation
        if br:
            log_torsion_command(br)
    
    def wheel(self, event):
        pick = self._picked_bond(event)
        br = self._picked_bond_rotation(pick, move_smaller_side = not event.shift_down())
        if br:
            d = event.wheel_value()
            br.angle += d
            self.session.bond_rotations.delete_rotation(br)

    def _picked_bond(self, event):
        x,y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        return pick

    def _picked_bond_rotation(self, pick, move_smaller_side = True):
        from chimerax.atomic import PickedBond
        if isinstance(pick, PickedBond):
            from .manager import BondRotationError
            try:
                br = self.session.bond_rotations.new_rotation(pick.bond, move_smaller_side = move_smaller_side)
                self.session.logger.status('Rotating bond %s' % str(pick.bond))
            except BondRotationError as e:
                self.session.logger.status(str(e))
                br = None
        else:
            br = None
        return br

    @property
    def _bond_rotation(self):
        br = self._bond_rot
        if br and br.bond.deleted:
            self._bond_rot = br = None
        return br
    
    def _delete_bond_rotation(self):
        br = self._bond_rotation
        if br is not None:
            self.session.bond_rotations.delete_rotation(br)
            self._bond_rot = None

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        self._bond_rot = br = self._picked_bond_rotation(pick)
        if br:
            br.bond.selected = True
        
        # Move the side of the bond the VR click is closest to.
        # Would like to have a command to enable this mode for rotating bonds
        # with small ligands
        move_closer_side = False
        if move_closer_side and br is not None:
            atom1 = br.moving_side
            atom2 = br.bond.other_atom(atom1)
            p = event.tip_position
            from chimerax.geometry import distance
            if distance(p, atom2.scene_coord) < distance(p, atom1.scene_coord):
                br.moving_side = atom2
        
    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        br = self._bond_rotation
        if br:
            axis, angle = event.motion.rotation_axis_and_angle()
            from chimerax.geometry import inner_product
            if inner_product(axis, br.axis) < 0:
                angle = -angle
            angle_change = self._speed_factor * angle
            if abs(angle_change) < self._minimum_angle_step:
                return "accumulate drag"
            br.angle += angle_change

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        br = self._bond_rotation
        if br:
            br.bond.selected = False
            self._log_command()
            self._delete_bond_rotation()

def log_torsion_command(bond_rotator):
    bond = bond_rotator.rotation.bond
    ms_atom = bond_rotator.moving_side
    fs_atom = bond.other_atom(ms_atom)
    ms_atom2 = _connected_atom(ms_atom, fs_atom)
    fs_atom2 = _connected_atom(fs_atom, ms_atom)
    if ms_atom2 is None or fs_atom2 is None:
        return 		# No connected atom to define a torsion
    side = '' if bond.smaller_side is ms_atom else 'move large'
    from chimerax.geometry import dihedral
    torsion = dihedral(fs_atom2.scene_coord, fs_atom.scene_coord,
                       ms_atom.scene_coord, ms_atom2.scene_coord)

    atom_specs = '%s %s %s %s' % (fs_atom2.string(style='command'), fs_atom.string(style='command'),
                                  ms_atom.string(style='command'), ms_atom2.string(style='command'))

    # Use simpler atom spec for the common case of rotating a side chain.
    res = ms_atom.residue
    if ms_atom2.residue is res and fs_atom.residue is res and fs_atom2.residue is res:
        if 'serial_number' not in atom_specs:  # serial_number indicates a duplicate atom name.
            atom_specs = '%s@%s,%s,%s,%s' % (res.string(style = 'command'),
                                             ms_atom2.name, ms_atom.name, fs_atom.name, fs_atom2.name)

    cmd = 'torsion %s %.2f %s' % (atom_specs, torsion, side)
    ses = ms_atom.structure.session
    from chimerax.core.commands import run
    run(ses, cmd)

def _connected_atom(atom, exclude_atom):
    for a in atom.neighbors:
        if a is not exclude_atom:
            return a
    return None
    
    
