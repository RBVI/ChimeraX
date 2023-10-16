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

from chimerax.atomic import Structure
class MarkerSet(Structure):

    def __init__(self, session, name = 'markers'):
        Structure.__init__(self, session, name = name, auto_style = False)
        self.ball_scale = 1.0

    def create_marker(self, xyz, rgba, radius, id = None):
        '''Position xyz is in the marker set coordinate system.'''
        a = self.new_atom('M', 'H')
        a.coord = xyz
        a.color = rgba	# 0-255 values
        a.radius = radius
        a.draw_mode = a.BALL_STYLE	# Sphere style hides bonds between markers, so use ball style.
        chain_id = 'M'
        if id is None:
            rnums = self.residues.numbers
            id = 1 if len(rnums) == 0 else (rnums.max() + 1)
        r = self.new_residue('mark', chain_id, id)
        a.serial_number = id
        r.add_atom(a)
        return a

    @staticmethod
    def restore_snapshot(session, data):
        s = MarkerSet(session)
        Structure.set_state_from_snapshot(s, session, data)
        return s

    def save_marker_attribute_in_sessions(self, attr_name, attr_type = None):
        from chimerax.atomic import Atom
        Atom.register_attr(self.session, attr_name, "markers", attr_type=attr_type)
    
def create_link(atom1, atom2, rgba = None, radius = None, log = False):
    m = atom1.structure
    b = m.new_bond(atom1,atom2)
    if rgba is None:
        rgba = (255,255,0,255)
    if radius is None:
        radius = 1.0
    b.radius = radius
    b.color = rgba
    b.halfbond = False
    if log:
        _log_link_command(atom1, atom2, rgba, radius)
    return b

def _log_link_command(marker1, marker2, rgba, radius):
    mspec = '%s:%d,%d' % (marker1.structure.atomspec, marker1.residue.number, marker2.residue.number)
    from chimerax.core.colors import color_name
    cmd = 'marker link %s color %s radius %.4g' % (mspec, color_name(rgba), radius)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(marker1.structure.session, cmd)

def selected_markers(session):
    from chimerax import atomic
    atoms = atomic.selected_atoms(session)
    mask = [isinstance(a.structure, MarkerSet) for a in atoms]
    return atoms.filter(mask)

def selected_links(session):
    from chimerax import atomic
    bonds = atomic.selected_bonds(session)
    mask = [isinstance(b.structure, MarkerSet) for b in bonds]
    return bonds.filter(mask)
