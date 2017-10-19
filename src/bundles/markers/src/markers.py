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

from chimerax.core.atomic import Structure
class MarkerSet(Structure):

    def __init__(self, session, name = 'markers'):
        from chimerax.core.atomic import Structure
        Structure.__init__(self, session, name = 'markers', auto_style = False)
        self.ball_scale = 1.0

    def create_marker(self, xyz, rgba, radius, id):
        a = self.new_atom('', 'H')
        a.coord = xyz
        a.color = rgba	# 0-255 values
        a.radius = radius
        a.draw_mode = a.BALL_STYLE	# Sphere style hides bonds between markers, so use ball style.
        chain_id = 'M'
        r = self.new_residue('mark', chain_id, id)
        r.add_atom(a)
        self.new_atoms()
        return a

    @staticmethod
    def restore_snapshot(session, data):
        s = MarkerSet(session)
        Structure.set_state_from_snapshot(s, session, data)
        return s

    
def create_link(atom1, atom2, rgba = None, radius = None):
    m = atom1.structure
    b = m.new_bond(atom1,atom2)
    if rgba is None:
        rgba = (255,255,0,255)
    if radius is None:
        radius = 1.0
    b.radius = radius
    b.color = rgba
    b.halfbond = False
    return b
