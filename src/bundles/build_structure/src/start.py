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

from chimerax.atomic import Element

def place_helium(structure, res_name, position=None):
    '''If position is None, place in the center of view'''
    max_existing = 0
    for r in structure.residues:
        if r.chain_id == "het" and r.number > max_existing:
            max_existing = r.number
    res = structure.new_residue(res_name, "het", max_existing+1)
    if position is None:
        if len(structure.session.models) == 0:
            from numpy import array
            position = array([0.0,0.0,0.0])
        else:
            view = structure.session.view
            n, f = view.near_far_distances(view.camera, None)
            position = view.camera.position.origin() + (n+f) * view.camera.view_direction() / 2
    from chimerax.atomic.struct_edit import add_atom
    helium = Element.get_element("He")
    a = add_atom("He", helium, res, position)
    from chimerax.atomic.colors import element_color
    a.color = element_color(helium.number)
    a.draw_mode = a.BALL_STYLE
    return a
