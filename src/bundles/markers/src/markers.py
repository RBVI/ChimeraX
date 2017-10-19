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

def marker_settings(session, attr = None):
    if not hasattr(session, '_marker_settings'):
        session._marker_settings = {
            'molecule': None,
            'next_marker_num': 1,
            'marker_chain_id': 'M',
            'marker color': (255,255,0,255),	# yellow
            'marker radius': 1.0,
            'link color': (101,156,239,255),	# cornflowerblue
            'link radius': 0.5,
            'placement_mode': 'maximum',        # Modes: 'maximum', 'plane', 'surface', 'surface center'
                                                #        'link', 'move', 'resize', 'delete'
            'link_new_markers': False,
        }
    s = session._marker_settings
    return s if attr is None else s[attr]

def _marker_molecule(session):
    ms = marker_settings(session)
    m = ms['molecule']
    if m is None or m.was_deleted:
        from chimerax.core.atomic import Structure
        mlist = [m for m in session.models.list(type = Structure) if m.name == 'markers']
        if mlist:
            m = mlist[0]
        else:
            m = new_markerset(session)
        ms['molecule'] = m
    return m

def new_markerset(session, add_to_session = True):
    from chimerax.core.atomic import Structure
    m = Structure(session, name = 'markers', auto_style = False)
    m.ball_scale = 1.0
    m.is_markerset = True
    if add_to_session:
        session.models.add([m])
    return m
    
def place_marker(session, center, link_to_selected = False, select = True):
    m = _marker_molecule(session)
    ms = marker_settings(session)
    a = create_marker(m, center, ms['marker color'], ms['marker radius'], ms['next_marker_num'])
    ms['next_marker_num'] += 1
    session.logger.status('Placed marker')
    if link_to_selected:
        from chimerax.core.atomic import selected_atoms
        atoms = selected_atoms(session)
        if len(atoms) == 1:
            al = atoms[0]
            if a.structure == al.structure and a is not al:
                create_link(al, a)
    if select:
        session.selection.clear()
        a.selected = True

def create_marker(mset, xyz, rgba, radius, id):
    a = mset.new_atom('', 'H')
    a.coord = xyz
    a.color = rgba	# 0-255 values
    a.radius = radius
    a.draw_mode = a.BALL_STYLE	# Sphere style hides bonds between markers, so use ball style.
    ms = marker_settings(a.structure.session)
    cid = ms['marker_chain_id']
    r = mset.new_residue('mark', cid, id)
    r.add_atom(a)
    mset.new_atoms()
    return a
    
def create_link(atom1, atom2, rgba = None, radius = None):
    m = atom1.structure
    b = m.new_bond(atom1,atom2)
    if rgba is None:
        ms = marker_settings(m.session)
        rgba = ms['link color']
    if radius is None:
        ms = marker_settings(m.session)
        radius = ms['link radius']
    b.radius = radius
    b.color = rgba
    b.halfbond = False
    return b
