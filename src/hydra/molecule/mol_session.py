mol_attrs = ('path', 'id', 'displayed', 'show_atoms', 'atom_style',
             'color_mode', 'show_ribbons', 'ribbon_radius',
             'ball_scale')

def molecule_state(m):  
    ms = {'place':m.place.matrix}
    for attr in mol_attrs:
        ms[attr] = getattr(m,attr)
    if m.copies:
        ms['copies'] = tuple(c.matrix for c in m.copies)
    if not m.bonds is None:
        ms['has_bonds'] = True
    return ms

def restore_molecules(mstate, viewer, attributes_only = False):
    if attributes_only:
        mids = dict((m.id, m) for m in viewer.molecules())
    from ..file_io.opensave import open_files
    for ms in mstate:
        if attributes_only:
            m = mids.get(ms['id'])
        else:
            mlist = open_files([ms['path']], set_camera = False)
            if len(mlist) != 1:
                from ..ui.gui import show_info
                show_info('File %s unexpectedly contained %d models' % (ms['path'], len(mlist),))
                continue
            m = mlist[0]
        if m:
            set_molecule_state(m, ms)
    return True

def set_molecule_state(m, ms):
    from ..geometry.place import Place
    m.place = Place(ms['place'])
    m.copies = [Place(c) for c in ms.get('copies', [])]
    for attr in mol_attrs:
        if attr in ms:
            setattr(m, attr, ms[attr])
    if 'has_bonds' in ms and ms['has_bonds'] and m.bonds is None:
        from . import connect
        connect.create_molecule_bonds(m)
    m.need_graphics_update = True
