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

def save_molecules(f, mlist):
    mstates = [molecule_state(m) for m in mlist]
    f.write("'molecules':(\n")
    from ..file_io.SessionUtil import objecttree
    objecttree.write_basic_tree(mstates, f, indent = ' ')
    f.write('),\n')

def restore_molecules(d, viewer):
    mstate = d.get('molecules')
    if mstate is None:
        return False
    from ..file_io.opensave import open_files
    for ms in mstate:
        p = ms['path']
        mlist = open_files([p], set_camera = False)
        if len(mlist) != 1:
            from ..ui.gui import show_info
            show_info('File %s unexpectedly contained %d models' % (len(mlist),))
            continue
        m = mlist[0]
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
