mol_attrs = ('path', 'id', 'displayed', 'atom_style',
             'color_mode', 'ribbon_radius', 'ball_scale')

def molecule_state(m):  
    ms = {'place':m.place.matrix}
    for attr in mol_attrs:
        ms[attr] = getattr(m,attr)
    if m.copies:
        ms['copies'] = tuple(c.matrix for c in m.copies)
    if not m.bonds is None:
        ms['has_bonds'] = True
    ms['atom_shown'] = array_to_string(m.atom_shown)
    ms['ribbon_shown'] = array_to_string(m.ribbon_shown)
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
        m.bonds = connect.molecule_bonds(m)
    from numpy import bool
    if 'atom_shown' in ms:
        m.atom_shown = string_to_array(ms['atom_shown'], bool)
        m.atom_shown_count = m.atom_shown.sum()
    if 'ribbon_shown' in ms:
        m.ribbon_shown = string_to_array(ms['ribbon_shown'], bool)
        m.ribbon_shown_count = m.ribbon_shown.sum()

    m.need_graphics_update = True

# convert a numpy array to a string
def array_to_string(a):
    s = a.tostring()
    import bz2
    cs = bz2.compress(s)
    import base64
    ecs = base64.b64encode(cs)
    return ecs

# convert a string to a numpy array
def string_to_array(s, dtype):
    import base64, bz2, numpy
    ds = base64.b64decode(s)
    dds = bz2.decompress(ds)
    a = numpy.fromstring(dds, dtype)
    return a
