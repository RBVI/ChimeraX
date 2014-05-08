mol_attrs = ('path', 'id', 'database_fetch', 'display', 'atom_style',
             'ribbon_radius', 'ball_scale')

def molecule_state(m):  
    ms = {'place':m.position.matrix}
    for attr in mol_attrs:
        if hasattr(m,attr):
            ms[attr] = getattr(m,attr)
    if len(m.positions) > 1:
        ms['copies'] = tuple(c.matrix for c in m.positions)
    if not m.bonds is None:
        ms['has_bonds'] = True
    ms['atom_shown'] = array_to_string(m.atom_shown)
    if not m.atom_colors is None:
        ms['atom_colors'] = array_to_string(m.atom_colors)
    ms['ribbon_shown'] = array_to_string(m.ribbon_shown)
    if not m.ribbon_colors is None:
        ms['ribbon_colors'] = array_to_string(m.ribbon_colors)
    return ms

def restore_molecules(mstate, session, file_paths, attributes_only = False):
    if attributes_only:
        mids = dict((m.id, m) for m in session.molecules())
    from ..file_io.opensave import open_files
    for ms in mstate:
        if attributes_only:
            m = mids.get(ms['id'])
        else:
            if 'database_fetch' in ms:
                db_id, db_name = ms['database_fetch']
                from ..file_io import fetch
                mlist = fetch.fetch_from_database(db_id, db_name, session)
                if len(mlist) != 1:
                    session.show_info('Database fetch %s from %s unexpectedly contained %d models'
                                      % (db_id, db_name, len(mlist),))
                    continue
                session.add_models(mlist)
            else:
                p = file_paths.find(ms['path'])
                if p is None:
                    continue
                mlist = open_files([p], session, set_camera = False)
                if len(mlist) != 1:
                    session.show_info('File %s unexpectedly contained %d models' % (p, len(mlist),))
                    continue
            m = mlist[0]
        if m:
            set_molecule_state(m, ms, session)
    return True

def set_molecule_state(m, ms, session):
    from ..geometry.place import Place
    m.position = Place(ms['place'])
    if 'copies' in ms and len(ms['copies']) > 0:
        m.positions = [Place(c) for c in ms['copies']]
    if 'displayed' in ms:
        ms['display'] = ms['displayed']     # Fix old session files
    for attr in mol_attrs:
        if attr in ms:
            setattr(m, attr, ms[attr])
    if 'has_bonds' in ms and ms['has_bonds'] and m.bonds is None:
        from . import connect
        bonds, missing = connect.molecule_bonds(m, session)
        m.bonds = bonds
    from numpy import bool
    if 'atom_shown' in ms:
        sa = string_to_array(ms['atom_shown'], bool)
        sac = sa.sum()
        if sac == m.atom_count():
            m.atom_shown = sa
            m.atom_shown_count = sac
    if 'atom_colors' in ms:
        m.atom_colors = string_to_array(ms['atom_colors'])
    if 'color_mode' in ms:
        # Old sessions.
        cm = ms['color_mode']
        if cm == 'by element':
            m.color_by_element()
        elif cm == 'by chain':
            m.color_by_chain()
    if 'ribbon_shown' in ms:
        sr = string_to_array(ms['ribbon_shown'], bool)
        src = sr.sum()
        if src == m.atom_count():
            m.ribbon_shown = sr
            m.ribbon_shown_count = src
    if 'ribbon_colors' in ms:
        m.ribbon_colors = string_to_array(ms['ribbon_colors'])

    m.need_graphics_update = True

# convert a numpy array to a string
def array_to_string(a):
    s = a.tostring()
    import bz2
    cs = bz2.compress(s)
    import base64
    ecs = base64.b64encode(cs)
    aspec = (a.shape, a.dtype.str, ecs)
    return aspec

# convert a string to a numpy array
def string_to_array(s, dtype = None):
    if isinstance(s,tuple):
        shape, atype, astr = s
        import base64, bz2, numpy
        ds = base64.b64decode(astr)
        dds = bz2.decompress(ds)
        a1d = numpy.fromstring(dds, atype)
        a = numpy.reshape(a1d, shape)
    else:
        # Old style only handled 1d arrays.
        import base64, bz2, numpy
        ds = base64.b64decode(s)
        dds = bz2.decompress(ds)
        a = numpy.fromstring(dds, dtype)
    return a
