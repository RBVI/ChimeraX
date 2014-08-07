from ..ui.qt import QtGui, QtWidgets

def show_open_file_dialog(session):
    '''
    Display the Open file dialog for opening data files.
    '''
    filter_lines = ['%s (%s)' % (name, ' '.join('*.%s' % s for s in suffixes))
                    for name, suffixes, read_func in file_types(session)]
    filter_lines.insert(0, 'All (*.*)')
    filters = ';;'.join(filter_lines)
    dir = session.file_history.most_recent_directory()
    if dir is None:
        dir = '.'
    v = session.main_window.view
    qpaths = QtWidgets.QFileDialog.getOpenFileNames(v.widget, 'Open File', dir, filters)
    # Return value is a 2-tuple holding list of paths and filter string.
    paths = qpaths[0]
    mlist = open_files(paths, session)
    if mlist:
        session.file_history.add_entry(','.join(paths), models = mlist)
    session.main_window.show_graphics()

# -----------------------------------------------------------------------------
# Add label to open file dialog
#
def locate_file_dialog(path):

    pdir = existing_directory(path)
    caption = 'Locate %s' % path
    from ..ui.qt import QtWidgets
    fd = QtWidgets.QFileDialog(None, caption, pdir)
    # Cannot add widgets to native dialog.
    fd.setOptions(QtWidgets.QFileDialog.DontUseNativeDialog)
    result = [None]
    def file_selected(path, result = result):
        result[0] = path
    fd.fileSelected.connect(file_selected)
    lo = fd.layout()
    lbl = QtWidgets.QLabel('Locate missing file %s' % path, fd)
    lo.addWidget(lbl)
    fd.exec()
    p = result[0]
    return p

# -----------------------------------------------------------------------------
# Find the deepest directory of the given path that exists.
#
def existing_directory(path):

    from os.path import dirname, isdir
    while path:
        d = dirname(path)
        if isdir(d):
            return d
        else:
            path = d
    return None

def file_types(session):
    '''
    Return a list of file readers, each reader being represented by 3-tuple
    consisting of a file type name, a list of recognized suffixes, and a function
    that opens that file type give a path.  The funtion returns a list of models
    which have not been added to the scene.
    '''
    ftypes = session.file_types
    if ftypes is None:
        from ..molecule.pdb import open_pdb_file
        from ..molecule.mmcif import open_mmcif_file
        from .read_stl import read_stl
        from .read_apr import open_autopack_results, read_ingredient_file, read_sphere_file
        from .read_swc import read_swc
        from .collada import read_collada_surfaces
        ftypes = [
            ('PDB', ['pdb','ent'], lambda path,s=session: open_pdb_file(path,s)),
            ('mmCIF', ['cif'], open_mmcif_file),
            ('Session', ['hy'], open_session),
            ('AutoPack', ['apr'], open_autopack_results),
            ('AutoPack Ingredient', ['xml'], read_ingredient_file),
            ('AutoPack sphere file', ['sph'], read_sphere_file),
            ('STL', ['stl'], read_stl),
            ('Collada', ['dae'], read_collada_surfaces),
            ('Neuron SWC', ['swc'], read_swc),
            ('Python', ['py'], read_python),
        ]
        # Add map file types
        from ..map.data.fileformats import file_types as mft
        map_file_types = [(d, suffixes, lambda p,s=session: open_map(p,s))
                          for d,t,prefixes,suffixes,batch in mft]
        ftypes.extend(map_file_types)
        session.file_types = ftypes
    return ftypes

def file_readers(session):
    r = {}
    for name, suffixes, read_func in file_types(session):
        for s in suffixes:
            r['.' + s] = read_func
    return r

def file_writers(session):
    from . import session_file, write_stl
    from ..map.data import fileformats
    w = {'.png': save_image_command,
         '.jpg': save_image_command,
         '.ppm': save_image_command,
         '.bmp': save_image_command,
         '.hy': lambda cmdname,path,session: session_file.save_session(path, session),
         '.mrc': fileformats.save_map_command,
         '.stl': write_stl.write_stl_command,
         }
    return w

def open_files(paths, session, set_camera = None):
    '''
    Open data files and add the models created to the scene.  The file types are recognized
    using the file suffix as listed in the list returned by file_types().
    '''
    if set_camera is None:
        set_camera = (session.model_count() == 0)
    r = file_readers(session)
    opened = []
    models = []
    from os.path import splitext, isfile
    for path in paths:
        ext = splitext(path)[1]
        if not isfile(path):
            session.show_status('File not found "%s"' % path)
            # TODO issue warning.
        elif ext in r:
            file_reader = r[ext]
            mlist = file_reader(path, session)
            if not isinstance(mlist, (list, tuple)):
                mlist = [mlist]
            models.extend(mlist)
            for m in mlist:
                session.add_model(m)
            opened.append(path)
            if set_camera and file_reader == open_session:
                set_camera = False
        else:
            session.show_status('Unknown file suffix "%s"' % ext)
            # TODO issue warning.
    finished_opening(opened, set_camera, session)
    return models

def finished_opening(opened, set_camera, session):
    s = session
    if opened and set_camera:
        view = s.main_window.view
        view.remove_overlays()
        view.initial_camera_view()

def report_opening(opened, session):
    s = session
    if len(opened) == 1 and opened:
        msg = 'Opened %s' % opened[0]
        s.show_info(msg, color = '#000080')
        s.show_status(msg)
    elif len(opened) > 1:
        msg = 'Opened %d files' % len(opened)
        s.show_info(msg, color = '#000080')
        s.show_status(msg)

def open_map(map_path, session):
    '''
    Open a density map file having any of the known density map formats.
    '''
    maps = []
    from ..map import data, volume_from_grid_data
    grids = data.open_file(map_path)
    for i,d in enumerate(grids):
        show = (i == 0)
        v = volume_from_grid_data(d, session, open_model = False, show_data = show)
        v.new_region(ijk_step = (1,1,1), adjust_step = False, show = show)
        maps.append(v)
    if len(maps) > 1:
        from os.path import basename
        name = basename(map_path)
        from ..map.series import Map_Series
        return [Map_Series(name, maps)]
    return maps

def open_session(path, session):
    '''
    Open a session file.  The current session is closed.
    '''
    from . import session_file
    session_file.restore_session(path, session)
    session.last_session_path = path
    session.show_info('Opened %s' % path, color = '#000080')
    return []

def save_session(session):
    '''
    Save a session file using the session file path of the last loaded
    session, or if no session has been loaded then show a dialog to
    get the save path.
    '''
    if session.last_session_path is None:
        save_session_as(session)
    else:
        from . import session_file
        session_file.save_session(session.last_session_path, session)

def save_session_as(session):
    '''
    Save a session file, raising a dialog to enter the file path.
    '''

    dir = session.last_session_path
    if dir is None:
        dir = session.file_history.most_recent_directory()
    filters = 'Session (*.hy)'
    parent = session.main_window.view.widget
    path = QtWidgets.QFileDialog.getSaveFileName(parent, 'Save Session',
                                                 dir, filters)
    if isinstance(path, tuple):
        path = path[0]      # PySide returns path and filter, not PyQt
    if not path:
        return

    path = str(path)        # Convert from QString
    from . import session_file
    session_file.save_session(path, session)
    session.show_info('Saved %s' % path, color = '#000080')

def save_command(cmdname, args, session):
    a0 = args.split()[0]
    from os.path import splitext
    e = splitext(a0)[1]
    fw = file_writers(session)
    if e in fw:
        save_func = fw[e]
        save_func(cmdname, args, session)
    else:
        session.show_status('Unknown save file type %s' % a0)

def save_image(path, session, width = None, height = None, format = None):
    '''
    Save an image of the current graphics window contents.
    '''
    view = session.view
    if path is None:
        filters = 'Image (*.jpg *.png *.ppm *.bmp)'
        pf = QtWidgets.QFileDialog.getSaveFileName(view.widget, 'Save Image', '.', filters)
        # Returns path and filter name.
        if pf is None:
            return
        path = pf[0]

    import os.path
    path = os.path.expanduser(path)         # Tilde expansion
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        from ..ui import commands
        raise commands.CommandError('Directory "%s" does not exist' % dir)

    if format is None:
        from os.path import splitext
        format = splitext(path)[1][1:].upper()
        if not format in ('PNG', 'JPG', 'PPM', 'BMP'):
            from ..ui import commands
            raise commands.CommandError('Unrecognized image file suffix "%s"' % format)

    i = view.image(width, height)
    i.save(path, format)
    print ('saved image', path)

def save_image_command(cmdname, args, session):

    from ..ui.commands import string_arg, int_arg, parse_arguments
    req_args = (('path', string_arg),)
    opt_args = ()
    kw_args = (('width', int_arg),
               ('height', int_arg),
               ('format', string_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    save_image(**kw)

def open_image(session):
    filters = 'JPEG image (*.jpg)'
    view = session.main_window.view
    parent = view.widget
    path = QtWidgets.QFileDialog.getOpenFileName(parent, 'Open Image', '.',
                                             filters)
    if isinstance(path, tuple):
        path = path[0]      # PySide returns path and filter, not PyQt
    if not path:
        return
    i = QtGui.QImage(path, 'JPG')
    from ..graphics import Drawing, image_drawing
    from os.path import basename
    s = Drawing(basename(path))
    image_drawing(i, (-.5,-.5), 1, s)
    view.add_overlay(s)

def open_command(cmdname, args, session):

    from ..ui.commands import string_arg
    from ..ui.commands import parse_arguments
    req_args = (('path', string_arg),)
    opt_args = ()
    kw_args = (('fromDatabase', string_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    if 'fromDatabase' in kw:
        kw['from_database'] = kw['fromDatabase']
        kw.pop('fromDatabase')
    open_data(**kw)

def open_data(path, session, from_database = None, set_camera = None, history = True):

    db = from_database
    if not db is None:
        ids = path.split(',')
        mlist = open_from_database(ids, session, db, set_camera)
    else:
        from os.path import expanduser
        paths = [expanduser(p) for p in path.split(',')]
        p0 = paths[0]
        from os.path import isfile
        if isfile(p0):
            mlist = open_files(paths, session)
        else:
            ids = path.split(',')
            id0 = ids[0]
            if len(id0) != 4:
                session.show_status('Unknown file %s' % path)
                return []
            db = 'EMDB' if is_integer(id0) else 'PDB'
            mlist = open_from_database(ids, session, db, set_camera)

    session.main_window.show_graphics()
    if history and mlist:
        session.file_history.add_entry(path, from_database = db, models = mlist)
    return mlist

def open_from_database(ids, session, from_database, set_camera = None):

    if set_camera is None:
        set_camera = (session.model_count() == 0)
    from . import fetch
    mlist = []
#    from time import time
#    t0 = time()
    for id in ids:
        m = fetch.fetch_from_database(id, from_database, session)
        if isinstance(m, (list, tuple)):
            mlist.extend(m)
        else:
            mlist.append(m)
#    t1 = time()
#    print('opened in %.2f seconds, %s' % (t1-t0, ids))
    session.add_models(mlist)
    finished_opening([m.path for m in mlist], set_camera, session)
    return mlist

def close_command(cmdname, args, session):

    from ..ui.commands import models_arg, parse_arguments
    req_args = ()
    opt_args = (('models', models_arg),)
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    session.close_models(**kw)

def read_python(path, session):
    '''
    Read a Python file and execute the code.
    '''
    f = open(path)
    code = f.read()
    f.close()
    ccode = compile(code, path, 'exec')
    globals = locals = {}
    exec(ccode, globals, locals)
    return []

def is_integer(s):
    try:
        int(s)
    except ValueError:
        return False
    return True
