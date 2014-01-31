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

def file_types(session):
    '''
    Return a list of file readers, each reader being represented by 3-tuple
    consisting of a file type name, a list of recognized suffixes, and a function
    that opens that file type give a path.  The funtion returns a list of models
    which have not been added to the scene.
    '''
    ftypes = session.file_types
    if ftypes is None:
        from .pdb import open_pdb_file, open_mmcif_file
        from .read_stl import read_stl
        from .read_apr import open_autopack_results
        from .read_swc import read_swc
        ftypes = [
            ('PDB', ['pdb','ent'], lambda path,s=session: open_pdb_file(path,s)),
            ('mmCIF', ['cif'], open_mmcif_file),
            ('Session', ['hy'], open_session),
            ('AutoPack', ['apr'], open_autopack_results),
            ('STL', ['stl'], read_stl),
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
    from .. import map
    i = map.data.open_file(map_path)[0]
    map_drawing = map.volume_from_grid_data(i, session)
    map_drawing.new_region(ijk_step = (1,1,1), adjust_step = False)
    return map_drawing

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

def save_image(path, session, width = None, height = None, format = 'JPG'):
    '''
    Save a JPEG image of the current graphics window contents.
    '''
    view = session.view
    if path is None:
        filters = 'JPEG image (*.jpg)'
        path = QtWidgets.QFileDialog.getSaveFileName(view.widget, 'Save Image', '.',
                                                     filters)
        if path is None:
            return

    import os.path
    path = os.path.expanduser(path)         # Tilde expansion
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        from ..ui import commands
        raise commands.CommandError('Directory "%s" does not exist' % dir)

    # Match current window aspect ratio
    # TODO: Allow different aspect ratios
    ww,wh = view.window_size
    if not width is None and not height is None:
        w,h = width,height
    elif not width is None:
        w = width
        h = (wh*w)//ww          # Choose height to match window aspect ratio.
    elif not height is None:
        h = height
        w = (ww*h)//wh          # Choose width to match window aspect ratio.
    else:
        w,h = ww,wh
    i = view.image(w, h)
    i.save(path, format)
    print ('saved image', path)

def imagesave_command(cmdname, args, session):

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
    from ..surface import Surface, surface_image
    from os.path import basename
    s = Surface(basename(path))
    surface_image(i, (-.5,-.5), 1, s)
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

def open_data(path, session, from_database = None, set_camera = None):

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
            ids = p.split(',')
            id0 = ids[0]
            if len(id0) != 4:
                session.show_status('Unknown file %s' % path)
                return []
            db = 'EMDB' if is_integer(id0) else 'PDB'
            mlist = open_from_database(ids, session, db, set_camera)

    session.main_window.show_graphics()
    if mlist:
        session.file_history.add_entry(path, from_database = db, models = mlist)
    return mlist

def open_from_database(ids, session, from_database, set_camera = None):

    if set_camera is None:
        set_camera = (session.model_count() == 0)
    from . import fetch
    mlist = []
    for id in ids:
        m = fetch.fetch_from_database(id, from_database, session)
        if isinstance(m, (list, tuple)):
            mlist.extend(m)
        else:
            mlist.append(m)
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

def read_python(path):
    '''
    Read a Python file and execute the code.
    '''
    f = open(path)
    code = f.read()
    f.close()
    ccode = compile(code, path, 'exec')
    globals = locals = None
    exec(ccode, globals, locals)
    return []

def is_integer(s):
    try:
        int(s)
    except ValueError:
        return False
    return True
