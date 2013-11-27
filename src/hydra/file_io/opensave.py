from ..ui.qt import QtGui, QtWidgets

def show_open_file_dialog(view):
    '''
    Display the Open file dialog for opening data files.
    '''
    filter_lines = ['%s (%s)' % (name, ' '.join('*.%s' % s for s in suffixes))
                    for name, suffixes, read_func in file_types()]
    filter_lines.insert(0, 'All (*.*)')
    filters = ';;'.join(filter_lines)
    from .history import history
    dir = history.most_recent_directory()
    if dir is None:
        dir = '.'
    qpaths = QtWidgets.QFileDialog.getOpenFileNames(view, 'Open File', dir, filters)
    open_files(qpaths[0], view)
    from ..ui.gui import main_window as mw
    mw.show_graphics()

ftypes = None
def file_types():
    '''
    Return a list of file readers, each reader being represented by 3-tuple
    consisting of a file type name, a list of recognized suffixes, and a function
    that opens that file type give a path.  The funtion returns a list of models
    which have not been added to the scene.
    '''
    global ftypes
    if ftypes is None:
        from .pdb import open_pdb_file, open_mmcif_file
        from .read_stl import read_stl
        from .read_apr import open_autopack_results
        from .read_swc import read_swc
        ftypes = [
            ('PDB', ['pdb','ent'], open_pdb_file),
            ('mmCIF', ['cif'], open_mmcif_file),
            ('Session', ['hy'], open_session),
            ('AutoPack', ['apr'], open_autopack_results),
            ('STL', ['stl'], read_stl),
            ('Neuron SWC', ['swc'], read_swc),
            ('Python', ['py'], read_python),
        ]
        # Add map file types
        from ..map.data.fileformats import file_types as mft
        map_file_types = [(d, suffixes, open_map) for d,t,prefixes,suffixes,batch in mft]
        ftypes.extend(map_file_types)
    return ftypes

def file_readers():
    r = {}
    for name, suffixes, read_func in file_types():
        for s in suffixes:
            r['.' + s] = read_func
    return r

def open_files(paths, view = None, set_camera = None):
    '''
    Open data files and add the models created to the scene.  The file types are recognized
    using the file suffix as listed in the list returned by file_types().
    '''
    if view is None:
        from ..ui.gui import main_window
        view = main_window.view
    if set_camera is None:
        set_camera = (len(view.models) == 0)
    r = file_readers()
    opened = []
    models = []
    from os.path import splitext, isfile
    for path in paths:
        ext = splitext(path)[1]
        if not isfile(path):
            from ..ui import gui
            gui.show_status('File not found "%s"' % path)
            # TODO issue warning.
        elif ext in r:
            file_reader = r[ext]
            mlist = file_reader(path)
            if not isinstance(mlist, (list, tuple)):
                mlist = [mlist]
            models.extend(mlist)
            for m in mlist:
                view.add_model(m)
            opened.append(path)
            if set_camera and file_reader == open_session:
                set_camera = False
        else:
            from ..ui import gui
            gui.show_status('Unknown file suffix "%s"' % ext)
            # TODO issue warning.
    finished_opening(opened, set_camera, view)
    return models

def finished_opening(opened, set_camera, view):
    if opened and set_camera:
        view.remove_overlays()
        view.initial_camera_view()
    from ..ui.gui import show_info, show_status
    if len(opened) == 1 and opened:
        msg = 'Opened %s' % opened[0]
        show_info(msg, color = '#000080')
        show_status(msg)
    elif len(opened) > 1:
        msg = 'Opened %d files' % len(opened)
        show_info(msg, color = '#000080')
        show_status(msg)

def open_map(map_path):
    '''
    Open a density map file having any of the known density map formats.
    '''
    from .. import map
    i = map.data.open_file(map_path)[0]
    map_drawing = map.volume_from_grid_data(i)
    map_drawing.new_region(ijk_step = (1,1,1), adjust_step = False)
    return map_drawing

last_session_path = None
def open_session(path):
    '''
    Open a session file.  The current session is closed.
    '''
    from ..ui.gui import main_window as mw
    from . import session
    session.restore_session(path, mw.view)
    global last_session_path
    last_session_path = path
    from ..ui.gui import show_info
    show_info('Opened %s' % path, color = '#000080')
    return []

def save_session(view):
    '''
    Save a session file using the session file path of the last loaded
    session, or if no session has been loaded then show a dialog to
    get the save path.
    '''
    global last_session_path
    if last_session_path is None:
        save_session_as(view)
    else:
        from . import session
        session.save_session(last_session_path, view)

def save_session_as(view):
    '''
    Save a session file, raising a dialog to enter the file path.
    '''

    global last_session_path
    dir = last_session_path
    if dir is None:
        from .history import history
        dir = history.most_recent_directory()
    filters = 'Session (*.hy)'
    path = QtWidgets.QFileDialog.getSaveFileName(view, 'Save Session',
                                                 dir, filters)
    if isinstance(path, tuple):
        path = path[0]      # PySide returns path and filter, not PyQt
    if not path:
        return

    path = str(path)        # Convert from QString
    from . import session
    session.save_session(path, view)
    from ..ui.gui import show_info
    show_info('Saved %s' % path, color = '#000080')

def save_image(view):
    '''
    Save a JPEG image of the current graphics window contents.
    '''
    filters = 'JPEG image (*.jpg)'
    path = QtWidgets.QFileDialog.getSaveFileName(view, 'Save Image', '.',
                                             filters)
    if isinstance(path, tuple):
        path = path[0]      # PySide returns path and filter, not PyQt
    if not path:
        return
    i = view.image()
    i.save(path, 'JPG')

def open_image(view):
    filters = 'JPEG image (*.jpg)'
    path = QtWidgets.QFileDialog.getOpenFileName(view, 'Open Image', '.',
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

def open_command(cmdname, args):

    from ..ui.commands import string_arg
    from ..ui.commands import parse_arguments
    req_args = (('path', string_arg),)
    opt_args = ()
    kw_args = (('fromDatabase', string_arg),)

    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    open_file(**kw)

def open_file(path, from_database = None, set_camera = None):
    from ..ui.gui import main_window as mw
    view = mw.view
    if from_database is None:
        from os.path import expanduser
        p = expanduser(path)
        from os.path import isfile
        if isfile(p):
            open_files([p], view)
        else:
            if ':' in p:
                dbname, id = p.split(':', 1)
            elif len(p) == 4 or len(p.split(',', maxsplit = 1)[0]) == 4:
                dbname, id = 'PDB', p
            else:
                from ..ui import gui
                gui.show_status('Unknown file %s' % path)
                return
            open_file(id, from_database = dbname)
    else:
        ids = path.split(',')
        if set_camera is None:
            set_camera = (len(view.models) == 0)
        from . import fetch
        mlist = []
        for id in ids:
            m = fetch.fetch_from_database(id, from_database)
            if isinstance(m, (list, tuple)):
                mlist.extend(m)
            else:
                mlist.append(m)
        view.add_models(mlist)
        finished_opening([m.path for m in mlist], set_camera, view)
    mw.show_graphics()

def close_command(cmdname, args):

    from ..ui.commands import models_arg, parse_arguments
    req_args = ()
    opt_args = (('models', models_arg),)
    kw_args = ()

    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    close_models(**kw)

def close_models(models = None):
    '''
    Close a list of models, or all models if none are specified.
    '''
    from ..ui.gui import main_window
    v = main_window.view
    if models is None:
        v.close_all_models()
    else:
        v.close_models(models)

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
