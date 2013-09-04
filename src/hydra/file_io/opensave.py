from ..ui.qt import QtGui, QtWidgets

def show_open_file_dialog(view):
    filter_lines = ['%s (%s)' % (name, ' '.join('*.%s' % s for s in suffixes))
                    for name, suffixes, read_func in file_types()]
    filter_lines.insert(0, 'All (*.*)')
    filters = ';;'.join(filter_lines)
    qpaths = QtWidgets.QFileDialog.getOpenFileNames(view, 'Open File', '.', filters)
    open_files(qpaths[0], view)
    from ..ui.gui import main_window as mw
    mw.show_graphics()

ftypes = None
def file_types():
    global ftypes
    if ftypes is None:
        from .pdb import open_pdb_file, open_mmcif_file
        from .read_stl import read_stl
        from .read_apr import open_autopack_results
        from .read_swc import read_swc
        ftypes = [
            ('PDB', ['pdb'], open_pdb_file),
            ('mmCIF', ['cif'], open_mmcif_file),
            ('Session', ['hy'], open_session),
            ('AutoPack', ['apr'], open_autopack_results),
            ('STL', ['stl'], read_stl),
            ('Neuron SWC', ['swc'], read_swc),
        ]
        # Add map file types
        from ..VolumeData.fileformats import file_types as mft
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
        else:
            from ..ui import gui
            gui.show_status('Unknown file suffix "%s"' % ext)
            # TODO issue warning.
    finished_opening(opened, set_camera, view)
    return models

def finished_opening(opened, set_camera, view):
    if opened and set_camera:
        view.remove_overlays()
        view.initial_camera_view() # TODO: don't do for session restore
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
    from .. import VolumeData
    i = VolumeData.open_file(map_path)[0]
    from .. import VolumeViewer
    map_drawing = VolumeViewer.volume_from_grid_data(i)
    map_drawing.new_region(ijk_step = (1,1,1), adjust_step = False)
    return map_drawing

last_session_path = None
def open_session(path):
    from ..ui.gui import main_window as mw
    from . import session
    session.restore_session(path, mw.view)
    global last_session_path
    last_session_path = path
    from ..ui.gui import show_info
    show_info('Opened %s' % path, color = '#000080')
    return []

def save_session(view):
    global last_session_path
    if last_session_path is None:
        save_session_as(view)
    else:
        from . import session
        session.save_session(last_session_path, view)

def save_session_as(view):
    global last_session_path
    dir = '.' if last_session_path is None else last_session_path
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

def open_file(path, fromDatabase = None, set_camera = None):
    from ..ui.gui import main_window as mw
    view = mw.view
    if fromDatabase is None:
        from os.path import expanduser
        p = expanduser(path)
        from os.path import isfile
        if isfile(p):
            open_files([p], view)
        else:
            open_file(p, fromDatabase = 'PDB')
    else:
        ids = path.split(',')
        if set_camera is None:
            set_camera = (len(view.models) == 0)
        from . import fetch
        mlist = []
        for id in ids:
            m = fetch.fetch_from_database(id, fromDatabase)
            if isinstance(m, (list, tuple)):
                mlist.extend(m)
            else:
                mlist.append(m)
        view.add_models(mlist)
        finished_opening([m.path for m in mlist], set_camera, view)
    mw.show_graphics()
