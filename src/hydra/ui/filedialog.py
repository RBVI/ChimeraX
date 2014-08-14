from .qt import QtWidgets

# -----------------------------------------------------------------------------
#
def show_open_file_dialog(session):
    '''
    Display the Open file dialog for opening data files.
    '''
    from ..file_io.opensave import file_readers, open_files
    filter_lines = ['%s (%s)' % (r.name, ' '.join('*.%s' % s for s in r.suffixes))
                    for r in file_readers(session)]
    filter_lines.insert(0, 'All (*.*)')
    filters = ';;'.join(filter_lines)
    dir = session.file_history.most_recent_directory()
    if dir is None:
        dir = '.'
    v = session.main_window.view
    qpaths = QtWidgets.QFileDialog.getOpenFileNames(v.widget, 'Open File', dir, filters)
    paths = qpaths[0]    # Return value is a 2-tuple holding list of paths and filter string.
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

# -----------------------------------------------------------------------------
#
def save_session_dialog(session):
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
    from ..file_io import session_file
    session_file.save_session(path, session)
    session.show_info('Saved %s' % path, color = '#000080')

# -----------------------------------------------------------------------------
#
def image_save_dialog(session):
    filters = 'Image (*.jpg *.png *.ppm *.bmp)'
    pf = QtWidgets.QFileDialog.getSaveFileName(session.view.widget, 'Save Image', '.', filters)
    # Returns path and filter name.
    if pf is None:
        return None
    path = pf[0]
    return path

