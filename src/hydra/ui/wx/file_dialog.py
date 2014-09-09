# vim: set expandtab ts=4 sw=4:

def save_session_dialog(session):
    '''
    Save a session file, raising a dialog to enter the file path.
    '''

    dir = session.last_session_path
    if dir is None:
        dir = session.file_history.most_recent_directory()
    filters = 'Session (*.hy)'
    import wx
    dlg = wx.FileDialog(session.view, "Save Session", dir, wildcard=filters,
        style=wx.FDSAVE|wx.FD_OVERWRITE_PROMPT)
    if dlg.ShowModal() == wx.ID_CANCEL:
        return
    path = dlg.GetPath()
    from ...files import session_file
    session_file.save_session(path, session)
    session.show_info('Saved %s' % path, color = '#000080')

def show_open_file_dialog(session):
    '''
    Display the Open file dialog for opening data files.
    '''
    from ...files.opensave import file_readers, open_files
    filter_lines = ['%s (%s)' % (r.name, ' '.join('*.%s' % s
        for s in r.suffixes)) for r in file_readers(session)]
    filter_lines.insert(0, 'All (*.*)')
    filters = '|'.join(filter_lines)
    dir = session.file_history.most_recent_directory()
    if dir is None:
        dir = '.'
    import wx
    dlg = wx.FileDialog(session.view, "Open File", dir, wildcard=filters,
        style=wx.FDOPEN|wx.FD_FILE_MUST_EXIST|wx.FD_MULTIPLE|wx.FD_PREVIEW)

    if dlg.ShowModal() == wx.ID_CANCEL:
        return
    paths = dlg.GetPaths()
    mlist = open_files(paths, session)
    if mlist:
        session.file_history.add_entry(','.join(paths), models = mlist)
    session.main_window.show_graphics()
