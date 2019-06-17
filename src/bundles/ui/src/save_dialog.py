# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

class SaveOptionsGUI:
    '''
    This base class is used to register panels of options associated with a
    particular file format for display in the Save File dialog.
    Call the register() method with an instance to register an options gui.
    '''

    def __init__(self, format = None):
        '''If a format (chimerax.core.io.FileFormat) is specified it is used to make default wildcard and save methods.'''
        self._window = None
        self._format = format	

    @property
    def format_name(self):
        '''Override to give format name.'''
        fmt = self._format
        return fmt.name if fmt else ''
    
    def make_ui(self, parent):
        '''Override this and create QFrame containing the GUI of save file options.'''
        return None

    def update(self, session, save_dialog):
        '''Override this to fill in GUI initial values.'''
        pass

    def wildcard(self):
        '''Override this to return a file filter string, e.g. "ChimeraX session files (*.cxs)"'''
        fmt = self._format
        if fmt:
            from .open_save import export_file_filter
            return export_file_filter(format_name=fmt.name)
        return ''

    def save(self, session, filename):
        '''
        Override this to perform the save operation.
        Usually this forms a save command string and executes it.
        '''
        fmt = self._format
        if fmt:
            fmt.export(session, self.add_missing_file_suffix(filename, fmt), fmt.name)

    def add_missing_file_suffix(self, filename, fmt):
        import os.path
        ext = os.path.splitext(filename)[1]
        exts = fmt.extensions
        if exts and ext not in exts:
            filename += exts[0]
        return filename
    
    def window(self, parent):
        if self._window is None:
            self._window = self.make_ui(parent)
        return self._window

    def register(self, session):
        '''Registers this options panel with the Save File dialog.'''
        if hasattr(session, 'ui') and hasattr(session.ui, 'main_window') and hasattr(session.ui.main_window, 'save_dialog'):
            session.ui.main_window.save_dialog.register(self)
        elif session.ui.is_gui:
            # Wait for main window to be created
            session.ui.triggers.add_handler('ready', lambda *args,s=session: self.register(s))
        return 'delete handler'

class MainSaveDialog:

    def __init__(self, default_format = 'ChimeraX session'):
        self._default_format = default_format
        self.file_dialog = None
        self._registered_formats = {}

    def register(self, save_options_gui):
        '''Argument must be an instance of SaveOptionsGUI.'''
        self._registered_formats[save_options_gui.format_name] = save_options_gui
        self._update_format_menu()

    def deregister(self, format_name):
        del self._registered_formats[format_name]

    def display(self, parent, session, format = None,
                initial_directory = None, initial_file = None, model = None):

        self.session = session

        if self.file_dialog is None:
            from .open_save import SaveDialog
            self.file_dialog = fd = SaveDialog(parent, "Save File")
            fd.filterSelected.connect(self._format_selected)
            fd.setLabelText(SaveDialog.FileType, 'Format:')
            self._customize_file_dialog()
            self._format_selected()

        if format is not None:
            fd = self._file_dialog
            if fd:
                fd.setFilter(fmt.wildcard())
        
        fmt = self.current_format()
        if fmt:
            fmt.update(session, self)

        if initial_directory is not None:
            if initial_directory == '':
                from os import getcwd
                initial_directory = getcwd()
            self.file_dialog.setDirectory(initial_directory)
            
        if initial_file is not None:
            self.file_dialog.selectFile(initial_file)

        if model is not None:
            fmt = self.current_format()
            from chimerax.map.savemap import ModelSaveOptionsGUI
            if isinstance(fmt, ModelSaveOptionsGUI):
                fmt.set_model(model)
            
        try:
            if not self.file_dialog.exec():
                return
        finally:
            del self.session
        fmt = self.current_format()
        filename = self.file_dialog.selectedFiles()[0]
        fmt.save(session, filename)

    def current_format(self):
        fd = self.file_dialog
        if fd:
            ftext = fd.selectedNameFilter()
            for fmt in self._registered_formats.values():
                if ftext == fmt.wildcard():
                    return fmt
        return None

    def _format_selected(self, format_wildcard = None):
        fmt = self.current_format()
        if fmt is None:
            return
        w = fmt.window(self._options_panel) or self._no_options_label
        fmt.update(self.session, self)
        if w is self._current_option:
            return
        self._current_option.hide()
        if w not in self._known_options:
            from PyQt5.QtWidgets import QFrame
            self._options_layout.addWidget(w)
            self._known_options.add(w)
        w.show()
        self._current_option = w

    def _update_format_menu(self):
        fd = self.file_dialog
        if fd:
            fmts = self._registered_formats.values()
            filters = [fmt.wildcard() for fmt in fmts]
            filters.sort(key = lambda n: n.lower())
            fd.setNameFilters(filters)

    def _customize_file_dialog(self):
        from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QFrame
        self._options_panel = options_panel = self.file_dialog.custom_area
        self._no_options_label = no_opt_label = QLabel(options_panel)
        no_opt_label.setText("No user-settable options")
        self._known_options = set([no_opt_label])
        self._current_option = no_opt_label
        self._update_format_menu()
        fmt = self._registered_formats.get(self._default_format)
        if fmt:
            fd = self.file_dialog
            fd.selectNameFilter(fmt.wildcard())
        self._options_layout = options_layout = QHBoxLayout(options_panel)
        options_layout.addWidget(no_opt_label)
        options_panel.setLayout(options_layout)
        return options_panel

def register_save_dialog_options(save_dialog):
    from chimerax.core.io import formats
    for fmt in formats(open=False):
        save_dialog.register(SaveOptionsGUI(fmt))

    # Register session and image save gui options here instead of in core
    # because ui does not exist when core registers these file formats.
    from chimerax.core.session import register_session_save_options_gui
    register_session_save_options_gui(save_dialog)

    from chimerax.core.image import register_image_save_options_gui
    register_image_save_options_gui(save_dialog)
