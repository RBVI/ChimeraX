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

from .open_save import OpenDialog
class OpenFolderDialog(OpenDialog):
    def __init__(self, parent, session):
        OpenDialog.__init__(self, parent=parent, caption='Open Folder')
        from PyQt5.QtWidgets import QFileDialog
        self.setFileMode(QFileDialog.Directory)

        self._customize_file_dialog(session)

    def _customize_file_dialog(self, session):
        from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QFrame
        options_panel = self.custom_area
        label = QLabel(options_panel)
        label.setText("Format:")
        self._format_selector = selector = QComboBox(options_panel)
        from chimerax.core import io
        fmt_names = [fmt.name for fmt in io.formats()
                     if fmt.has_open_func() and fmt.nicknames and fmt.allow_directory]
        selector.addItems(fmt_names)
        options_layout = QHBoxLayout(options_panel)
        options_layout.addWidget(label)
        options_layout.addWidget(selector)
        options_panel.setLayout(options_layout)
        return options_panel

    @property
    def format_nickname(self):
        fmt_name = self._format_selector.currentText()
        from chimerax.core import io
        for f in io.formats():
            if f.name == fmt_name and f.has_open_func() and f.nicknames and f.allow_directory:
                return f.nicknames[0] 
        return None

    def display(self, session):
        from chimerax.core.filehistory import file_history
        fh = file_history(session)
        hfiles = [f for f in fh.files if f.path and f.database is None]
        if hfiles:
            from os.path import dirname
            initial_dir = dirname(hfiles[-1].path)
            self.setDirectory(initial_dir)
        if not self.exec():
            return
        dirs = self.selectedFiles()
        dir = dirs[0] if len(dirs) > 0 else self.directory().path()
        from chimerax.core.commands import run, quote_if_necessary
        cmd = 'open %s format %s' % (quote_if_necessary(dir), self.format_nickname)
        run(session, cmd)
