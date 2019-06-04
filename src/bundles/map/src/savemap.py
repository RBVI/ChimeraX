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

class ModelSaveOptionsGUI:

    def __init__(self, session, format, model_class, menu_label):
        self._session = session
        self._format = format			# FileFormat from chimerax.core.io
        self._model_class = model_class		# e.g. Volume
        self._menu_label = menu_label		# Menu label
        
    def make_ui(self, parent):
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel
        
        mf = QFrame(parent)
        mlayout = QHBoxLayout(mf)
        mlayout.setContentsMargins(0,0,0,0)
        mlayout.setSpacing(10)
        
        sm = QLabel(self._menu_label, mf)
        mlayout.addWidget(sm)
        from chimerax.ui.widgets import ModelMenuButton
        self._map_menu = mm = ModelMenuButton(self._session, class_filter = self._model_class)
        mlayout.addWidget(mm)
        mlayout.addStretch(1)    # Extra space at end

        return mf

    def save(self, session, filename):
        path = self._add_file_suffix(filename)
        from chimerax.core.commands import run, quote_if_necessary
        cmd = 'save %s model #%s' % (quote_if_necessary(path),
                                     self._map_menu.value.id_string)
        run(session, cmd)

    def _add_file_suffix(self, filename):
        suffixes = self._format.extensions
        for suffix in suffixes:
            if filename.endswith(suffix):
                return filename
        return filename + suffixes[0]

    def update(self, session, save_dialog):
        m = None
        mlist = session.models.list(type = self._model_class)
        msel = [m for m in mlist if m.selected]
        if msel:
            m = msel[0]
        else:
            mdisp = [m for m in mlist if m.visible]
            if mdisp:
                m = mdisp[0]
            elif mlist:
                m = mlist[0]
        self._map_menu.value = m

    def wildcard(self):
        f = self._format
        wildcard = '%s files (%s)' % (f.name,
                                      ','.join('*%s' % suffix for suffix in f.extensions))
        return wildcard

def register_map_save_options(session):
    from chimerax.map.data.fileformats import file_formats
    from chimerax.core.io import format_from_name
    formats = [format_from_name(fmt.description) for fmt in file_formats if fmt.writable]
    from chimerax.map import Volume
    sd = session.ui.main_window.save_dialog
    for format in formats:
        o = ModelSaveOptionsGUI(session, format, Volume, 'Map')
        sd.register(format.name, o.wildcard, o.make_ui, o.update, o.save)
    return 'delete handler'
