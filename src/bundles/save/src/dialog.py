# vim: set expandtab shiftwidth=4 softtabstop=4:

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

class MainSaveDialog:
    def __init__(self, settings=None):
        self._settings = settings

    def display(self, session, parent):
        from chimerax.ui.open_save import SaveDialog
        dialog = SaveDialog(parent, "Save File")
        self._customize_dialog(session, dialog)
        if not dialog.exec():
            return
        fmt = self._filter2fmt[dialog.selectedNameFilter()]
        from chimerax.core.commands import run, SaveFileNameArg, StringArg
        fname = self._add_missing_file_suffix(dialog.selectedFiles()[0], fmt)
        cmd = "save2 %s" % SaveFileNameArg.unparse(fname)
        if self._current_option != self._no_options_label:
            cmd += ' ' + session.save.save_arg_string_from_widget(self._current_option)
        run(session, cmd)
        if self._settings:
            self._settings.format_name = fmt.name

    def _add_missing_file_suffix(self, path, fmt):
        import os.path
        ext = os.path.splitext(path)[1]
        if ext not in fmt.suffixes:
            path += fmt.suffixes[0]
        return path

    def _customize_dialog(self, session, dialog):
        options_panel = dialog.custom_area
        saveable_formats = [fmt for fmt in session.save.save_data_formats if fmt.suffixes]
        file_filters = ["%s (%s)" % (fmt.synopsis, "*" + " *".join(fmt.suffixes)) for fmt in saveable_formats]
        self._fmt_name2filter = dict(zip([fmt.name for fmt in saveable_formats], file_filters))
        self._filter2fmt = dict(zip(file_filters, saveable_formats))
        file_filters.sort(key=lambda f: f.lower())
        dialog.setNameFilters(file_filters)
        if self._settings:
            try:
                file_filter = self._fmt_name2filter[self._settings.format_name]
            except KeyError:
                pass
            else:
                dialog.selectNameFilter(file_filter)
        from PyQt5.QtWidgets import QHBoxLayout, QLabel
        self._current_option = self._no_options_label = QLabel("No user-settable options")
        self._options_layout = QHBoxLayout()
        self._options_layout.addWidget(self._no_options_label)
        dialog.custom_area.setLayout(self._options_layout)
        dialog.filterSelected.connect(lambda *args, ses=session, dlg=dialog: self._format_selected(ses, dlg))
        self._format_selected(session, dialog)

    def _format_selected(self, session, dialog):
        fmt = self._filter2fmt[dialog.selectedNameFilter()]
        if self._current_option:
            self._current_option.hide()
        from PyQt5.QtWidgets import QLabel
        self._current_option = session.save.save_args_widget(fmt) or self._no_options_label
        self._options_layout.addWidget(self._current_option)


_settings = None
from chimerax.core.settings import Settings
class SaveDialogSettings(Settings):
    AUTO_SAVE = {
        'format_name': 'ChimeraX session'
    }

def create_dialog(session):
    global _settings
    if not _settings:
        _settings = SaveDialogSettings(session, "main save dialog")
    dlg = MainSaveDialog(settings=_settings)
    session.ui.main_window.add_menu_entry(["File"], "&Save...",
        lambda *args, dlg=dlg, ses=session: dlg.display(ses, ses.ui.main_window),
        tool_tip="Save output file", insertion_point="Close Session")
