# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.errors import UserError

class LaunchKVFinderTool(ToolInstance):

    #help = "help:user/tools/rotamers.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        structures_widget = QWidget()
        structures_layout = QHBoxLayout()
        structures_widget.setLayout(structures_layout)
        layout.addWidget(structures_widget, alignment=Qt.AlignCenter)
        structures_layout.addWidget(QLabel("Find cavities in:"), alignment=Qt.AlignRight)
        from chimerax.atomic.widgets import AtomicStructureListWidget
        class ShortASLWidget(AtomicStructureListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        self.structures_list = ShortASLWidget(session)
        structures_layout.addWidget(self.structures_list, alignment=Qt.AlignRight)


        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.find_cavities)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        if getattr(self, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def find_cavities(self):
        structures = self.structures_list.value
        if not structures:
            raise UserError("No structures chosen")
        from chimerax.atomic import AtomicStructure
        from chimerax.core.commands import run, concise_model_spec
        run(self.session, "kvfinder %s" % concise_model_spec(self.session, structures,
            relevant_types=AtomicStructure))


_settings = None

class KVFinderResultsDialog(ToolInstance):

    #help = "help:user/tools/rotamers.html"
    SESSION_SAVE = True

    def __init__(self, session, tool_name, *args):
        ToolInstance.__init__(self, session, tool_name)
        if args:
            # being called directly rather than during session restore
            self.finalize_init(*args)

    def finalize_init(self, structure, cavity_group, cavity_models, *, table_state=None):
        self.structure = structure
        self.cavity_group = cavity_group

        from chimerax.ui.widgets import ItemTable
        global _settings
        if _settings is None:
            from chimerax.core.settings import Settings
            class _KVFinderSettings(Settings):
                AUTO_SAVE = {
                    'focus': True
                }
            _settings = _KVFinderSettings(self.session, "KVFinder")

        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            _settings.triggers.add_handler('setting changed', self._action_changed_cb),
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
        ]

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        tw.title = "%s Cavities" % structure.name
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QLabel, QCheckBox, QGroupBox, QWidget, QHBoxLayout, \
            QPushButton, QRadioButton, QButtonGroup, QGridLayout
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        self.table = ItemTable()
        self.table.add_column("Color", "overall_color", format=self.table.COL_FORMAT_TRANSPARENT_COLOR,
            title_display=False)
        self.table.add_column("Cavity Size (voxels)", "num_atoms", format="%d ")
        def color_refresh_cb(trig_name, change_info, *args, table=self.table):
            s, changes = change_info
            if "color changed" in changes.atom_reasons():
                table.update_cell("Color", s)
        for cav_s in cavity_models:
            self.handlers.append(cav_s.triggers.add_handler('changes', color_refresh_cb))

        self.table.data = cavity_models
        self.table.launch(session_info=table_state)
        if not table_state:
            self.table.sortByColumn(1, Qt.DescendingOrder)
        #TODO
        #self.table.selection_changed.connect(self._selection_change)
        layout.addWidget(self.table, alignment=Qt.AlignCenter, stretch=1)

        gbox = QGroupBox("Choosing row in above table will...")
        layout.addWidget(gbox, alignment=Qt.AlignCenter)
        gbox_layout = QVBoxLayout()
        gbox.setLayout(gbox_layout)
        for attr_name, text in [("focus", "Focus view on cavity")]:
            ckbox = QCheckBox(text)
            ckbox.setChecked(getattr(_settings, attr_name))
            ckbox.toggled.connect(lambda checked, *args, settings=_settings, attr_name=attr_name:
                setattr(settings, attr_name, checked))
            gbox_layout.addWidget(ckbox, alignment=Qt.AlignLeft)
        from chimerax.ui.widgets import Citation
        layout.addWidget(Citation(self.session,
            "<b>pyKVFinder: an efficient and integrable Python package for biomolecular<br>cavity detection"
            " and characterization in data science</b><br>"
            "Guerra JVS, Ribeiro-Filho HV, Jara GE, Bortot LO, Pereira JGC, Lopes-de-Oliveira PS",
            prefix="The Find Cavities tool uses the <i>pyKVFinder</i> package.  Please cite:",
            pubmed_id=34930115), alignment=Qt.AlignCenter)

        #TODO?
        #tw.fill_context_menu = self.fill_context_menu
        self.tool_window.manage(placement=None)

    def delete(self, from_mgr=False):
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        act = QAction("Save CSV or TSV File...", parent=menu)
        act.triggered.connect(lambda *args, tab=self.table: tab.write_values())
        menu.addAction(act)

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        if "rot_lib_name" in data:
            lib_name = data['rot_lib_name']
            lib_names = session.rotamers.library_names(installed_only=True)
            ui_name = session.rotamers.ui_name(lib_name)
        else:
            lib_name = ui_name = data['lib_display_name']
            lib_names = session.rotamers.library_names(installed_only=True, for_display=True)
        if lib_name not in lib_names:
            raise RuntimeError("Cannot restore Rotamers tool because %s rotamer library is not installed"
                % ui_name)
        inst.finalize_init(data['mgr'], data['res_type'], session.rotamers.library(lib_name).name,
            table_info=data['table info'])
        return inst

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'mgr': self.mgr,
            'res_type': self.res_type,
            'rot_lib_name': self.rot_lib_name,
            'table info': (self.table.session_info(), [(col_type, c.title, c.data_fetch, c.display_format)
                for col_type, c in self.opt_columns.items()])
        }
        return data

    def _action_changed_cb(self, trig_name, trig_data):
        attr_name, previous, current = trig_data
        #TODO

    def _models_removed_cb(self, trig_name, removed_models):
        if self.structure in removed_models or self.cavity_group in removed_models:
            self.delete()
            return
        old_data = self.table.data
        new_data = [m for m in old_data if m not in removed_models]
        if len(new_data) == 0:
            self.delete()
            return
        if len(new_data) < len(old_data):
            self.table.data = new_data

    def _selection_change(self, selected, deselected):
        if self.table.selected:
            display = set(self.table.selected)
        else:
            display = set(self.mgr.rotamers)
        for rot in self.mgr.rotamers:
            rot.display = rot in display

