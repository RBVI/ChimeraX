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

    help = "help:user/tools/findcavities.html"
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
        self.structures_list = ShortASLWidget(session, autoselect=ShortASLWidget.AUTOSELECT_SINGLE)
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

    help = "help:user/tools/findcavities.html#cavitylist"
    SESSION_SAVE = True

    def __init__(self, session, tool_name, *args):
        ToolInstance.__init__(self, session, tool_name)
        if args:
            # being called directly rather than during session restore
            self.finalize_init(*args)

    def finalize_init(self, structure, cavity_group, cavity_models, probe_radius, *, table_state=None):
        self.structure = structure
        self.cavity_group = cavity_group
        self.probe_radius = probe_radius

        from chimerax.ui.widgets import ItemTable
        global _settings
        if _settings is None:
            from chimerax.core.settings import Settings
            class _KVFinderSettings(Settings):
                AUTO_SAVE = {
                    'focus': True,
                    'nearby': 3.5,
                    'select': False,
                    'show': True,
                    'surface': False,
                    "mlp": False,
                }
            _settings = _KVFinderSettings(self.session, "KVFinder")

        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
            _settings.triggers.add_handler('setting changed', self._setting_changed_cb)
        ]

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QLabel, QCheckBox, QGroupBox, QWidget, QHBoxLayout, \
            QPushButton, QRadioButton, QButtonGroup, QGridLayout, QLineEdit
        from Qt.QtGui import QDoubleValidator
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        layout.setSpacing(0)
        parent.setLayout(layout)
        self.table = ItemTable(session=self.session)
        self.table.add_column("ID", "atomspec", sort_func=lambda cav1, cav2: cav1.id < cav2.id)
        self.table.add_column("Color", "overall_color", format=self.table.COL_FORMAT_TRANSPARENT_COLOR,
            title_display=False, data_set="color {item.atomspec} {value}")
        self.table.add_column("Volume", "kvfinder_volume", format="%g")
        self.table.add_column("Surface Area", "kvfinder_area", format="%g")
        self.table.add_column("Points", "num_atoms", format="%d")
        def color_refresh_cb(trig_name, change_info, *args, table=self.table):
            s, changes = change_info
            if "color changed" in changes.atom_reasons():
                table.update_cell("Color", s)
        for cav_s in cavity_models:
            self.handlers.append(cav_s.triggers.add_handler('changes', color_refresh_cb))

        self.table.data = cavity_models
        self.table.launch(session_info=table_state)
        if not table_state:
            self.table.sortByColumn(2, Qt.DescendingOrder)
        self.table.selection_changed.connect(self._selection_change)
        layout.addWidget(self.table, alignment=Qt.AlignCenter, stretch=1)

        gbox = QGroupBox("Choosing row in above table will...")
        layout.addWidget(gbox, alignment=Qt.AlignCenter)
        gbox_layout = QVBoxLayout()
        gbox_layout.setSpacing(2)
        gbox.setLayout(gbox_layout)
        from chimerax.ui import shrink_font
        self.suboptions = {}
        prev_checked = None
        for attr_name, text, is_sub_option in [
                ("focus", "Focus view on cavity", False),
                ("select", "Select nearby atoms", False),
                ("show", "Show nearby residues", False),
                ("surface", "Surface nearby atoms", False),
                ("mlp", "Color surface by lipophilicity", True),
                ]:
            ckbox = QCheckBox(text)
            checked = getattr(_settings, attr_name)
            ckbox.setChecked(getattr(_settings, attr_name))
            ckbox.toggled.connect(lambda checked, *args, settings=_settings, attr_name=attr_name:
                setattr(settings, attr_name, checked))
            gbox_layout.addWidget(ckbox, alignment=(Qt.AlignCenter if is_sub_option else Qt.AlignLeft))
            if is_sub_option:
                self.suboptions["mlp"] = ckbox
                shrink_font(ckbox)
                ckbox.setEnabled(prev_checked)
            prev_checked = checked

        nearby_widget = QWidget()
        layout.addWidget(nearby_widget, alignment=Qt.AlignCenter)
        nearby_layout = QHBoxLayout()
        nearby_layout.setSpacing(0)
        nearby_layout.setContentsMargins(1,1,1,1)
        nearby_widget.setLayout(nearby_layout)
        nearby_layout.addWidget(QLabel('"Nearby" atoms/residues are within '))
        self.nearby_entry = QLineEdit()
        self.nearby_entry.setMaximumWidth(5 * self.nearby_entry.fontMetrics().averageCharWidth())
        self.nearby_entry.setAlignment(Qt.AlignCenter)
        self.nearby_entry.setText(str(_settings.nearby))
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        self.nearby_entry.setValidator(validator)
        nearby_layout.addWidget(self.nearby_entry)
        nearby_layout.addWidget(QLabel(" angstroms of cavity points"))

        layout.addSpacing(10)

        from chimerax.ui.widgets import Citation
        layout.addWidget(Citation(self.session,
            "<b>pyKVFinder: an efficient and integrable Python package for biomolecular<br>cavity detection"
            " and characterization in data science</b><br>"
            "Guerra JVS, Ribeiro-Filho HV, Jara GE, Bortot LO, Pereira JGC, Lopes-de-Oliveira PS<br>"
            "BMC Bioinformatics (2021) 22:607",
            prefix="The Find Cavities tool uses the <i>pyKVFinder</i> package.  Please cite:",
            pubmed_id=34930115), alignment=Qt.AlignCenter)

        self.tool_window.manage(placement=None)

    def delete(self, from_mgr=False):
        for handler in self.handlers:
            handler.remove()
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst.finalize_init(data['structure'], data['cavity_group'], data['cavity_models'],
            data['probe_radius'], table_state=data['table state'])
        return inst

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'structure': self.structure,
            'cavity_group': self.cavity_group,
            'cavity_models': self.table.data,
            'probe_radius': self.probe_radius,
            'table state': self.table.session_info()
        }
        return data

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

    def _process_settings(self, setting_name=None):
        selected = self.table.selected
        if not selected:
            return

        from chimerax.core.commands import concise_model_spec, run
        model_spec = concise_model_spec(self.session, selected)
        global _settings
        # setting_name is none when the selection changes, so only process the "on" settings in that case...
        if setting_name is None or setting_name == "focus":
            if _settings.focus:
                run(self.session, f"view {model_spec} @< {_settings.nearby} ; zoom 0.75")
            elif setting_name is not None:
                run(self.session, f"view {self.structure.atomspec}")
        if setting_name in ["select", "surface"] \
        or setting_name is None and (_settings.select or _settings.surface):
            if self.nearby_entry.hasAcceptableInput():
                _settings.nearby = float(self.nearby_entry.text())
            else:
                raise UserError('"Nearby" atom/residue distance not valid')
        if setting_name is None or setting_name == "select":
            spec = f"#!{self.structure.id_string} & ({model_spec} @< {_settings.nearby})"
            if _settings.select:
                run(self.session, "select " + spec)
            elif setting_name is not None:
                run(self.session, "~select " + spec)
        if setting_name is None or setting_name == "show":
            spec = f"#!{self.structure.id_string} & ({model_spec} :< {_settings.nearby})"
            if _settings.show:
                run(self.session, "show " + spec)
            elif setting_name is not None:
                run(self.session, "hide " + spec)
        if setting_name is None or setting_name == "surface":
            spec = f"#!{self.structure.id_string} & ({model_spec} @< {_settings.nearby})"
            if _settings.surface:
                probe_arg = "" if self.probe_radius == 1.4 else f" probeRadius {self.probe_radius}"
                surfs = run(self.session, f"surface {spec}{probe_arg} gridSpacing 0.3 visiblePatches 1")
                if surfs and _settings.mlp:
                    run(self.session,
                        f"mlp #!{self.structure.id_string} & @@structure_category=main surfaces %s"
                        % concise_model_spec(self.session, surfs))
            elif setting_name is not None:
                run(self.session, f"~surface {spec}")

        if setting_name == "mlp" and _settings.mlp:
            run(self.session, f"mlp #!{self.structure.id_string} & @@structure_category=main")

    def _selection_change(self, *args):
        self._process_settings()

    def _setting_changed_cb(self, trig_name, trig_data):
        setting_name, prev_val, new_val = trig_data
        self._process_settings(setting_name)
        if setting_name == "surface":
            self.suboptions["mlp"].setEnabled(new_val)
