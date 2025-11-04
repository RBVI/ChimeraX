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

_launch_settings = None

class LaunchKVFinderTool(ToolInstance):

    help = "help:user/tools/findcavities.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        # do the below here instead of at class or global scope so that importing pyKVFinder doesn't
        # happen during the import tests
        import inspect
        from .cmd import cmd_kvfinder
        cmd_params = inspect.signature(cmd_kvfinder).parameters
        self.cmd_defaults = {
            kw_name: cmd_params[kw_name].default
                for kw_name in ['grid_spacing', 'probe_in', 'probe_out', 'removal_distance', 'volume_cutoff']
        }
        global _launch_settings
        if _launch_settings is None:
            from chimerax.core.settings import Settings
            class _KVFinderLaunchSettings(Settings):
                EXPLICIT_SAVE = self.cmd_defaults
            _launch_settings = _KVFinderLaunchSettings(self.session, "KVFinder launch")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget, QGroupBox, QCheckBox
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


        group = QGroupBox("Cavity detection settings")
        layout.addWidget(group, alignment=Qt.AlignTop|Qt.AlignHCenter)
        group_layout = QHBoxLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group.setLayout(group_layout)
        from chimerax.ui.options import SettingsPanel, FloatOption
        self.options_panel = panel = SettingsPanel(sorting=False, scrolled=False)
        group_layout.addWidget(panel)
        tool_tips = {
            'probe_in':
                "A smaller probe that defines the biomolecular surface by rolling around\n"
                " the target biomolecule. Typically, this is set to the size of a water\n"
                " molecule (1.4 Å).",
            'probe_out':
                "A larger probe that defines inacessibility region, i.e., the cavities,\n"
                " and by rolling around the target biomolecule. Users can adjust the size\n"
                " of the probe based on the characteristics of the target structure.",
            'removal_distance':
                "A length that is removed from the boundary between the cavity and bulk\n"
                " (solvent) region.",
            'volume_cutoff':
                "A cavity volume filter to exclude cavities with smaller volumes than this\n"
                " limit. These smaller cavities are typically not relevant for function."
        }
        # some of the min/max values are there to make the entry areas less wide
        for label, attr_name, kw in [
                ("Grid spacing", 'grid_spacing', {'min': 'positive'}),
                ("Inner probe radius", 'probe_in', {'min': 'positive'}),
                ("Outer probe radius", 'probe_out', {'min': 'positive'}),
                ("Exterior trim distance", 'removal_distance', { 'min': -999.9}),
                ("Minimum cavity volume", 'volume_cutoff', {'min': 0.0})]:
            opt = FloatOption(label, getattr(_launch_settings, attr_name), None, decimal_places=2,
                balloon=tool_tips.get(attr_name, None), attr_name=attr_name, settings=_launch_settings,
                    max=1000.0, **kw)
            setattr(self, attr_name + '_option', opt)
            panel.add_option(opt)

        self.replace_prev = QCheckBox("Replace existing results, if any")
        self.replace_prev.setChecked(True)
        layout.addWidget(self.replace_prev, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.find_cavities)
        # Since ApplyRole is not AcceptRole, simply connecting to the Apply button won't dismiss the dialog
        bbox.button(qbbox.Apply).clicked.connect(self.find_cavities)
        bbox.rejected.connect(self.delete)
        if getattr(self, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def find_cavities(self):
        self.tool_window.shown = False
        self.session.ui.processEvents()
        from chimerax.ui import tool_user_error
        structures = self.structures_list.value
        if not structures:
            self.tool_window.shown = True
            return tool_user_error("No structures chosen")
        from chimerax.atomic import AtomicStructure
        from chimerax.core.commands import run, concise_model_spec
        cmd = "kvfinder %s" % concise_model_spec(self.session, structures, relevant_types=AtomicStructure)
        from chimerax.core.commands import camel_case
        global _launch_settings
        for attr_name, default_value in self.cmd_defaults.items():
            cur_val = getattr(_launch_settings, attr_name)
            if attr_name == "grid_spacing" and (cur_val <= 0.0 or cur_val >= 5.0):
                self.tool_window.shown = True
                return tool_user_error("Grid spacing value must be > 0.0 and < 5.0")
            if cur_val != default_value:
                cmd += " " + camel_case(attr_name) + " %g" % cur_val
        if not self.replace_prev.isChecked():
            cmd += " replace false"
        run(self.session, cmd)
        self.delete()


_results_settings = None

class KVFinderResultsDialog(ToolInstance):

    help = "help:user/tools/findcavities.html#cavitylist"
    SESSION_SAVE = True

    def __init__(self, session, tool_name, *args, **kw):
        ToolInstance.__init__(self, session, tool_name)
        if args:
            # being called directly rather than during session restore
            self.finalize_init(*args, False, **kw)

    def finalize_init(self, structure, cavity_group, cavity_models, probe_radius, surface_made,
            *, table_state=None, placement=None, contacting_info=None):
        self.structure = structure
        self.cavity_group = cavity_group
        self.probe_radius = probe_radius
        self._surface_made = surface_made
        if contacting_info is None:
            self.contacting = self.non_backbone_contacting = None
        else:
            self.contacting, self.non_backbone_contacting = contacting_info

        from chimerax.ui.widgets import ItemTable
        global _results_settings
        if _results_settings is None:
            from chimerax.core.settings import Settings
            class _KVFinderResultsSettings(Settings):
                AUTO_SAVE = {
                    'focus': True,
                    'nearby': 3.5,
                    'select_atoms': False,
                    'select_cavity': False,
                    'show': True,
                    'show_contacting': True,
                    'backbone_contacting': True,
                    'surface': True,
                }
            _results_settings = _KVFinderResultsSettings(self.session, "KVFinder results")

        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
            _results_settings.triggers.add_handler('setting changed', self._setting_changed_cb)
        ]
        for cav_s in cavity_models:
            self.handlers.append(cav_s.triggers.add_handler('changes', self._structure_change_cb))

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
        def color_func(s):
            if s.display and s.atoms.displays.any():
                return s.overall_color
            for srf in s.surfaces():
                if srf.display:
                    return srf.overall_color
            return s.overall_color
        self.color_column = self.table.add_column("Color", color_func,
            format=self.table.COL_FORMAT_TRANSPARENT_COLOR,
            title_display=False, data_set="color {item.atomspec} {value}")
        self.table.add_column("Volume", "kvfinder_volume", format="%g")
        self.table.add_column("Surface Area", "kvfinder_area", format="%g")
        self.table.add_column("Points", "num_atoms", format="%d")
        self.table.add_column("Average Depth", "kvfinder_average_depth", format="%g")
        self.table.add_column("Maximum Depth", "kvfinder_max_depth", format="%g")
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
        option_info = [
            ("focus", "Focus view on cavity", False),
            ("surface", "Show cavity surface", False),
            ("select_cavity", "Select cavity points", False),
            ("select_atoms", "Select nearby atoms", False)
        ]
        if self.contacting is None:
            option_info.append(("show", "Show nearby residues", False))
        else:
            option_info.extend([
                ("show_contacting", "Show contacting residues", False),
                ("backbone_contacting", "Include backbone contacts", True)
            ])
        for attr_name, text, is_sub_option in option_info:
            ckbox = QCheckBox(text)
            checked = getattr(_results_settings, attr_name)
            ckbox.setChecked(getattr(_results_settings, attr_name))
            ckbox.toggled.connect(lambda checked, *args, settings=_results_settings, attr_name=attr_name:
                setattr(settings, attr_name, checked))
            gbox_layout.addWidget(ckbox, alignment=(Qt.AlignCenter if is_sub_option else Qt.AlignLeft))
            if is_sub_option:
                self.suboptions[attr_name] = ckbox
                shrink_font(ckbox)
                ckbox.setEnabled(prev_checked)
            prev_checked = checked

        nearby_widget = QWidget()
        layout.addWidget(nearby_widget, alignment=Qt.AlignCenter)
        nearby_layout = QHBoxLayout()
        nearby_layout.setSpacing(0)
        nearby_layout.setContentsMargins(1,1,1,1)
        nearby_widget.setLayout(nearby_layout)
        nearby_layout.addWidget(
            QLabel('"Nearby" atoms%s are within ' % ("/residues" if self.contacting is None else "")))
        self.nearby_entry = QLineEdit()
        self.nearby_entry.setMaximumWidth(5 * self.nearby_entry.fontMetrics().averageCharWidth())
        self.nearby_entry.setAlignment(Qt.AlignCenter)
        self.nearby_entry.setText(str(_results_settings.nearby))
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        self.nearby_entry.setValidator(validator)
        nearby_layout.addWidget(self.nearby_entry)
        nearby_layout.addWidget(QLabel(" angstroms of cavity points"))

        depth_widget = QWidget()
        layout.addWidget(depth_widget, alignment=Qt.AlignCenter)
        depth_layout = QHBoxLayout()
        depth_layout.setSpacing(0)
        depth_widget.setLayout(depth_layout)
        depth_button = QPushButton("Color")
        depth_button.clicked.connect(self.color_by_depth)
        depth_layout.addWidget(depth_button)
        depth_layout.addWidget(QLabel(" open cavities by depth from outside"))

        layout.addSpacing(10)

        from chimerax.ui.widgets import Citation
        layout.addWidget(Citation(self.session,
            "<b>pyKVFinder: an efficient and integrable Python package for biomolecular<br>cavity detection"
            " and characterization in data science</b><br>"
            "Guerra JVS, Ribeiro-Filho HV, Jara GE, Bortot LO, Pereira JGC, Lopes-de-Oliveira PS<br>"
            "BMC Bioinformatics (2021) 22:607",
            prefix="The Find Cavities tool uses the <i>pyKVFinder</i> package.  Please cite:",
            pubmed_id=34930115), alignment=Qt.AlignCenter)

        self.tool_window.manage(placement=placement)

    def delete(self, from_mgr=False):
        for handler in self.handlers:
            handler.remove()
        # on Windows only (e.g. #18297), the dead table can try to fetch info from dead models
        # despite all the handlers being removed that would drive that; remove the table data
        self.table.data = []
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst.finalize_init(data['structure'], data['cavity_group'], data['cavity_models'],
            data['probe_radius'], data.get('surface_made', False), table_state=data['table state'],
            contacting_info=data.get('contacting_info', None))
        return inst

    def color_by_depth(self):
        open_cavities = [c for c in self.table.data if c.kvfinder_max_depth > 0]
        if not open_cavities:
            self.session.logger.warning("All cavities are completely enclosed")
            return

        from chimerax.core.commands import concise_model_spec, run
        model_spec = concise_model_spec(self.session, open_cavities)
        run(self.session, f"color byattr a:kvfinder_depth {model_spec}")

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'structure': self.structure,
            'cavity_group': self.cavity_group,
            'cavity_models': self.table.data,
            'probe_radius': self.probe_radius,
            'surface_made': self._surface_made,
            'table state': self.table.session_info(),
            'contacting_info': (self.contacting, self.non_backbone_contacting)
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
        global _results_settings
        # setting_name is none when the selection changes, so only process the "on" settings in that case...
        if setting_name is None or setting_name == "focus":
            if _results_settings.focus:
                run(self.session, f"view {model_spec} @< {_results_settings.nearby} ; zoom 0.75")
            elif setting_name is not None:
                run(self.session, f"view {self.structure.atomspec}")
        if setting_name is None or setting_name == "surface":
            if self._surface_made:
                run(self.session, f"~surface {self.cavity_group.atomspec}")
            if _results_settings.surface:
                probe_arg = "" if self.probe_radius == 1.4 else f" probeRadius {self.probe_radius}"
                if not self._surface_made:
                    from chimerax.atomic import surfaces_with_atoms
                    unsurfaced = set()
                    for cavity in self.table.data:
                        if not surfaces_with_atoms(cavity.atoms):
                            unsurfaced.add(cavity)
                    unsurfaced_spec = concise_model_spec(self.session, unsurfaced)
                    run(self.session, f"surface {unsurfaced_spec}{probe_arg} trans 50")
                    multicolor = False
                    for cavity in unsurfaced:
                        unique_colors = set([tuple(x) for x in cavity.atoms.colors])
                        if len(unique_colors) > 1:
                            multicolor = True
                            break
                    if multicolor:
                        run(self.session, f"color {unsurfaced_spec} fromatoms trans 50")
                    run(self.session, f"~surface {self.cavity_group.atomspec}")
                    self._surface_made = True
                run(self.session, f"surface {model_spec}{probe_arg}")
        if setting_name in ["select_atoms", "show"] or setting_name is None \
        and (_results_settings.select_atoms or _results_settings.show):
            if self.nearby_entry.hasAcceptableInput():
                _results_settings.nearby = float(self.nearby_entry.text())
            else:
                raise UserError('"Nearby" atom/residue distance not valid')
        if setting_name is None or setting_name == "select_atoms":
            spec = f"#!{self.structure.id_string} & ({model_spec} @< {_results_settings.nearby})"
            if _results_settings.select_atoms:
                run(self.session, "select " + spec)
            elif setting_name is not None:
                run(self.session, "~select " + spec)
        if setting_name is None or setting_name == "select_cavity":
            if _results_settings.select_cavity:
                run(self.session, "select " + model_spec)
            elif setting_name is not None:
                run(self.session, "~select " + model_spec)
        if self.contacting is None:
            if setting_name is None or setting_name == "show":
                spec = f"#!{self.structure.id_string} & ({model_spec} :< {_results_settings.nearby})"
                if _results_settings.show:
                    run(self.session, "show " + spec)
                elif setting_name is not None:
                    run(self.session, "hide " + spec)
        else:
            from chimerax.atomic import concise_residue_spec, concatenate
            if setting_name is None or setting_name == "show_contacting":
                if _results_settings.backbone_contacting:
                    residues = self.contacting
                else:
                    residues = self.non_backbone_contacting
                spec = concise_residue_spec(self.session, concatenate([residues[cav] for cav in selected]))
                if _results_settings.show_contacting:
                    run(self.session, "show " + spec)
                elif setting_name is not None:
                    run(self.session, "hide " + spec)
            if setting_name == "backbone_contacting":
                if _results_settings.show_contacting:
                    all_contacts = concatenate([self.contacting[cav] for cav in selected],
                        remove_duplicates=True)
                    non_bb_contacts = concatenate([self.non_backbone_contacting[cav] for cav in selected],
                        remove_duplicates=True)
                    bb_contacts = all_contacts - non_bb_contacts
                    if bb_contacts:
                        spec = concise_residue_spec(self.session, bb_contacts)
                        if _results_settings.backbone_contacting:
                            run(self.session, "show " + spec)
                        else:
                            run(self.session, "hide " + spec)

    def _selection_change(self, *args):
        self._process_settings()

    def _setting_changed_cb(self, trig_name, trig_data):
        setting_name, prev_val, new_val = trig_data
        self._process_settings(setting_name)

    def _structure_change_cb(self, trig_name, changes):
        if 'display changed' in changes[1].structure_reasons():
            self.table.update_column(self.color_column, data=True)
