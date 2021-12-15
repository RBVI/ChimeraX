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

from chimerax.core.tools import ToolInstance
from chimerax.seqalign.widgets import AlignmentListWidget, AlignSeqMenuButton

class ModellerLauncher(ToolInstance):
    """Generate the inputs needed by Modeller for comparative modeling"""

    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from .settings import get_settings
        self.settings = settings = get_settings(session, tool_name)
        self.help = "help:user/tools/modeller.html" if hasattr(settings, "multichain") \
            else "help:user/tools/modelloops.html"
        self.common_settings = common_settings = get_settings(session, "license")
        from Qt.QtWidgets import QFormLayout, QAbstractItemView, QGroupBox, QVBoxLayout
        from Qt.QtWidgets import QDialogButtonBox as qbbox
        from Qt.QtCore import Qt
        interface_layout = QVBoxLayout()
        interface_layout.setContentsMargins(0, 0, 0, 0)
        interface_layout.setSpacing(0)
        parent.setLayout(interface_layout)
        alignments_area = QGroupBox("Sequence alignments")
        interface_layout.addWidget(alignments_area)
        interface_layout.setStretchFactor(alignments_area, 1)
        alignments_layout = QVBoxLayout()
        alignments_layout.setContentsMargins(0, 0, 0, 0)
        alignments_area.setLayout(alignments_layout)
        self.alignment_list = AlignmentListWidget(session)
        self.alignment_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.alignment_list.keyPressEvent = session.ui.forward_keystroke
        self.alignment_list.value_changed.connect(self._list_selection_cb)
        self.alignment_list.alignments_changed.connect(self._update_sequence_menus)
        alignments_layout.addWidget(self.alignment_list)
        alignments_layout.setStretchFactor(self.alignment_list, 1)
        targets_area = QGroupBox("Target sequences")
        self.targets_layout = QFormLayout()
        targets_area.setLayout(self.targets_layout)
        interface_layout.addWidget(targets_area)
        self.seq_menu = {}
        self._update_sequence_menus(session.alignments.alignments)
        options_area = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_area.setLayout(options_layout)
        interface_layout.addWidget(options_area)
        interface_layout.setStretchFactor(options_area, 2)
        from chimerax.ui.options import CategorizedSettingsPanel, BooleanOption, IntOption, PasswordOption,\
            OutputFolderOption, SymbolicEnumOption, EnumOption, InputFileOption
        panel = CategorizedSettingsPanel(category_sorting=False, option_sorting=False, buttons=False)
        options_layout.addWidget(panel)
        if hasattr(settings, "multichain"):
            panel.add_option(
                "Basic"
                , BooleanOption(
                    "Make multichain model from multichain template",
                    settings.multichain, None, balloon=
                    "If false, all chains (templates) associated with an alignment will be used in\n"
                    "combination to model the target sequence of that alignment, i.e. a monomer will be\n"
                    "generated from the alignment.  If true, the target sequence will be modeled from each\n"
                    "template, i.e. a multimer will be generated from the alignment (assuming multiple chains\n"
                    "are associated).", attr_name="multichain", settings=settings
                )
            )
        max_models = 1000
        panel.add_option(
            "Basic"
            , IntOption(
                "Number of models", settings.num_models, None,
                attr_name="num_models", settings=settings, min=1, max=max_models, balloon=
                "Number of model structures to generate.  Must be no more than %d.\n"
                "Warning: please consider the calculation time" % max_models
            )
        )
        if hasattr(settings, "region"):
            from .loops import ALL_MISSING, INTERNAL_MISSING
            class RegionOption(SymbolicEnumOption):
                values = (ALL_MISSING, INTERNAL_MISSING, "active")
                labels = (
                    "all missing structure",
                    "internal missing structure",
                    "active sequence-viewer region"
                )
            panel.add_option(
                "Basic"
                , RegionOption(
                    "Model", settings.region, None, attr_name="region",
                    settings=settings, balloon="Parts of the structure(s) to remodel/refine"
                )
            )
        if hasattr(settings, "adjacent_flexible"):
            panel.add_option(
                "Basic"
                , IntOption(
                    "Adjacent flexible residues", settings.adjacent_flexible,
                    None, attr_name="adjacent_flexible", settings=settings, min=0, max=100,
                    balloon="Number of residues adjacent to explicitly modeled region to also treat as flexible\n"
                    "(i.e. remodel as needed)."
                )
            )
        class ExecutionTypeOption(SymbolicEnumOption):
            values = (False, True)
            labels = ("web service", "local machine")
        execution_option = ExecutionTypeOption(
            "Computation location", common_settings.local_execution,
            self.show_execution_options, attr_name="local_execution", settings=common_settings,
            balloon="Run computation using RBVI web service or on the local machine"
        )
        panel.add_option("Basic", execution_option)
        self.web_container, web_options = panel.add_option_group(
            "Basic", group_label="Web execution parameters"
        )
        layout = QVBoxLayout()
        self.web_container.setLayout(layout)
        layout.addWidget(web_options, alignment=Qt.AlignLeft)
        key = "" if common_settings.license_key is None else common_settings.license_key
        password_opt = PasswordOption(
            '<a href="https://www.salilab.org/modeller/registration.html">Modeller'
            ' license key</a>', key, None, attr_name="license_key", settings=common_settings, balloon=
            "Your Modeller license key.  You can obtain a license key by registering at the Modeller web"
            " site"
        )
        password_opt.widget.setMinimumWidth(120)
        web_options.add_option(password_opt)
        self.local_container, local_options = panel.add_option_group(
            "Basic", group_label="Local execution parameters"
        )
        layout = QVBoxLayout()
        self.local_container.setLayout(layout)
        layout.addWidget(local_options, alignment=Qt.AlignLeft)
        import sys
        if sys.platform == 'win32':
            balloon_add = ".\nThe executable is typically located within a subfolder of the 'lib'\nfolder of your Modeller installation."
        else:
            balloon_add = ""
        local_options.add_option(
            InputFileOption(
                "Executable location", common_settings.executable_path,
                None, attr_name="executable_path", settings=common_settings, balloon="Full path to Modeller"
                " executable" + balloon_add
            )
        )
        self.show_execution_options(execution_option)
        panel.add_option(
            "Advanced"
            , BooleanOption(
                "Use fast/approximate mode",
                settings.fast, None, attr_name="fast", settings=settings, balloon=
                "If enabled, use a fast approximate method to generate models.\n"
                "Typically used to get a rough idea what the models will look like or\n"
                "to check that the alignment is reasonable."
            )
        )
        if hasattr(settings, "het_preserve"):
            panel.add_option(
                "Advanced"
                , BooleanOption(
                    "Include non-water HETATM residues from template",
                    settings.het_preserve, None, attr_name="het_preserve", settings=settings, balloon=
                    "If enabled, all non-water HETATM residues in the template\n"
                    "structure(s) will be transferred into the generated models."
                )
            )
        if hasattr(settings, "hydrogens"):
            panel.add_option(
                "Advanced"
                , BooleanOption(
                    "Build models with hydrogens",
                    settings.hydrogens, None, attr_name="hydrogens", settings=settings, balloon=
                    "If enabled, the generated models will include hydrogen atoms.\n"
                    "Otherwise, only heavy atom coordinates will be built.\n"
                    "Increases computation time by approximately a factor of 4."
                )
            )
        if hasattr(settings, "protocol"):
            from .loops import protocols
            class ProtocolOption(EnumOption):
                values = protocols
            panel.add_option(
                "Advanced"
                , ProtocolOption(
                    "Protocol", settings.protocol, None,
                    attr_name="protocol", settings=settings,
                    balloon="Protocol to use to compute modeling"
                )
            )
        panel.add_option(
            "Advanced"
            , OutputFolderOption(
                "Temporary folder location (optional)",
                settings.temp_path, None, attr_name="temp_path", settings=settings, balloon=
                "Specify a folder for temporary files.  If not specified,\n"
                "a location will be generated automatically."
            )
        )
        if hasattr(settings, "water_preserve"):
            panel.add_option(
                "Advanced"
                , BooleanOption(
                    "Include water molecules from template",
                    settings.water_preserve, None, attr_name="water_preserve", settings=settings, balloon=
                    "If enabled, all water molecules in the template\n"
                    "structure(s) will be included in the generated models."
                )
            )
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import Citation
        interface_layout.addWidget(
            Citation(
                session,
                "A. Sali and T.L. Blundell.\n"
                "Comparative protein modelling by satisfaction of spatial restraints.\n"
                "J. Mol. Biol. 234, 779-815, 1993.",
                prefix="Publications using Modeller results should cite:", pubmed_id=18428767
            ),
            alignment=Qt.AlignCenter
        )
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.launch_modeller)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        interface_layout.addWidget(bbox)
        self.tool_window.manage(None)

    def delete(self):
        ToolInstance.delete(self)

    def launch_modeller(self):
        from chimerax.core.commands import run, FileNameArg, StringArg
        from chimerax.core.errors import UserError
        alignments = self.alignment_list.value
        if not alignments:
            raise UserError("No alignments chosen for modeling")
        aln_seq_args = []
        for aln in alignments:
            seq_menu = self.seq_menu[aln]
            seq = seq_menu.value
            if not seq:
                raise UserError("No target sequence chosen for alignment %s" % aln.ident)
            aln_seq_arg = "%s:%d" % (aln.ident, aln.seqs.index(seq) + 1)
            if hasattr(self.settings, "region"):
                from .loops import ALL_MISSING, INTERNAL_MISSING
                if self.settings.region in (ALL_MISSING, INTERNAL_MISSING):
                    reg_arg = self.settings.region
                else:
                    active_region = None
                    for viewer in aln.viewers:
                        r = getattr(viewer, 'active_region', None)
                        if r is not None:
                            if active_region is not None:
                                raise UserError("Multiple active regions for alignment %s" % aln)
                            active_region = r
                    if active_region is None:
                        raise UserError("No active regions for alignment %s" % aln)
                    line1, line2, *args = active_region.blocks[0]
                    seq_index = aln.seqs.index(seq)
                    if aln.seqs.index(line1) > seq_index or aln.seqs.index(line2) < seq_index:
                        raise UserError("Active region for alignment %s does not include target sequence"
                                        % aln)
                    sub_ranges = []
                    for l1, l2, start, end in active_region.blocks:
                        if start == end:
                            sub_ranges.append(str(start + 1))
                        else:
                            sub_ranges.append("%d-%d" % (start + 1, end + 1))
                    reg_arg = ','.join(sub_ranges)
                aln_seq_arg += ':' + reg_arg
            aln_seq_args.append(StringArg.unparse(aln_seq_arg))
        if hasattr(self.settings, "region"):
            sub_cmd = "refine"
            specific_args = "adjacentFlexible %d protocol %s" % (self.settings.adjacent_flexible,
                                                                 StringArg.unparse(self.settings.protocol))
        else:
            sub_cmd = "comparative"
            specific_args = "multichain %s hetPreserve %s hydrogens %s waterPreserve %s" % (
                repr(self.settings.multichain).lower(),
                repr(self.settings.het_preserve).lower(),
                repr(self.settings.hydrogens).lower(),
                repr(self.settings.water_preserve).lower())
        if self.common_settings.local_execution:
            specific_args += " executableLocation %s" % StringArg.unparse(
                self.common_settings.executable_path)
        run(
            self.session
            , ("modeller %s %s numModels %d fast %s " % (sub_cmd, " ".join(aln_seq_args),
               self.settings.num_models, repr(self.settings.fast).lower()) + specific_args + (" tempPath %s"
               % FileNameArg.unparse(self.settings.temp_path) if self.settings.temp_path else ""))
        )
        self.delete()

    def show_execution_options(self, opt):
        if opt.value:
            self.local_container.show()
            self.web_container.hide()
        else:
            self.web_container.show()
            self.local_container.hide()

    def _list_selection_cb(self):
        layout = self.targets_layout
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                break
            widget = item.widget()
            if isinstance(widget, AlignSeqMenuButton):
                widget.setHidden(True)
            else:
                widget.deleteLater()
        for sel_aln in self.alignment_list.value:
            mb = self.seq_menu[sel_aln]
            mb.setHidden(False)
            layout.addRow(sel_aln.ident, mb)

    def _update_sequence_menus(self, alignments):
        alignment_set = set(alignments)
        for aln, mb in list(self.seq_menu.items()):
            if aln not in alignment_set:
                row, role = self.targets_layout.getWidgetPosition(mb)
                if row >= 0:
                    self.targets_layout.removeRow(row)
                del self.seq_menu[aln]
        for aln in alignments:
            if aln not in self.seq_menu:
                self.seq_menu[aln] = AlignSeqMenuButton(aln, no_value_button_text="No target chosen")

class ModellerResultsViewer(ToolInstance):
    """ Viewer displays the models/results generated by Modeller"""

    help = "help:user/tools/modeller.html#output"

    def __init__(self, session, tool_name, models=None, attr_names=None):
        """ if 'models' is None, then we are being restored from a session and
            set_state_from_snapshot will be called later.
        """

        ToolInstance.__init__(self, session, tool_name)
        if models is None:
            return
        self._finalize_init(session, models, attr_names)

    def _finalize_init(self, session, models, attr_names, scores_fetched=False):
        self.models = models
        self.attr_names = attr_names
        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb)]

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        self.tool_window.fill_context_menu = self.fill_context_menu
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QTableWidget, QVBoxLayout, QAbstractItemView
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.keyPressEvent = session.ui.forward_keystroke
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._table_selection_cb)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)
        layout.setStretchFactor(self.table, 1)
        self._fill_table()
        self.tool_window.manage('side')
        self.scores_fetched = scores_fetched
        for m in models:
            self.handlers.append(m.triggers.add_handler("changes", self._changes_cb))

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.row_item_lookup = {}
        ToolInstance.delete(self)

    def fetch_additional_scores(self, refresh=False):
        self.scores_fetched = True
        from chimerax.core.commands import run, concise_model_spec
        run(self.session, "modeller scores %s refresh %s" % (
            concise_model_spec(self.session, self.models), str(refresh).lower()))

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        if self.scores_fetched:
            refresh_action = QAction("Refresh Scores", menu)
            refresh_action.triggered.connect(lambda arg: self.fetch_additional_scores(refresh=True))
            menu.addAction(refresh_action)
        else:
            fetch_action = QAction("Fetch Additional Scores", menu)
            fetch_action.triggered.connect(lambda arg: self.fetch_additional_scores())
            menu.addAction(fetch_action)

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(session, data['models'], data['attr_names'], data['scores_fetched'])
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'models': self.models,
            'attr_names': self.attr_names,
            'scores_fetched': self.scores_fetched
        }
        return data

    def _changes_cb(self, trig_name, trig_data):
        structure, changes_obj = trig_data
        need_update = False
        if changes_obj.modified_structures():
            for reason in changes_obj.structure_reasons():
                if reason.startswith("modeller_") and reason.endswith(" changed"):
                    need_update = True
                    attr_name = reason[:-8]
                    if attr_name not in self.attr_names:
                        self.attr_names.append(attr_name)
        if need_update:
            # atomic changes are already collated, so don't need to delay table update
            self._fill_table()

    def _fill_table(self, *args):
        self.table.clear()
        self.table.setColumnCount(len(self.attr_names) + 1)
        self.table.setHorizontalHeaderLabels(
            ["Model"] + [attr_name[9:].replace('_', ' ') for attr_name in self.attr_names]
        )
        self.table.setRowCount(len(self.models))
        from Qt.QtWidgets import QTableWidgetItem
        for row, m in enumerate(self.models):
            item = QTableWidgetItem('#' + m.id_string)
            self.table.setItem(row, 0, item)
            for c, attr_name in enumerate(self.attr_names):
                self.table.setItem(
                    row
                    , c + 1
                    , QTableWidgetItem("%g" % getattr(m, attr_name) if hasattr(m, attr_name) else "")
                )
        for i in range(self.table.columnCount()):
            self.table.resizeColumnToContents(i)

    def _models_removed_cb(self, *args):
        remaining = [m for m in self.models if m.id is not None]
        if remaining == self.models:
            return
        self.models = remaining
        if not self.models:
            self.delete()
        else:
            self._fill_table()

    def _table_selection_cb(self):
        rows = set([index.row() for index in self.table.selectedIndexes()])
        sel_ids = set([self.table.item(r, 0).text() for r in rows])
        for m in self.models:
            m.display = ('#' + m.id_string) in sel_ids or not sel_ids
