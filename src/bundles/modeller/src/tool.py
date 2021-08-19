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

    help = "help:user/tools/modeller.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QListWidget, QFormLayout, QAbstractItemView, QGroupBox, QVBoxLayout
        from Qt.QtWidgets import QDialogButtonBox as qbbox
        interface_layout = QVBoxLayout()
        interface_layout.setContentsMargins(0,0,0,0)
        interface_layout.setSpacing(0)
        parent.setLayout(interface_layout)
        alignments_area = QGroupBox("Sequence alignments")
        interface_layout.addWidget(alignments_area)
        interface_layout.setStretchFactor(alignments_area, 1)
        alignments_layout = QVBoxLayout()
        alignments_layout.setContentsMargins(0,0,0,0)
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
        options_layout.setContentsMargins(0,0,0,0)
        options_area.setLayout(options_layout)
        interface_layout.addWidget(options_area)
        interface_layout.setStretchFactor(options_area, 2)
        from chimerax.ui.options import CategorizedSettingsPanel, BooleanOption, IntOption, PasswordOption, \
            OutputFolderOption
        panel = CategorizedSettingsPanel(category_sorting=False, buttons=False)
        options_layout.addWidget(panel)
        from .settings import get_settings
        settings = get_settings(session)
        panel.add_option("Basic", BooleanOption("Make multichain model from multichain template",
            settings.multichain, None,
            balloon=
            "If false, all chains (templates) associated with an alignment will be used in combination\n"
            "to model the target sequence of that alignment, i.e. a monomer will be generated from the\n"
            "alignment.  If true, the target sequence will be modeled from each template, i.e. a multimer\n"
            "will be generated from the alignment (assuming multiple chains are associated).",
            attr_name="multichain", settings=settings))
        max_models = 1000
        panel.add_option("Basic", IntOption("Number of models", settings.num_models, None,
            attr_name="num_models", settings=settings, min=1, max=max_models, balloon=
            "Number of model structures to generate.  Must be no more than %d.\n"
            "Warning: please consider the calculation time" % max_models))
        key = "" if settings.license_key is None else settings.license_key
        panel.add_option("Basic", PasswordOption('<a href="https://www.salilab.org/modeller/registration.html">Modeller license key</a>', key, None, attr_name="license_key",
            settings=settings, balloon=
            "Your Modeller license key.  You can obtain a license key by registering at the Modeller web site"))
        panel.add_option("Advanced", BooleanOption("Use fast/approximate mode (produces only one model)",
            settings.fast, None, attr_name="fast", settings=settings, balloon=
            "If enabled, use a fast approximate method to generate a single model.\n"
            "Typically use to get a rough idea what the model will look like or\n"
            "to check that the alignment is reasonable."))
        panel.add_option("Advanced", BooleanOption("Include non-water HETATM residues from template",
            settings.het_preserve, None, attr_name="het_preserve", settings=settings, balloon=
            "If enabled, all non-water HETATM residues in the template\n"
            "structure(s) will be transferred into the generated models."))
        panel.add_option("Advanced", BooleanOption("Build models with hydrogens",
            settings.hydrogens, None, attr_name="hydrogens", settings=settings, balloon=
            "If enabled, the generated models will include hydrogen atoms.\n"
            "Otherwise, only heavy atom coordinates will be built.\n"
            "Increases computation time by approximately a factor of 4."))
        panel.add_option("Advanced", OutputFolderOption("Temporary folder location (optional)",
            settings.temp_path, None, attr_name="temp_path", settings=settings, balloon=
            "Specify a folder for temporary files.  If not specified,\n"
            "a location will be generated automatically."))
        panel.add_option("Advanced", BooleanOption("Include water molecules from template",
            settings.water_preserve, None, attr_name="water_preserve", settings=settings, balloon=
            "If enabled, all water molecules in the template\n"
            "structure(s) will be included in the generated models."))
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import Citation
        interface_layout.addWidget(Citation(session,
            "A. Sali and T.L. Blundell.\n"
            "Comparative protein modelling by satisfaction of spatial restraints.\n"
            "J. Mol. Biol. 234, 779-815, 1993.",
            prefix="Publications using Modeller results should cite:", pubmed_id=18428767),
            alignment=Qt.AlignCenter)
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
            aln_seq_args.append(StringArg.unparse("%s:%d" % (aln.ident, aln.seqs.index(seq)+1)))
        from .settings import get_settings
        settings = get_settings(self.session)
        run(self.session, "modeller comparative %s multichain %s numModels %d fast %s hetPreserve %s"
            " hydrogens %s%s waterPreserve %s"% (
            " ".join(aln_seq_args),
            repr(settings.multichain).lower(),
            settings.num_models,
            repr(settings.fast).lower(),
            repr(settings.het_preserve).lower(),
            repr(settings.hydrogens).lower(),
            " tempPath %s" % FileNameArg.unparse(settings.temp_path) if settings.temp_path else "",
            repr(settings.water_preserve).lower()
            ))
        self.delete()

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

        from Qt.QtWidgets import QTableWidget, QVBoxLayout, QAbstractItemView, QWidget, QPushButton
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
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
        from Qt.QtWidgets import QAction
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
        self.table.setColumnCount(len(self.attr_names)+1)
        self.table.setHorizontalHeaderLabels(["Model"] + [attr_name[9:].replace('_', ' ')
            for attr_name in self.attr_names])
        self.table.setRowCount(len(self.models))
        from Qt.QtWidgets import QTableWidgetItem
        for row, m in enumerate(self.models):
            item = QTableWidgetItem('#' + m.id_string)
            self.table.setItem(row, 0, item)
            for c, attr_name in enumerate(self.attr_names):
                self.table.setItem(row, c+1, QTableWidgetItem("%g" % getattr(m, attr_name)
                    if hasattr(m, attr_name) else ""))
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
