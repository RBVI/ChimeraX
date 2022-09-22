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

from Qt.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QSizePolicy
from Qt.QtCore import Qt


class SaveOptionsWidget(QFrame):
    tip_text = "To save multiple structures into <i>multiple</i> files," \
            ' include the literal strings "[NAME]" and/or "[NUMBER]" in the file name you provide.' \
            "  Those strings will be replaced with the model name / ID number in the final file names."

    def __init__(self, session):
        super().__init__()
        self.session = session

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(5)

        from chimerax.atomic import Structure
        from chimerax.ui import shrink_font
        show_tip = len([m for m in session.models if isinstance(m, Structure)]) > 1
        if show_tip:
            self.multiple_models_tip = mmt = QLabel(self.tip_text)
            shrink_font(mmt, 0.85)
            mmt.setWordWrap(True)
            mmt.setAlignment(Qt.AlignLeft)
            layout.addWidget(mmt)

        arguments_layout = QHBoxLayout()
        layout.addLayout(arguments_layout)

        models_layout = QVBoxLayout()
        arguments_layout.addLayout(models_layout, stretch=1)
        models_layout.setSpacing(0)
        models_label = QLabel("Save models")
        shrink_font(models_label)
        models_layout.addWidget(models_label, alignment=Qt.AlignLeft)
        from chimerax.atomic.widgets import StructureListWidget
        self.structure_list = StructureListWidget(session)
        self.structure_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        models_layout.addWidget(self.structure_list)
        if show_tip:
            self.structure_list.value_changed.connect(self._update_models_tip)
            self._update_models_tip()

        options_layout = QVBoxLayout()
        arguments_layout.addLayout(options_layout)

        self.displayed_only = QCheckBox('Save displayed atoms only')
        options_layout.addWidget(self.displayed_only, alignment=Qt.AlignLeft)

        self.selected_only = QCheckBox('Save selected atoms only')
        options_layout.addWidget(self.selected_only, alignment=Qt.AlignLeft)

        self.fixed_width = QCheckBox('Use fixed-width columns in output')
        options_layout.addWidget(self.fixed_width, alignment=Qt.AlignLeft)
        self.fixed_width.setToolTip(
            "Whether to write fixed-width columns, as in mmCIF files from the RCSB"
            " Protein Data Bank.\nFixed-width columns make reading the file faster in ChimeraX")
        self.fixed_width.setChecked(True)

        self.best_guess = QCheckBox('Estimate sequence information if necessary')
        options_layout.addWidget(self.best_guess, alignment=Qt.AlignLeft)
        self.best_guess.setToolTip(
            "Whether to try to write polymer sequence information"
            " when it was missing from the input file")

        self.computed_sheets = QCheckBox('Computed sheets')
        options_layout.addWidget(self.computed_sheets, alignment=Qt.AlignLeft)
        self.computed_sheets.setToolTip(
            "Workaround ChimeraX not retaining sheet information from opened files"
            " and save computed sheet and strand information")

        self.simple_rel_models = len(self.structure_list.value) < 2
        if self.simple_rel_models:
            self.rel_models = QCheckBox('Use untransformed coordinates')
            options_layout.addWidget(self.rel_models, alignment=Qt.AlignLeft)
        else:
            rel_layout = QHBoxLayout()
            options_layout.addLayout(rel_layout)
            self.rel_models = QCheckBox('Save relative to model:')
            rel_layout.addWidget(self.rel_models, alignment=Qt.AlignLeft)
            from chimerax.ui.widgets import ModelMenuButton
            self.rel_model_menu = ModelMenuButton(session)
            rel_layout.addWidget(self.rel_model_menu, alignment=Qt.AlignLeft)
        self.rel_models.setChecked(True)

        self.setLayout(layout)

    def options_string(self):
        models = self.structure_list.value
        from chimerax.core.errors import UserError
        if not models:
            raise UserError("No models chosen for saving")
        from chimerax.atomic import Structure
        from chimerax.core.commands import concise_model_spec
        spec = concise_model_spec(self.session, models, relevant_types=Structure)
        if spec:
            cmd = "models " + spec
        else:
            cmd = ""
        if self.displayed_only.isChecked():
            if cmd:
                cmd += ' '
            cmd += "displayedOnly true"
        if self.selected_only.isChecked():
            if cmd:
                cmd += ' '
            cmd += "selectedOnly true"
        if not self.fixed_width.isChecked():
            if cmd:
                cmd += ' '
            cmd += "fixedWidth false"
        if self.best_guess.isChecked():
            if cmd:
                cmd += ' '
            cmd += "bestGuess true"
        if self.compute_sheets.isChecked():
            if cmd:
                cmd += ' '
            cmd += "computeSheets true"
        if self.rel_models.isChecked():
            if cmd:
                cmd += ' '
            if self.simple_rel_models:
                rel_model = models[0]
            else:
                rel_model = self.rel_model_menu.value
                if rel_model is None:
                    raise UserError("No model chosen to save relative to")
            cmd += "relModel #" + rel_model.id_string
        return cmd

    def _update_models_tip(self):
        self.multiple_models_tip.setEnabled(len(self.structure_list.value) > 1)
