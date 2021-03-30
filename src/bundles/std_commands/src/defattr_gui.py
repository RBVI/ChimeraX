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

from Qt.QtWidgets import QFrame, QGridLayout
from Qt.QtCore import Qt
from chimerax.core.errors import UserError

class SaveOptionsWidget(QFrame):
    def __init__(self, session):
        super().__init__()
        self.session = session
        layout = QGridLayout()
        layout.setContentsMargins(2, 0, 0, 0)

        from chimerax.atomic.widgets import StructureListWidget
        self._structure_list = StructureListWidget(session)
        layout.addWidget(self._structure_list, 0, 0, 1, 2)
        layout.setRowStretch(0, 1)

        from chimerax.ui.options import OptionsPanel, StringOption, EnumOption, BooleanOption, \
            SymbolicEnumOption
        left_options = OptionsPanel(sorting=False, contents_margins=(1,2,1,2))
        right_options = OptionsPanel(sorting=False, contents_margins=(1,2,1,2))
        layout.addWidget(left_options, 1, 0)
        layout.addWidget(right_options, 1, 1)

        self._attr_name_opt = StringOption("Attribute name", "", None)
        left_options.add_option(self._attr_name_opt)
        class AttrOfOption(SymbolicEnumOption):
            values = ('a', 'r', 'm')
            labels = ("atoms", "residues", "structures")
        self._attr_of_opt = AttrOfOption("Attribute of", 'r', None)
        left_options.add_option(self._attr_of_opt)

        self._sel_restrict_opt = BooleanOption("Also restrict to selection", False, None)
        right_options.add_option(self._sel_restrict_opt)
        from .defattr import match_modes
        class MatchModesOption(EnumOption):
            values = match_modes
        self._match_mode_opt = MatchModesOption("Match mode", "1-to-1", None, balloon=
            "Expected matches per assignment line")
        right_options.add_option(self._match_mode_opt)
        class IncludeModelOption(EnumOption):
            values = ('auto', 'true', 'false')
        self._include_model_opt = IncludeModelOption("Include model in specifiers", 'auto', None,
            balloon="Include model part of atom specifiers.  If 'auto' in the model part\n"
            " will be included only if multiple structures are open.")
        right_options.add_option(self._include_model_opt)

        self.setLayout(layout)

    def options_string(self):
        structures = self._structure_list.value
        if not structures:
            raise UserError("No structures chosen")

        arglets = []
        from chimerax.core.commands import concise_model_spec, AttrNameArg, AnnotationError
        from chimerax.atomic import Structure
        spec = concise_model_spec(self.session, structures, relevant_types=Structure)
        if spec:
            arglets.append(spec)

        raw_attr_name_text = self._attr_name_opt.value.strip()
        if not raw_attr_name_text:
            raise UserError("'Attribute name' field is blank")
        try:
             attr_name, log_text, remainder = AttrNameArg.parse(raw_attr_name_text, self.session)
        except AnnotationError as e:
            raise UserError(str(e))
        if remainder:
            raise UserError("Attribute names cannot contain whitespace")
        arglets.append("attrName %s:%s" % (self._attr_of_opt.value, attr_name))

        if self._sel_restrict_opt.value:
            arglets.append("selected true")

        match_mode = self._match_mode_opt.value
        if match_mode != "1-to-1":
            arglets.append("matchMode " + match_mode)

        include_model = self._include_model_opt.value
        if include_model != "auto":
            arglets.append("modelIds " + include_model)
        return " ".join(arglets)

