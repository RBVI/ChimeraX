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

from Qt.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QRadioButton, QLineEdit, QWidget
from Qt.QtWidgets import QCheckBox, QSizePolicy, QHBoxLayout, QTextEdit, QDialog, QTableWidget
from Qt.QtWidgets import QTableWidgetItem, QPushButton, QMenu, QGraphicsView, QGraphicsScene
from Qt.QtGui import QDoubleValidator, QIntValidator, QPalette, QPainter
from Qt.QtCore import Qt
from chimerax.ui.options import SymbolicEnumOption, OptionsPanel
from chimerax.core.errors import UserError
import abc

class StartStructureProvider(metaclass=abc.ABCMeta):
    def __init__(self, session):
        self.session = session

    @abc.abstractmethod
    def command_string(self, widget):
        # Return the command (sub)string corresponding to the settings in the widget.
        # Can return None if the provider doesn't directly add atoms [e.g. links to another tool].
        # If 'new_model_only' was "true" in the Provider tag, then the returned string should be
        # the _entire_ command for opening the model.
        # Otherwise it should be the argument substring of your 'build start' provider's subcommand
        pass

    def execute_command(self, structure, args):
        # Execute this providers subcommand into the given structure and with the given args.
        # This method will not be called if 'new_model_only' was "true" in the Provider tag
        # (i.e. command_string() returns a full command) or if 'indirect' was "true" in the
        # Provider tag (i.e. provider doesn't directly add atoms)
        raise NotImplementedError("Start-structure provider failed to implement 'execute_command'")

    @abc.abstractmethod
    def fill_parameters_widget(self, widget):
        # populate the given widget with controls for specifying the arg values for this providers
        # command
        pass

def get_provider(session, name, mgr):
    from .fragment_manager import FragmentManager
    if isinstance(mgr, FragmentManager):
        return get_fragment(name)
    return {
        "atom": AtomProvider,
        "fragment": FragmentProvider,
        "nucleic": NucleicProvider,
        "peptide": PeptideProvider,
    }[name](session, name)

from chimerax.core.commands import register, CmdDesc, Command, Annotation, StringArg, Float2Arg, Float3Arg, \
    BoolArg, DynamicEnum, EnumOf

class AtomProvider(StartStructureProvider):
    def __init__(self, session, name):
        super().__init__(session)
        self.name = name
        # register commands to private registry
        from chimerax.core.commands.cli import RegisteredCommandInfo
        self.registry = RegisteredCommandInfo()
        register(name,
            CmdDesc(
                keyword=[("position", Float3Arg), ("res_name", StringArg), ("select", BoolArg)],
                synopsis="place helium atom"
            ), shim_place_atom, registry=self.registry)

    def command_string(self, widget):
        from chimerax.core.commands import StringArg
        args = []
        button = widget.findChild(QRadioButton, "atom centered")
        if not button.isChecked():
            coords = []
            for axis in "xyz":
                coord_entry = widget.findChild(QLineEdit, axis + " coord")
                if not coord_entry.hasAcceptableInput():
                    raise UserError("%s coordinate must be a number" % axis)
                coords.append(coord_entry.text().strip())
            args.append("pos " + ','.join(coords))
        res_name_entry = widget.findChild(QLineEdit, "res name")
        res_name = res_name_entry.text().strip()
        if not res_name:
            raise UserError("Residue name must not be empty/blank")
        if res_name != "UNL":
            args.append("res %s" % StringArg.unparse(res_name))
        check_box = widget.findChild(QCheckBox, "select atom")
        if not check_box.isChecked():
            args.append("select false")
        return " ".join(args)

    def execute_command(self, structure, args):
        # the command uses the trick that the structure arg is temporarily made available in the
        # global namespace as '_structure'
        global _structure
        _structure = structure
        try:
            cmd = Command(self.session, registry=self.registry)
            return cmd.run(self.name + ' ' + args, log=False)[0]
        finally:
            _structure = None

    def fill_parameters_widget(self, widget):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.addWidget(QLabel("Place helium atom at:"), alignment=Qt.AlignCenter)
        button_area = QWidget()
        layout.addWidget(button_area, alignment=Qt.AlignCenter)
        button_layout = QGridLayout()
        button_area.setLayout(button_layout)
        button = QRadioButton("Center of view")
        button.setChecked(True)
        button.setObjectName("atom centered")
        button_layout.addWidget(button, 0, 0, 1, 7, alignment=Qt.AlignLeft)
        button_layout.addWidget(QRadioButton(""), 1, 0, alignment=Qt.AlignLeft)
        for i, axis in enumerate("xyz"):
            button_layout.addWidget(QLabel(axis + ":"), 1, i*2+1, alignment=Qt.AlignRight)
            coord_entry = QLineEdit()
            coord_entry.setText("0")
            coord_entry.setFixedWidth(50)
            coord_entry.setValidator(QDoubleValidator())
            coord_entry.setObjectName(axis + " coord")
            button_layout.addWidget(coord_entry, 1, 2*(i+1), alignment=Qt.AlignLeft)
        tip = QLabel("Use 'Modify Structure' section to change element type and add bonded atoms")
        tip.setWordWrap(True)
        # specify alignment within the label itself (instead of the layout) so that the label
        # is given the full width of the layout to work with, otherwise you get unneeded line
        # wrapping
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)
        res_name_area = QWidget()
        layout.addWidget(res_name_area, alignment=Qt.AlignCenter)
        res_name_layout = QHBoxLayout()
        res_name_area.setLayout(res_name_layout)
        res_name_layout.addWidget(QLabel("Residue name:"))
        res_name_entry = QLineEdit()
        res_name_entry.setText("UNL")
        res_name_entry.setFixedWidth(50)
        res_name_entry.setObjectName("res name")
        res_name_layout.addWidget(res_name_entry)
        check_box = QCheckBox("Select placed atom")
        check_box.setChecked(True)
        check_box.setObjectName("select atom")
        layout.addWidget(check_box, alignment=Qt.AlignCenter)
        layout.addStretch(1)

class FragmentProvider(StartStructureProvider):
    def __init__(self, session, name):
        super().__init__(session)
        self.name = name
        # register commands to private registry
        from chimerax.core.commands.cli import RegisteredCommandInfo
        self.registry = RegisteredCommandInfo()

        from .fragment_manager import get_manager
        register(name,
            CmdDesc(
                required=[("fragment_name", EnumOf(get_manager(session).fragment_names))],
                keyword=[("position", Float3Arg), ("res_name", StringArg)],
                synopsis="place initial structure fragment"
            ), shim_place_fragment, registry=self.registry)

    def command_string(self, widget):
        return widget.findChild(QPushButton, "fragment name").text()

    def execute_command(self, structure, args):
        # the command uses the trick that the structure arg is temporarily made available in the
        # global namespace as '_structure'
        global _structure
        _structure = structure
        try:
            cmd = Command(self.session, registry=self.registry)
            return cmd.run(self.name + ' ' + args, log=False)[0]
        finally:
            _structure = None

    def fill_parameters_widget(self, widget):
        from .fragment_manager import get_manager
        mgr = get_manager(self.session)
        frag_names = mgr.fragment_names
        init_fragment_name = "benzene" if "benzene" in frag_names else frag_names[0]

        layout = QHBoxLayout()
        layout.setContentsMargins(3,0,3,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        but = QPushButton()
        but.setObjectName("fragment name")
        layout.addWidget(but)

        gscene = QGraphicsScene()
        gscene.setBackgroundBrush(widget.palette().brush(QPalette.Background))
        gview = QGraphicsView(gscene)
        gview.setRenderHint(QPainter.Antialiasing, True)
        layout.addWidget(gview)

        # organize the fragment names so that the menus can be alphabetized
        by_cat = {}
        for frag_name in frag_names:
            category = mgr.fragment_category(frag_name)
            by_cat.setdefault(category, []).append(frag_name)

        def set_button_text(frag_name, but=but, scene=gscene, mgr=mgr):
            but.setText(frag_name)
            frag = mgr.fragment(frag_name)
            scene.clear()
            frag.depict(scene, 15)
            scene.setSceneRect(scene.itemsBoundingRect())

        root_menu = QMenu(widget)
        but.setMenu(root_menu)
        categories = sorted(list(by_cat.keys()))
        for cat in categories:
            cat_menu = QMenu(cat, widget)
            cat_menu.triggered.connect(lambda action, func=set_button_text: func(action.text()))
            root_menu.addMenu(cat_menu)
            for entry in sorted(list(by_cat[cat]), key=lambda x: x.casefold()):
                cat_menu.addAction(entry)
        set_button_text(init_fragment_name)

class NucleicProvider(StartStructureProvider):
    def __init__(self, session, name):
        super().__init__(session)
        self.name = name
        # register commands to private registry
        from chimerax.core.commands.cli import RegisteredCommandInfo
        self.registry = RegisteredCommandInfo()

        from .start import nucleic_forms
        register(name,
            CmdDesc(
                required=[("sequence", StringArg)],
                keyword=[("form", EnumOf(nucleic_forms)), ("type", EnumOf(["dna", "rna", "hybrid"])),
                    ("position", Float3Arg)],
                synopsis="construct helical nucleic acid from sequence"
            ), shim_place_nucleic_acid, registry=self.registry)

    def command_string(self, widget):
        args = []
        seq_edit = widget.findChild(QTextEdit, "sequence")
        seq = "".join(seq_edit.toPlainText().split()).upper()
        if not seq:
            raise UserError("No nucleotide sequence entered")
        args.append(StringArg.unparse(seq))
        for nuc_type in ["dna", "rna", "hybrid"]:
            if widget.findChild(QRadioButton, nuc_type).isChecked():
                args.append("type " + nuc_type)
                break
        from .start import nucleic_forms
        for form in nucleic_forms:
            if widget.findChild(QRadioButton, form).isChecked():
                args.append("form " + form)
                break
        return " ".join(args)

    def execute_command(self, structure, args):
        # the command uses the trick that the structure arg is temporarily made available in the
        # global namespace as '_structure'
        global _structure
        _structure = structure
        try:
            cmd = Command(self.session, registry=self.registry)
            return cmd.run(self.name + ' ' + args, log=False)[0]
        finally:
            _structure = None

    def fill_parameters_widget(self, widget):
        layout = QVBoxLayout()
        layout.setContentsMargins(3,0,3,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.addWidget(QLabel("Sequence", alignment=Qt.AlignCenter))
        seq = QTextEdit()
        seq.setObjectName("sequence")
        layout.addWidget(seq, stretch=1)
        guidance = QLabel("Enter single strand; double helix will be generated", alignment=Qt.AlignCenter)
        from chimerax.ui import shrink_font
        shrink_font(guidance)
        layout.addWidget(guidance)
        radios_layout = QHBoxLayout()
        layout.addLayout(radios_layout)
        type_layout = QVBoxLayout()
        type_layout.setContentsMargins(0,0,0,0)
        type_widget = QWidget()
        type_widget.setLayout(type_layout)
        radios_layout.addStretch(1)
        radios_layout.addWidget(type_widget)
        button = QRadioButton("DNA")
        button.setChecked(True)
        button.setObjectName("dna")
        type_layout.addWidget(button, alignment=Qt.AlignLeft)
        button = QRadioButton("RNA")
        button.setObjectName("rna")
        type_layout.addWidget(button, alignment=Qt.AlignLeft)
        button = QRadioButton("DNA/RNA hybrid (enter DNA sequence)")
        button.setObjectName("hybrid")
        type_layout.addWidget(button, alignment=Qt.AlignLeft)
        from .start import nucleic_forms
        radios_layout.addStretch(1)
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0,0,0,0)
        form_widget = QWidget()
        form_widget.setLayout(form_layout)
        radios_layout.addWidget(form_widget)
        for form in nucleic_forms:
            button = QRadioButton(form + "-form")
            button.setChecked(form == 'B')
            button.setObjectName(form)
            form_layout.addWidget(button, alignment=Qt.AlignLeft)
        radios_layout.addStretch(1)

class PeptideProvider(StartStructureProvider):
    def __init__(self, session, name):
        super().__init__(session)
        self.name = name
        # register commands to private registry
        from chimerax.core.commands.cli import RegisteredCommandInfo
        self.registry = RegisteredCommandInfo()

        class RepeatableFloat2Arg(Annotation):
            allow_repeat = True
            parse = Float2Arg.parse
            unparse = Float2Arg.unparse

        register(name,
            CmdDesc(
                required=[("sequence", StringArg), ("phi_psis", RepeatableFloat2Arg)],
                keyword=[("position", Float3Arg), ("chain_id", StringArg),
                    ("rot_lib", DynamicEnum(self.session.rotamers.library_names))],
                synopsis="construct peptide from sequence"
            ), shim_place_peptide, registry=self.registry)

    def command_string(self, widget):
        args = []
        seq_edit = widget.findChild(QTextEdit, "peptide sequence")
        seq = "".join(seq_edit.toPlainText().split()).upper()
        if not seq:
            raise UserError("No peptide sequence entered")
        param_dialog = PeptideParamDialog(self.session, widget, seq)
        if not param_dialog.exec():
            from chimerax.core.errors import CancelOperation
            raise CancelOperation("peptide building cancelled")
        args.append(StringArg.unparse(seq))
        args.append(" ".join([Float2Arg.unparse(pp) for pp in param_dialog.phi_psis]))
        args.append("rotLib %s" % StringArg.unparse(param_dialog.rot_lib))
        chain_id = param_dialog.chain_id
        if chain_id:
            args.append("chainId %s" % StringArg.unparse(chain_id))
        return " ".join(args)

    def execute_command(self, structure, args):
        # the command uses the trick that the structure arg is temporarily made available in the
        # global namespace as '_structure'
        global _structure
        _structure = structure
        try:
            cmd = Command(self.session, registry=self.registry)
            return cmd.run(self.name + ' ' + args, log=False)[0]
        finally:
            _structure = None

    def fill_parameters_widget(self, widget):
        layout = QVBoxLayout()
        layout.setContentsMargins(3,0,3,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.addWidget(QLabel("Peptide Sequence", alignment=Qt.AlignCenter))
        peptide_seq = QTextEdit()
        peptide_seq.setObjectName("peptide sequence")
        layout.addWidget(peptide_seq, stretch=1)
        tip = QLabel("'Apply' button will bring up dialog for setting"
            "\N{GREEK CAPITAL LETTER PHI}/\N{GREEK CAPITAL LETTER PSI} angles")
        tip.setWordWrap(True)
        # specify alignment within the label itself (instead of the layout) so that the label
        # is given the full width of the layout to work with, otherwise you get unneeded line
        # wrapping
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

class PeptideParamDialog(QDialog):
    def __init__(self, session, parent, seq):
        super().__init__(parent)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        self.setLayout(layout)

        self.table = QTableWidget(len(seq), 3)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setHorizontalHeaderLabels(["Res", "\N{GREEK CAPITAL LETTER PHI}",
            "\N{GREEK CAPITAL LETTER PSI}"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        for i, c in enumerate(seq):
            item = QTableWidgetItem(c)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item)
            self.table.setItem(i, 1, QTableWidgetItem(""))
            self.table.setItem(i, 2, QTableWidgetItem(""))
        layout.addWidget(self.table, stretch=1)

        phi_psi_layout = QHBoxLayout()
        phi_psi_layout.setContentsMargins(0,0,0,0)
        phi_psi_layout.setSpacing(2)
        set_button = QPushButton("Set")
        set_button.setDefault(False)
        set_button.clicked.connect(self._set_table)
        phi_psi_layout.addWidget(set_button, alignment=Qt.AlignRight)
        angle_range = QDoubleValidator(-180.0, 180.0, 1)
        phi_psi_layout.addWidget(QLabel("selected rows to \N{GREEK CAPITAL LETTER PHI}:"),
            alignment=Qt.AlignRight)
        self.phi_entry = QLineEdit()
        self.phi_entry.setMaximumWidth(45)
        self.phi_entry.setValidator(angle_range)
        phi_psi_layout.addWidget(self.phi_entry, alignment=Qt.AlignLeft)
        phi_psi_layout.addWidget(QLabel("\N{GREEK CAPITAL LETTER PSI}:"), alignment=Qt.AlignRight)
        self.psi_entry = QLineEdit()
        self.psi_entry.setMaximumWidth(45)
        self.psi_entry.setValidator(angle_range)
        phi_psi_layout.addWidget(self.psi_entry, alignment=Qt.AlignLeft)
        container = QWidget()
        container.setLayout(phi_psi_layout)
        layout.addWidget(container, alignment=Qt.AlignCenter)

        seed_option = PhiPsiOption("Seed above \N{GREEK CAPITAL LETTER PHI}/"
            "\N{GREEK CAPITAL LETTER PSI} with values for:",
            PhiPsiOption.values[0], self._seed_phi_psi)
        seed_widget = OptionsPanel(scrolled=False, contents_margins=(1,2,1,2))
        seed_widget.add_option(seed_option)
        layout.addWidget(seed_widget)

        lib_chain_layout = QHBoxLayout()
        lib_chain_layout.setContentsMargins(0,0,0,0)
        lib_chain_layout.setSpacing(2)
        lib_chain_layout.addWidget(QLabel("Rotamer library:"), alignment=Qt.AlignRight)
        self.rot_lib_button = session.rotamers.library_name_menu()
        lib_chain_layout.addWidget(self.rot_lib_button, alignment=Qt.AlignLeft)
        lib_chain_layout.addWidget(QLabel("chain ID:"), alignment=Qt.AlignRight)
        self.chain_entry = QLineEdit()
        self.chain_entry.setPlaceholderText("auto")
        self.chain_entry.setMaximumWidth(35)
        lib_chain_layout.addWidget(self.chain_entry, alignment=Qt.AlignLeft)
        container = QWidget()
        container.setLayout(lib_chain_layout)
        layout.addWidget(container, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._seed_phi_psi(seed_option)
        self._set_table()

    @property
    def chain_id(self):
        return self.chain_entry.text()

    @property
    def phi_psis(self):
        phi_psis = []
        for i in range(self.table.rowCount()):
            phi_psis.append([float(self.table.item(i, col).text()) for col in [1,2]])
        return phi_psis

    @property
    def rot_lib(self):
        return self.rot_lib_button.text()

    def _seed_phi_psi(self, option):
        phi, psi = option.value
        self.phi_entry.setText("%g" % phi)
        self.psi_entry.setText("%g" % psi)

    def _set_table(self, *args):
        if not self.phi_entry.hasAcceptableInput():
            raise UserError("\N{GREEK CAPITAL LETTER PHI} must be in the range -180 to 180")
        if not self.psi_entry.hasAcceptableInput():
            raise UserError("\N{GREEK CAPITAL LETTER PSI} must be in the range -180 to 180")
        phi = self.phi_entry.text().strip()
        psi = self.psi_entry.text().strip()
        row_indices = self.table.selectionModel().selectedRows()
        if row_indices:
            rows = [ri.row() for ri in row_indices]
        else:
            rows = range(self.table.rowCount())
        for row in rows:
            self.table.item(row, 1).setText(phi)
            self.table.item(row, 2).setText(psi)
        self.table.resizeColumnsToContents()

class PhiPsiOption(SymbolicEnumOption):
    values = [(-57, -47), (-139, 135), (-119, 113), (-49, -26), (-57, -70)]
    labels = [
        "\N{GREEK SMALL LETTER ALPHA} helix",
        "antiparallel \N{GREEK SMALL LETTER BETA} strand",
        "parallel \N{GREEK SMALL LETTER BETA} strand",
        "3\N{SUBSCRIPT ONE}\N{SUBSCRIPT ZERO} helix",
        "\N{GREEK SMALL LETTER PI} helix"
    ]

def show_sticks_and_elements(atoms):
    from chimerax.atomic import Atom
    atoms.draw_modes = Atom.STICK_STYLE
    from chimerax.atomic.colors import element_colors
    atoms.colors = element_colors(atoms.element_numbers)

def shim_place_atom(session, position=None, res_name="UNL", select=True):
    from .start import place_helium
    a = place_helium(_structure, res_name, position=position)
    if select:
        session.selection.clear()
        a.selected = True
    return a

def shim_place_fragment(session, fragment_name, position=None, res_name="UNL"):
    set_styles = _structure.num_atoms != 0
    from .start import place_fragment
    res = place_fragment(_structure, fragment_name, res_name, position=position)
    if set_styles:
        show_sticks_and_elements(res.atoms)
    return res

def shim_place_nucleic_acid(session, sequence, **kw):
    from .start import place_nucleic_acid, NucleicError
    try:
        chains = place_nucleic_acid(_structure, sequence, **kw)
    except NucleicError as e:
        raise UserError(str(e))
    show_sticks_and_elements(chains.existing_residues.atoms)
    return chains

def shim_place_peptide(session, sequence, phi_psis, **kw):
    set_styles = _structure.num_atoms != 0
    from .start import place_peptide, PeptideError
    try:
        residues = place_peptide(_structure, sequence, phi_psis, **kw)
    except PeptideError as e:
        raise UserError(str(e))
    if set_styles:
        show_sticks_and_elements(residues.atoms)
    return residues

def get_fragment(name):
    import os
    fragment_file = os.path.join(os.path.dirname(__file__), "fragments", name + ".py")
    frag_vars = {}
    with open(fragment_file) as f:
        exec(f.read(), frag_vars)
    from .fragment import Fragment
    return Fragment(frag_vars['name'], frag_vars['atoms'], frag_vars['bonds'])
