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

from chimerax.core.tools import ToolInstance


class ColorActions(ToolInstance):

    SESSION_ENDURING = True

    help = "help:user/tools/coloractions.html"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QDialogButtonBox, QWidget, QPushButton, \
            QLabel, QCheckBox, QFrame, QGroupBox, QGridLayout, QScrollArea
        from Qt.QtGui import QColor, QPixmap, QIcon
        from Qt.QtCore import Qt, QTimer
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        main_dialog_area = QWidget()
        layout.addWidget(main_dialog_area)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0,7,0,7)
        main_dialog_area.setLayout(main_layout)

        fav_color_area = QWidget()
        main_layout.addWidget(fav_color_area)
        fav_color_layout = QVBoxLayout()
        fav_color_layout.setContentsMargins(0,0,0,0)
        fav_color_layout.setSpacing(0)
        fav_color_area.setLayout(fav_color_layout)
        spaced_names = [ "red", "orange red", "orange", "yellow", "lime", "forest green", "cyan",
                "light sea green", "blue", "cornflower blue", "medium blue", "purple", "hot pink",
                "magenta", "white", "light gray", "gray", "dark gray", "dim gray", "black"]
        for spaced_name in spaced_names:
            svg_name = "".join(spaced_name.split())
            color = QColor(svg_name)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            button = QPushButton(icon, spaced_name.title())
            button.released.connect(lambda *, clr=spaced_name: self._color(clr))
            button.setStyleSheet("QPushButton { text-align: left; }")
            fav_color_layout.addWidget(button)

        actions_area = QWidget()
        main_layout.addWidget(actions_area)
        actions_layout = QVBoxLayout()
        actions_area.setLayout(actions_layout)

        actions_layout.addStretch(1)

        header = QLabel("Coloring applies to:")
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignCenter)
        actions_layout.addWidget(header, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.target_button_info = []
        for label, target, initial_on in [("Atoms/Bonds", 'a', True),  ("Cartoons", 'c', True),
                ("Surfaces", 's', True), ("Pseudobonds", 'p', True), ("Ring Fill", 'f', True),
                ("Labels", 'l', False)]:
            chk = QCheckBox(label)
            chk.setChecked(initial_on)
            chk.clicked.connect(self._clear_global_buttons)
            actions_layout.addWidget(chk)
            self.target_button_info.append((chk, target))

        sep = QFrame()
        sep.setFrameStyle(QFrame.HLine)
        actions_layout.addWidget(sep, stretch=1)

        self.global_button_info = []
        for label, command in [("Background", "set bg %s")]:
            chk = QCheckBox(label)
            chk.setChecked(False)
            chk.clicked.connect(self._clear_targeted_buttons)
            actions_layout.addWidget(chk)
            self.global_button_info.append((chk, command))

        actions_layout.addStretch(1)

        from chimerax.core.commands import run
        grp = QGroupBox("Other colorings:")
        actions_layout.addWidget(grp)
        grp_layout = QVBoxLayout()
        grp_layout.setContentsMargins(0,0,0,0)
        grp_layout.setSpacing(0)
        grp.setLayout(grp_layout)
        for label, arg, tooltip in [
                ("Heteroatom", "het", "Color non-carbon atoms by chemical element"),
                ("Element", "element", "Color atoms by chemical element"),
                ("Nucleotide Type", "nucleotide", "Color nucleotide residues by the type of their base"),
                ("Chain", "chain", "Give each chain a different color"),
                ("Polymer", "polymer", "Color chains differently, except that chains with the same sequence"
                    " receive the same color")]:
            but = QPushButton("By " + label)
            if tooltip:
                but.setToolTip(tooltip)
            but.clicked.connect(lambda *, run=run, ses=self.session, arg=arg: run(ses,
                "color " + ("" if ses.selection.empty() else "sel ") + "by" + arg))
            grp_layout.addWidget(but)
        but = QPushButton("From Editor")
        but.setToolTip("Bring up a color editor to choose the color")
        but.clicked.connect(self.session.ui.main_window.color_by_editor)
        grp_layout.addWidget(but)

        actions_layout.addStretch(1)

        self.all_colors_check_box = chk = QCheckBox("Show all colors \N{RIGHTWARDS ARROW}")
        chk.setChecked(False)
        chk.clicked.connect(self._toggle_all_colors)
        actions_layout.addWidget(chk, alignment=Qt.AlignRight)

        actions_layout.addStretch(1)

        self.all_colors_area = QScrollArea()
        # hack to get reasonable initial size; allow smaller resize after show()
        self.ac_preferred_width = 500
        self.all_colors_area.setHidden(True)
        all_colors_widget = QWidget()
        from chimerax.core.colors import BuiltinColors
        # for colors with the exact same RGBA value, only keep one
        canonical = {}
        for name, color in BuiltinColors.items():
            key = tuple(color.rgba)
            if key[-1] == 0.0:
                continue
            try:
                canonical_name = canonical[key]
            except KeyError:
                canonical[key] = name
                continue
            if 'grey' in name:
                continue
            if 'grey' in canonical_name:
                canonical[key] = name
                continue
            if len(name) > len(canonical_name):
                # names with spaces in them are longer...
                canonical[key] = name
                continue
            if len(name) == len(canonical_name) and name > canonical_name:
                # tie breaker for otherwise equal-value names, so that
                # the set of names is the same between invocations of the tool
                canonical[key] = name
        rgbas = list(canonical.keys())
        rgbas.sort(key=self._rgba_key)
        color_names = [canonical[rgba] for rgba in rgbas]
        all_colors_layout = QGridLayout()
        all_colors_layout.setContentsMargins(0,0,0,0)
        all_colors_layout.setSpacing(0)
        all_colors_widget.setLayout(all_colors_layout)
        num_rows = len(spaced_names)
        row = column = 0
        for spaced_name in color_names:
            #svg_name = "".join(spaced_name.split())
            #color = QColor(svg_name)
            # QColor doesn't know the name "rebeccapurple", so...
            color = QColor(*[int(c*255+0.5) for c in BuiltinColors[spaced_name].rgba])
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            button = QPushButton(icon, spaced_name.title())
            button.released.connect(lambda *, clr=spaced_name: self._color(clr))
            button.setStyleSheet("QPushButton { text-align: left; }")
            all_colors_layout.addWidget(button, row, column)
            row += 1
            if row >= num_rows:
                row = 0
                column += 1
        self.all_colors_area.setWidget(all_colors_widget)
        main_layout.addWidget(self.all_colors_area)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close | qbbox.Help)
        bbox.rejected.connect(self.delete)
        if self.help:
            bbox.helpRequested.connect(lambda *, run=run, ses=self.session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    @classmethod
    def get_singleton(cls, session, tool_name):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, tool_name)

    def _clear_targeted_buttons(self, *args):
        for button, *args in self.target_button_info:
            button.setChecked(False)

    def _clear_global_buttons(self, *args):
        for button, *args in self.global_button_info:
            button.setChecked(False)

    def _color(self, color_name):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        target = ""
        for but, targ_char in self.target_button_info:
            if but.isChecked():
                target += targ_char
        commands = []
        if target:
            commands.append("color "
                + ("" if self.session.selection.empty() else "sel ") + color_name
                + ("" if target == "acspf" else " target " + target))

        for but, cmd in self.global_button_info:
            if but.isChecked():
                commands.append(cmd % color_name)

        if commands:
            run(self.session, " ; ".join(commands))
        else:
            raise UserError("No target buttons for the coloring action are checked")

    def _rgba_key(self, rgba):
        brightness = rgba[0] + rgba[1] + rgba[2]
        if brightness > 2.2:
            return (7, rgba)
        if brightness < 0.5:
            return (8, rgba)
        reddish, greenish, bluish = rgba[0] > 0.5, rgba[1] > 0.5, rgba[2] > 0.5
        if reddish and not greenish and not bluish:
            return (0, -rgba[0]+rgba[1]+rgba[2])
        if reddish and greenish and not bluish:
            return (1, -rgba[0]-rgba[1]+rgba[2])
        if not reddish and greenish and not bluish:
            return (2, rgba[0]-rgba[1]+rgba[2])
        if not reddish and greenish and bluish:
            return (3, rgba[0]-rgba[1]-rgba[2])
        if not reddish and not greenish and bluish:
            return (4, rgba[0]+rgba[1]-rgba[2])
        if reddish and not greenish and bluish:
            return (5, -rgba[0]+rgba[1]-rgba[2])
        return (6, rgba)

    def _toggle_all_colors(self, *args):
        if self.all_colors_check_box.isChecked():
            self.all_colors_area.setMinimumWidth(self.ac_preferred_width)
            self.all_colors_area.setHidden(False)
            self.all_colors_area.setMinimumWidth(1)
        else:
            self.ac_preferred_width = self.all_colors_area.width()
            self.all_colors_area.setHidden(True)
            self.tool_window.shrink_to_fit()
