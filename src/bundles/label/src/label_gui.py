# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# User interface for building cages.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class LabelGUI(ToolInstance):
    help = "help:user/tools/2dlabels.html"

    def __init__(self, session, tool_name):

        self._current_label = None		# Label instance, "all", or None
        self._label_position = (0.5, 0.5)
        self._current_arrow = None
        self._move_label_handler = None
        self._delayed_log_timer = None
        self._delayed_command = None
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        tw.shown_changed = self._shown_changed
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Create Label, mouse and Arrow setting controls
        self._create_label_controls(parent)
        self._create_arrow_controls(parent)
        self._create_mouse_controls(parent)
        self._create_hide_show_text(parent)
        
        # Create buttons
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Delete label', self._delete_label),
                         ('Delete arrow', self._delete_arrow),
                         ('Help', self._show_help)],
                        spacing = 10)
        layout.addWidget(bf)
        
        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, LabelGUI, '2D Labels and Arrows', create=create)

    def _create_label_controls(self, parent):
        from chimerax.ui.widgets import EntriesRow, ColorButton
        EntriesRow(parent, 'Click on graphics to create a label.  Drag labels to move them.')
        font_names = self._font_names
        styles = ('normal', 'bold', 'italic', 'bold italic')
        c1 = EntriesRow(parent, 'Label', ('new', 'all'), 'Text', '', 'Size', 24, 'Color', ColorButton)
        c2 = EntriesRow(parent, '   ', 'Font', tuple(font_names), 'Style', styles,
                        False, 'Background', ColorButton)
        self._label_menu, self._text, self._size, self._color = label_menu, text, size, color = c1.values
        self._font, self._style, self._use_background, self._background_color  = \
            font, style, use_background, bg_color = c2.values
        lmenu = label_menu.widget.menu()
        lmenu.triggered.connect(self._label_menu_changed)
        lmenu.triggered.disconnect(label_menu._menu_selection_cb)  # Don't include label name in menu button text.
        lmenu.aboutToShow.connect(self._fill_label_menu)
        text.widget.textChanged.connect(self._update_label_text)
        text.widget.returnPressed.connect(self._update_label_text)
        text.widget.setMaximumWidth(200)
        size.widget.returnPressed.connect(self._update_label_size)
        size.widget.setMaximumWidth(30)
        color.color_changed.connect(self._update_color)
        color.color = 'white'
        self._user_set_color = False
        c2.labels[0].setMinimumWidth(100)	# Indent second line of controls
        font.widget.menu().triggered.connect(self._update_font)
        style.widget.menu().triggered.connect(self._update_style)
        if 'Arial' in font_names:
            font.widget.setText('Arial')
        use_background.changed.connect(self._update_background_color)
        bg_color.color_changed.connect(self._update_background_color)
        bg_color.color = 'black'

    def _create_arrow_controls(self, parent):
        from chimerax.ui.widgets import EntriesRow, ColorButton
        EntriesRow(parent, 'Click and drag on graphics to create an arrow.  Drag arrow ends to reposition them.')
        styles = ('solid', 'blocky', 'pointy', 'pointer')
        c = EntriesRow(parent, 'Arrow', ('new', 'all'), 'Color', ColorButton, 'Weight', 1.0, 'Style', styles)
        self._arrow_menu, self._arrow_color, self._arrow_weight, self._arrow_style = \
            arrow_menu, color, weight, style = c.values
        amenu = arrow_menu.widget.menu()
        amenu.triggered.connect(self._arrow_menu_changed)
        amenu.aboutToShow.connect(self._fill_arrow_menu)
        color.color_changed.connect(self._update_arrow_color)
        color.color = 'white'
        self._user_set_arrow_color = False
        weight.widget.returnPressed.connect(self._update_arrow_weight)
        weight.widget.setMaximumWidth(30)
        style.widget.menu().triggered.connect(self._update_arrow_style)

    def _create_mouse_controls(self, parent):
        from chimerax.ui.widgets import EntriesRow
        ms = EntriesRow(parent, True, 'Use mouse to create or move labels and arrows')
        self._use_mouse = use_mouse = ms.values[0]
        use_mouse.changed.connect(self._update_use_mouse)
        self._update_use_mouse(True)

    def _create_hide_show_text(self, parent):
        from chimerax.ui.widgets import EntriesRow
        EntriesRow(parent, 'Hide or show labels and arrows using checkbuttons in the Models tool')

    def _shown_changed(self, shown):
        self._use_mouse.value = shown
        if shown and self._current_label is None:
            self._text.widget.setFocus()

    @property
    def _label_spec(self):
        label = self._current_label
        spec = '' if label == 'all' else (label.drawing.atomspec if label else None)
        return spec

    @property
    def _font_names(self):
        from Qt.QtGui import QFontDatabase
        names = QFontDatabase.families()
        return names

    def _label_menu_changed(self, action):
        value = action.text().split()[0]  # Show only the atomspec on the menu button
        self._label_menu.value = value
        if value == 'all':
            self._current_label = 'all'
            self._text.value = ''
        else:
            for label in self._all_labels:
                if label.drawing.atomspec == value:
                    if label is not self._current_label:
                        self._set_current_label(label)
                    break

    def _fill_label_menu(self):
        menu = self._label_menu.widget.menu()
        menu.clear()
        items = ['all'] + [f'{label.drawing.atomspec} {label.text}' for label in self._all_labels]
        for item in items:
            menu.addAction(item)

    @property
    def _all_labels(self):
        from .label2d import session_labels
        labels_model = session_labels(self.session)
        return labels_model.all_labels if labels_model else []
            
    def _update_label_text(self, text = None):
        label = self._current_label
        if label is None:
            self._create_new_label()
        elif label == 'all':
            pass  # Don't update text of all labels
        elif text != label.text:
            text = _quote_if_needed(self._text.value)
            self._run_command(f'2dlabel {self._label_spec} text {text}')

    def _create_new_label(self):
        text = self._text.value
        options = [f'text {_quote_if_needed(text)}']
        size = self._size.value
        if size != 24:
            options.append(f'size {size}')
        from chimerax.core.colors import color_name
        color = color_name(self._color.color) if self._user_set_color else None
        if color is not None:
            options.append(f'color {color}')
        x, y = ('%.3f' % self._label_position[0]), ('%.3f' % self._label_position[1])
        options.append(f'xpos {x} ypos {y}')
        font = self._font.value
        if font != 'Arial':
            options.append(f'font {_quote_if_needed(font)}')
        style = self._style.value
        if 'bold' in style:
            options.append('bold true')
        if 'italic' in style:
            options.append('italic true')
        if self._use_background.enabled:
            options.append(f'bgColor {self._background_color_name}')

        command = '2dlabel ' + ' '.join(options)
        label = self._run_command(command)
        self._current_label = label
        self._label_menu.value = label.drawing.atomspec

    def _update_label_size(self):
        label = self._current_label
        if label is None:
            return
        
        size = self._size.value
        if label == 'all' or size != label.size:
            self._run_command(f'2dlabel {self._label_spec} size {size}')
    
    def _update_color(self):
        self._user_set_color = True
        label = self._current_label
        if label is None:
            return

        color = self._color.color
        if label == 'all' or label.color is None or tuple(color) != tuple(label.color):
            from chimerax.core.colors import color_name
            cname = color_name(color)
            self._run_command(f'2dlabel {self._label_spec} color {cname}')
    
    def _update_background_color(self):
        label = self._current_label
        if label is None:
            return

        if self._use_background.enabled:
            self._run_command(f'2dlabel {self._label_spec} bgColor {self._background_color_name}')
        elif label == 'all' or label.background is not None:
            self._run_command(f'2dlabel {self._label_spec} bgColor none')

    @property
    def _background_color_name(self):
        if self._use_background.enabled:
            from chimerax.core.colors import color_name
            return color_name(self._background_color.color)
        return 'none'

    def _update_style(self):
        label = self._current_label
        if label is None:
            return

        style = self._style.value
        bold, italic = ('bold' in style), ('italic' in style)
        options = []
        if label == 'all' or bold != label.bold:
            options.append(f'bold {bold}')
        if label == 'all' or italic != label.italic:
            options.append(f'italic {italic}')
        if options:
            self._run_command(f'2dlabel {self._label_spec} ' + ' '.join(options))

    def _update_font(self):
        label = self._current_label
        if label is None:
            return

        font = self._font.value
        if label == 'all' or font != label.font:
            self._run_command(f'2dlabel {self._label_spec} font {_quote_if_needed(font)}')

    def _update_use_mouse(self, enable):
        triggers = self.session.triggers
        if enable:
            self._run_command('mousemode left "label or arrow"')
        else:
            m = self.session.ui.mouse_modes.mode('left')
            if m and m.name == 'label or arrow':
                self._run_command('mousemode left rotate')
        if enable:
            if self._move_label_handler is None:
                h = triggers.add_handler('move label', self._mouse_moved_label)
                self._move_label_handler = h
                triggers.add_handler('set mouse mode', self._mouse_mode_changed)
        elif self._move_label_handler:
            triggers.remove_handler(self._move_label_handler)
            self._move_label_handler = None

    def _mouse_mode_changed(self, tname, tdata):
        button, modifiers, mode = tdata
        if not self._use_mouse.enabled:
            return 'delete handler'
        if button == 'left' and mode.name != 'label or arrow':
            self._use_mouse.enabled = False
            return 'delete handler'
        
    def _mouse_moved_label(self, tname, xy_or_label):
        from .label2d import Label
        from .arrows import Arrow
        if isinstance(xy_or_label, Label):
            label = xy_or_label
            if label is self._current_label:
                self._label_position = (label.xpos, label.ypos)
                if label.text == 'New label':
                    self._text.widget.setFocus()
            else:
                self._set_current_label(label)
        elif isinstance(xy_or_label, tuple):
            # Label move clicked on empty space.  Set current x,y to clicked position.
            self._label_position = tuple(xy_or_label)
            self._start_new_label()
        elif isinstance(xy_or_label, Arrow):
            arrow = xy_or_label
            if arrow is not self._current_arrow:
                self._set_current_arrow(arrow)

    def _start_new_label(self):
        self._current_label = None
        text = self._text
        text.value = 'New label'
        self._update_label_text('New label')  # Make sure text update gets called even if text was already "New label"
        text.widget.setFocus()
        text.widget.selectAll()

    def _set_current_label(self, label):
        self._label_menu.value = label.drawing.atomspec
        self._current_label = label
        self._text.value = label.text
        if label.text == 'New label':
            text = self._text.widget
            text.selectAll()
            text.setFocus()
        self._size.value = label.size
        self._color.color = 'white' if label.color is None else label.color
        self._label_position = (label.xpos, label.ypos)
        self._font.value = label.font
        style = ('bold italic' if label.italic else 'bold') if label.bold else ('italic' if label.italic else 'normal')
        self._style.value = style
        if label.background is None:
            self._use_background.enabled = False
        else:
            self._background_color.color = label.background
            self._use_background.enabled = True

    def _set_current_arrow(self, arrow):
        self._arrow_menu.value = arrow.drawing.atomspec
        self._current_arrow = arrow
        self._arrow_color.color = 'white' if arrow.color is None else arrow.color
        self._arrow_weight.value = arrow.weight
        self._arrow_style.value = arrow.head_style
        self._set_arrow_mouse_mode_defaults()
    
    @property
    def _arrow_spec(self):
        arrow = self._current_arrow
        spec = '' if arrow == 'all' else (arrow.drawing.atomspec if arrow else None)
        return spec

    def _arrow_menu_changed(self):
        value = self._arrow_menu.value
        if value == 'all':
            self._current_arrow = 'all'
        else:
            for arrow in self._all_arrows:
                if arrow.drawing.atomspec == value:
                    if arrow is not self._current_arrow:
                        self._set_current_arrow(arrow)
                    break

    def _fill_arrow_menu(self):
        menu = self._arrow_menu.widget.menu()
        menu.clear()
        items = ['all'] + [arrow.drawing.atomspec for arrow in self._all_arrows]
        for item in items:
            menu.addAction(item)

    @property
    def _all_arrows(self):
        from .arrows import all_arrows
        return all_arrows(self.session)
    
    def _update_arrow_color(self):
        self._set_arrow_mouse_mode_defaults()
        self._user_set_arrow_color = True
        arrow = self._current_arrow
        if arrow is None:
            return

        color = self._arrow_color.color
        if arrow == 'all' or arrow.color is None or tuple(color) != tuple(arrow.color):
            from chimerax.core.colors import color_name
            cname = color_name(color)
            self._run_command(f'2dlabel arrow {self._arrow_spec} color {cname}')
            
    def _update_arrow_weight(self):
        self._set_arrow_mouse_mode_defaults()
        arrow = self._current_arrow
        if arrow is None:
            return
        
        weight = self._arrow_weight.value
        if arrow == 'all' or weight != arrow.weight:
            self._run_command(f'2dlabel arrow {self._arrow_spec} weight {weight}')

    def _update_arrow_style(self):
        self._set_arrow_mouse_mode_defaults()
        arrow = self._current_arrow
        if arrow is None:
            return

        style = self._arrow_style.value
        if arrow == 'all' or style != arrow.head_style:
            self._run_command(f'2dlabel arrow {self._arrow_spec} headStyle {style}')

    def _set_arrow_mouse_mode_defaults(self):
        arrow_mode = self.session.ui.mouse_modes.named_mode('label or arrow')
        if arrow_mode:
            arrow_mode.color = self._arrow_color.color if self._user_set_arrow_color else None
            arrow_mode.weight = self._arrow_weight.value
            arrow_mode.style = self._arrow_style.value

    def _run_command(self, command, delay_time = 1.0):
        self._delay_logging_command(command, delay_time)
        from chimerax.core.commands import run
        return run(self.session, command, log = False)

    def _delay_logging_command(self, command, delay_time = 1.0):
        self._delayed_command = command
        t = self._delayed_log_timer
        if t is None:
            self._delayed_log_timer = self.session.ui.timer(1000*delay_time, self._log_delayed_command)
        else:
            t.start(int(1000*delay_time))  # Reset delay time

    def _log_delayed_command(self):
        cmd = self._delayed_command
        if cmd:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)
            self._delayed_command = None
        self._delayed_log_timer = None

    def _delete_label(self):
        label_spec = self._label_spec
        if label_spec:
            self._run_command(f'2dlabel delete {label_spec}')
            self._current_label = None

    def _delete_arrow(self):
        arrow_spec = self._arrow_spec
        if arrow_spec:
            self._run_command(f'2dlabel delete {arrow_spec}')
            self._current_arrow = None

    def _show_arrows_gui(self):
        self._run_command('ui tool show Arrows')
        
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

def _quote_if_needed(text):
    if ' ' in text or text == '':
        return f'"{text}"'
    return text
