# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def buttonpanel(session, title, add = None, command = None, row = None, column = None,
                rows = None, columns = None, fill = None):
    '''
    Create a custom user interface panel containing buttons which run specified commands.
    Use the command multiple times with the same title to add each button to the panel.

    Parameters
    ----------
    title : string
      Name of the button panel shown on its titlebar.
    add : string
      Name of a button to add to the panel.  This text appears on the button.
    command : string
      Command to run when the button is pressed.
    row : integer
      Which row to place the button in.  Row 1 is at the top and row numbers increase downward.
    column : integer
      Which column to place the button in.  Column 1 is at the left and column numbers increase rightward.
    rows : integer
      How many rows the panel will have.  Use in combination with the "fill" option for placing buttons
      without specifying a row and column for each button.
    columns : integer
      How many columns the panel will have.  Use in combination with the "fill" option for placing buttons
      without specifying a row and column for each button.
    fill : "rows" or "columns"
      When adding a button to a panel fill rows first of columns first.
    '''

    bp = _button_panel_with_title(session, title)

    if rows is not None or columns is not None:
        bp.set_grid_size(rows, columns)

    if fill is not None:
        bp.set_fill_order(fill)

    if add is not None:
        if command is None:
            from chimerax.core.errors import UserError
            raise UserError('No command specified for button "%s"' % add)
        bp.add_button(name = add, command = command, row = row, column = column)
    elif command is not None:
        from chimerax.core.errors import UserError
        raise UserError('No button name specified for command "%s".\nUse the "add" option to specify a button name' % command)
        
# ------------------------------------------------------------------------------
#
def register_buttonpanel_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, IntArg, EnumOf
    desc = CmdDesc(required = [('title', StringArg)],
                   keyword = [('add', StringArg),
                              ('command', StringArg),
                              ('row', IntArg),
                              ('column', IntArg),
                              ('rows', IntArg),
                              ('columns', IntArg),
                              ('fill', EnumOf(('rows', 'columns')))],
                   synopsis = 'Create a user interface button panel.')
    register('buttonpanel', desc, buttonpanel, logger=logger)

# ------------------------------------------------------------------------------
#
def _button_panel_with_title(session, title):
    bps = _button_panels(session)
    for bp in bps:
        if bp.title == title:
            return bp

    bp = ButtonPanel(session, title)
    session._button_panels.append(bp)
    return bp

# ------------------------------------------------------------------------------
#
def _button_panels(session):
    if not hasattr(session, '_button_panels'):
        session._button_panels = []
    return session._button_panels

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance

class ButtonPanel(ToolInstance):

    def __init__(self, session, title):
        ToolInstance.__init__(self, session, title)

        self.title = title
        self._rows = None
        self._columns = None
        self._next_row_col = (1,1)
        self._fill_order = 'rows'
        self._buttons = {}		# Map (row,column) -> QPushButton

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        from Qt.QtWidgets import QGridLayout
        self._layout = layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)
        tw.manage(placement="side")

    def add_button(self, name, command, row=None, column=None):
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QPushButton
        from Qt.QtCore import Qt
        b = QPushButton(name, parent)
        b.clicked.connect(lambda e, cmd=command: self._run_command(command=cmd))
        b.name = name
        b.command = command
        b.setToolTip(command)
        r, c = self._placement_row_column(row, column)
        self._layout.addWidget(b, r, c, Qt.AlignCenter)
        self._buttons[(r,c)] = b
        self._next_row_col = self._next_position(r,c)

    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)
        
    def _placement_row_column(self, row, column):
        if row is not None and column is not None:
            b = self._buttons.get((row,column), None)
            if b:
                self._layout.removeWidget(b)
            return (row, column)  

        if row is None and column is None:
            return self._next_unfilled_position()
            
        if row is not None:
            c = 1
            while (row, c) in self._buttons:
                c += 1
            return (row, c)
            
        if column is not None:
            r = 1
            while (r, column) in self._buttons:
                r += 1
            return (r, column)

    def _next_unfilled_position(self):
        r,c = self._next_row_col
        while (r,c) in self._buttons:
            (r,c) = self._next_position(r,c)
        return (r,c)

    def _next_position(self, r, c):
        if self._fill_order == 'rows':
            nr,nc = (r, c+1)
            if self._columns is not None and nc > self._columns:
                nr,nc = (r+1, 1)
        elif self._fill_order == 'columns':
            nr,nc = (r+1, c)
            if self._rows is not None and nr > self._rows:
                nr,nc = (1, c+1)
        return nr,nc
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def set_grid_size(self, rows, columns):
        if rows is not None:
            self._rows = rows
        if columns is not None:
            self._columns = columns

    def set_fill_order(self, fill):
        self._fill_order = fill

    SESSION_SAVE = True
    
    # Session saving.
    def take_snapshot(self, session, flags):
        buttons = [(r,c,b.name,b.command) for (r,c),b in self._buttons.items()]
        data = {
            "super": super().take_snapshot(session, flags),
            'title': self.title,
            'rows': self._rows,
            'columns': self._columns,
            'fill_order': self._fill_order,
            'buttons': buttons,
        }
        return data

    # Session restore
    @classmethod
    def restore_snapshot(cls, session, data):
        bp = ButtonPanel(session, data['title'])
        ToolInstance.set_state_from_snapshot(bp, session, data["super"])
        bp.set_grid_size(data['rows'], data['columns'])
        bp.set_fill_order(data['fill_order'])
        for r,c,name,command in data['buttons']:
            bp.add_button(name, command, row = r, column = c)
        return bp
