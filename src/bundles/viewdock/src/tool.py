# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic import AtomicStructure
from chimerax.core.tools import ToolInstance
from chimerax.hbonds.gui import HBondsGUI
from chimerax.ui.widgets import ItemTable
from chimerax.core.commands import run, concise_model_spec
from chimerax.core.models import REMOVE_MODELS
from Qt.QtWidgets import (QStyledItemDelegate, QComboBox, QAbstractItemView, QVBoxLayout, QStyle, QStyleOptionComboBox,
                          QHBoxLayout, QPushButton, QDialog, QDialogButtonBox)
from Qt.QtCore import Qt


class ViewDockTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name, structures):
        super().__init__(session, tool_name)
        self.display_name = "ViewDock"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)

        # Create a vertical layout for the tool
        self.main_v_layout = QVBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_v_layout)

        self.structures = structures

        self.top_buttons_layout = QHBoxLayout()
        self.top_buttons_setup()
        self.main_v_layout.addLayout(self.top_buttons_layout)


        self.struct_table = ItemTable(session=self.session)
        self.table_setup()
        self.handlers = []
        self.add_handlers()
        self.tool_window.manage('side')

    def top_buttons_setup(self):
        """
        Create the top buttons for the tool (HBonds).
        """
        self.hbonds_button = QPushButton("HBonds")
        self.hbonds_button.clicked.connect(self.hbonds_callback)
        self.top_buttons_layout.addWidget(self.hbonds_button)

    def hbonds_callback(self):
        """
        Callback function for the HBonds button click.

        This method creates a popup dialog containing the HBondsGUI widget for configuring hydrogen bond analysis.
        The dialog includes "Ok" and "Cancel" buttons. When "Ok" is clicked, the method retrieves the command
        from the HBondsGUI, constructs a command string to perform hydrogen bond analysis, and executes it.

        The analysis is restricted to the current set of binding analysis structures against any other AtomicStructures.
        """

        # Create the HBondsGUI instance
        hbonds_gui = HBondsGUI(self.session, show_model_restrict=False, show_bond_restrict=False)

        # Create a QDialog to act as the popup
        dialog = QDialog(self.tool_window.ui_area)
        dialog.setWindowTitle("HBonds")

        # Set the layout for the dialog
        layout = QVBoxLayout(dialog)
        dialog.setLayout(layout)

        # Add the HBondsGUI widget to the dialog's layout
        layout.addWidget(hbonds_gui)

        # Add Ok/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        # Connect the Ok button to call hbonds_gui.get_command()
        def on_ok():
            command = hbonds_gui.get_command()
            # Binding analysis structures
            mine = concise_model_spec(self.session, self.structures)
            all_structures = self.session.models.list(type=AtomicStructure)
            # All structures that are AtomicStructures but not in the binding analysis structures
            others = concise_model_spec(self.session, set(all_structures) - set(self.structures))

            # command[0] = command name ('hbonds'), command[1] = model selection (omit and force our binding structures),
            # command[2] = all other keyword arguments from hbonds gui
            run(self.session, f"{command[0]} {mine} restrict {others} {command[2]}")
            dialog.accept()

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(dialog.reject)

        # Show the dialog
        dialog.exec()

    def table_setup(self):
        """
        Create the ItemTable for the structures. Add a column for the display with check boxes, a column for the
        structure ID, a column for the Rating with a custom delegate, and columns for each key in the viewdockx_data
        attribute of each structure (e.g., Name, Description, Energy Score...). If a structure does not have a key from
        the set, the cell will be empty.
        """

        # Fixed columns. Generic based on ChimeraX model attributes.
        self.struct_table.add_column('Show', lambda s: s.display, data_set=self.set_visibility, format=ItemTable.COL_FORMAT_BOOLEAN)
        self.struct_table.add_column('ID', lambda s: s.id_string)

        # Custom Rating delegate
        delegate = RatingDelegate(self.struct_table)  # Create the delegate instance
        self.struct_table.add_column('Rating', lambda s: s.viewdockx_data.get('Rating', 2),
                                     data_set = lambda item, value: None,
                                     editable=True)

        # Associate the delegate with the "Rating" column
        rating_column_index = self.struct_table.column_names.index('Rating')
        self.struct_table.setItemDelegateForColumn(rating_column_index, delegate)

        # Set an edit trigger for the table whenever the current selected item changes. Prevents having to click through
        # multiple selections to edit the rating of a structure.
        self.struct_table.setEditTriggers(QAbstractItemView.EditTrigger.CurrentChanged)

        # Collect all unique keys from viewdockx_data of all structures and add them as columns
        viewdockx_keys = set()
        for structure in self.structures:
            viewdockx_keys.update(structure.viewdockx_data.keys())
        for key in viewdockx_keys:
            self.struct_table.add_column(key, lambda s, k=key: s.viewdockx_data.get(k, ''))

        # Set the data for the table and launch it
        self.struct_table.data = self.structures
        self.struct_table.launch()

        # Add the table to the layout
        self.main_v_layout.addWidget(self.struct_table)

    def add_handlers(self):
        """
        Add trigger handlers for updating the structure table.
        """
        self.handlers.append(self.session.triggers.add_handler(
            REMOVE_MODELS,
            lambda trigger_name, trigger_data: self.remove_models_cb(trigger_name, trigger_data)
        ))

    def remove_models_cb(self, trigger_name, trigger_data):
        """
        Callback for when models are removed from the session. Removes the models from the structure attribute and
        updates the table.

        Args:
            trigger_name: The name of the trigger that was activated. Expecting REMOVE_MODELS.
            trigger_data: The data that was passed with the trigger. Expecting a list of structures.
        """
        if trigger_name != REMOVE_MODELS:
            return

        for model in trigger_data:
            if model in self.structures:
                self.structures.remove(model)
        self.struct_table.data = self.structures

    def set_visibility(self, structure, value):
        """
        Callback for when the model display column has changes. Shows or hides the structure based on the value.

        Args:
            structure: The structure that is being changed.
            value: The new value for the display column. True/False for show/hide.
        """
        if value:
            run(self.session, f'show #{structure.id_string} models')
        else:
            run(self.session, f'hide #{structure.id_string} models')

    def delete(self):
        """
        Remove all trigger handlers from the tool before deleting.
        """
        for handler in self.handlers:
            self.session.triggers.remove_handler(handler)
        super().delete()

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'structures': self.structures,
            'tool_name': self.tool_name
        }

    @classmethod
    def restore_snapshot(cls, session, snapshot):
        if snapshot['version'] != 1:
            session.logger.warning("Incompatible snapshot version for ViewDock tool.")
            return None
        tool = cls(session, snapshot['tool_name'], snapshot['structures'])
        return tool

class RatingDelegate(QStyledItemDelegate):
    """
    A delegate that provides a QComboBox editor for editing ratings in a table view.

    The RatingDelegate class is responsible for rendering a combo box in the table view and handling the interaction
    between the editor widget (QComboBox) and the model (QAbstractItemModel). It ensures that the combo box is displayed
    correctly both when the cell is selected and not selected, and that the data is properly synced to the
    AtomicStructure when editing is finished.
    """

    def __init__(self, parent=None):
        """
        Initialize the RatingDelegate with a list of items for the QComboBox.

        Args:
            parent: The parent widget for the delegate.
        """
        super().__init__(parent)
        self.items = ["1", "2", "3"]  # or ["Red", "Yellow", "Green"]

    def createEditor(self, parent, option, index):
        """
        Create and return a QComboBox editor for the delegate.

        Args:
            parent: The parent widget for the editor.
            option: The style options for the item.
            index: The index of the item in the model.

        Returns:
            QComboBox: The editor widget (QComboBox).
        """
        editor = QComboBox(parent)
        editor.addItems(self.items)
        return editor

    def setEditorData(self, editor, index):
        """
        Set the data from the model into the QComboBox editor.

        Args:
            editor: The editor widget (QComboBox).
            index: The index of the item in the model.
        """
        value = index.data(Qt.EditRole) or index.data(Qt.DisplayRole)
        i = editor.findText(str(value))
        if i >= 0:
            editor.setCurrentIndex(i)

    def setModelData(self, editor, model, index):
        """
        Set the data from the QComboBox editor back into both the ChimeraX Model and the QAbstractItemModel.

        The ItemTable will always call the paint method of the delegate using its own data_fetch which accesses attributes
        from a ChimeraX model, not a Qt table model. We need to set the data on the AtomicStructure and let the ItemTable
        handle giving this delegate's paint method the correct data.

        Args:
            editor: The editor widget (QComboBox).
            model: The QAbstractItemModel to set the data into.
            index: The index of the item in the model.
        """
        # Get the structure (chimerax Structure) from the table row.
        structure = self.parent().data[index.row()]
        new_rating = int(editor.currentText())
        structure.viewdockx_data['Rating'] = new_rating  # Update the rating in the structure's data

        model.setData(index, new_rating)  # Optionally, set the value in the model too. This is for Qt completeness


    def updateEditorGeometry(self, editor, option, index):
        """
        Update the geometry of the QComboBox editor to match the item.

        Args:
            editor: The editor widget (QComboBox).
            option: The style options for the item.
            index: The index of the item in the model.
        """
        editor.setGeometry(option.rect)

    def paint(self, painter, option, index):
        """Always display a combo box UI rendering in the table, ensuring proper background and selection handling."""

        # Get the table view widget (the parent of this delegate)
        view = option.widget

        # Ensure that we do not paint over an actively edited combo box
        if view and hasattr(view, 'indexWidget'):
            # If the cell is currently being edited, skip custom painting. The editor will be open and handle its own
            # painting.
            if view.state() == QAbstractItemView.EditingState and view.currentIndex() == index:
                return

        # Retrieve the value to be displayed in the combo box
        # First, check the EditRole (active editing value); if unavailable, fall back to DisplayRole (stored value)
        value = index.data(Qt.EditRole) or index.data(Qt.DisplayRole)

        # Get the style of the widget to use for drawing
        style = option.widget.style()

        # Initialize a QStyleOptionComboBox to mimic the appearance of a real combo box
        combo_style = QStyleOptionComboBox()
        combo_style.rect = option.rect  # Set the position and size of the combo box
        combo_style.currentText = str(value) if value else "2"  # Ensure there is always some text displayed
        combo_style.state = QStyle.State_Enabled  # Enable the combo box appearance by default

        if option.state & QStyle.State_Selected:
            # If the cell is selected, use the highlighted background color
            painter.fillRect(option.rect, option.palette.highlight())
            combo_style.state |= QStyle.State_Selected  # Mark the combo box as selected
            painter.setPen(option.palette.highlightedText().color())  # Set text color to contrast selection
        else:
            # If the cell is not selected, use the default background color
            painter.fillRect(option.rect, option.palette.base())
            painter.setPen(option.palette.text().color())  # Set text color to default

        # This renders the full combo box frame (border + drop-down arrow) inside the cell
        style.drawComplexControl(QStyle.CC_ComboBox, combo_style, painter)

        # Get the rectangle where the text should be placed inside the combo box
        text_rect = style.subControlRect(QStyle.CC_ComboBox, combo_style, QStyle.SC_ComboBoxEditField, None)

        # Draw the text
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, combo_style.currentText)
