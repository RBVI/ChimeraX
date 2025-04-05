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

from chimerax.core.tools import ToolInstance
from chimerax.ui.widgets import ItemTable
from chimerax.core.commands import run
from chimerax.core.models import REMOVE_MODELS
from Qt.QtWidgets import QStyledItemDelegate, QComboBox, QAbstractItemView, QVBoxLayout, QStyle, QStyleOptionComboBox
from Qt.QtCore import Qt, QTimer

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
        self.struct_table = ItemTable(session=self.session)
        self.table_setup()
        self.handlers = []
        self.add_handlers()
        self.tool_window.manage('side')

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
        self.struct_table.add_column('Rating', lambda s: s.viewdockx_data.get('Rating', "2"),
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


class RatingDelegate(QStyledItemDelegate):
    """
    A delegate that provides a QComboBox editor for editing ratings in a table view.

    The RatingDelegate class is responsible for rendering a combo box in the table view and handling the interaction
    between the editor widget (QComboBox) and the model. It ensures that the combo box is displayed correctly both
    when the cell is selected and not selected, and that the data is properly committed to the model when editing
    is finished.

    Methods:
        createEditor(parent, option, index):
            Creates and returns a QComboBox editor for the delegate.

        setEditorData(editor, index):
            Sets the data from the model into the QComboBox editor.

        setModelData(editor, model, index):
            Sets the data from the QComboBox editor back into the model.

        updateEditorGeometry(editor, option, index):
            Updates the geometry of the QComboBox editor to match the item.

        paint(painter, option, index):
            Renders the combo box UI in the table view, ensuring the text is displayed correctly.
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
        Set the data from the QComboBox editor back into the ChimeraX Model (not the QAbstractItemModel). The ItemTable
        will always call the paint method of the delegate using its own data_fetch which accesses attributes from a
        chimerax model, not a qt table model. We need to set the data on the AtomicStructure and let the ItemTable
        handle giving this delegates paint the correct data.

        Args:
            editor: The editor widget (QComboBox).
            model: The model to set the data into.
            index: The index of the item in the model.
        """
        # Get the structure (chimerax Structure) from the table row.
        structure = self.parent().data[index.row()]
        new_rating = editor.currentText()
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
            # If the cell is currently being edited, skip custom painting
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

        # === Background Handling (Ensures Selection Highlights Work) ===

        if option.state & QStyle.State_Selected:
            # If the cell is selected, use the highlighted background color
            painter.fillRect(option.rect, option.palette.highlight())
            combo_style.state |= QStyle.State_Selected  # Mark the combo box as selected
            painter.setPen(option.palette.highlightedText().color())  # Set text color to contrast selection
        else:
            # If the cell is not selected, use the default background color
            painter.fillRect(option.rect, option.palette.base())
            painter.setPen(option.palette.text().color())  # Set text color to default

        # === Drawing the Combo Box Frame and Arrow ===
        # This renders the full combo box frame (border + drop-down arrow) inside the cell
        style.drawComplexControl(QStyle.CC_ComboBox, combo_style, painter)

        # === Drawing the Text (Prevents Overlapping Artifacts) ===
        # Get the rectangle where the text should be placed inside the combo box
        text_rect = style.subControlRect(QStyle.CC_ComboBox, combo_style, QStyle.SC_ComboBoxEditField, None)

        # Draw the text manually to avoid rendering issues (double text, overlapping numbers, etc.)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, combo_style.currentText)
