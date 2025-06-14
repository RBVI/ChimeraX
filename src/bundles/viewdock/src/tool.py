# vim: set expandtab ts=4 sw=4:

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
from PyQt6.QtWidgets import QMenu

from chimerax.atomic import AtomicStructure
from chimerax.ui import MainToolWindow
from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings
from chimerax.hbonds.gui import HBondsGUI
from chimerax.clashes.gui import ClashesGUI
from chimerax.ui.widgets import ItemTable
from chimerax.core.commands import run, concise_model_spec
from chimerax.core.models import REMOVE_MODELS, MODEL_DISPLAY_CHANGED
from Qt.QtWidgets import (QStyledItemDelegate, QComboBox, QAbstractItemView, QVBoxLayout, QStyle, QStyleOptionComboBox,
                          QHBoxLayout, QPushButton, QDialog, QDialogButtonBox, QGroupBox, QGridLayout, QLabel, QWidget,)
from Qt.QtGui import QFont
from Qt.QtCore import Qt
from chimerax.viewdock import RATING_KEY, DEFAULT_RATING


class ViewDockTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    registered_mousemode = False

    def __init__(self, session, tool_name, structures):
        """
        Initialize the ViewDock tool with, table, table controls, and model descriptions.

        Args:
            session: The ChimeraX session.
            tool_name (str): The name of the tool. Used for saving settings.
            structures: A list of structures to display in the tool. Structures must have the .viewdock_data attribute
            with at minimum a rating key. ex {'rating': 2, 'name': 'Docked Structure 1', 'energy_score': -5.0, ...}
        """

        super().__init__(session, tool_name)
        self.display_name = "ViewDock"

        self.tool_window = MainToolWindow(self)

        # Create a vertical layout for the tool
        self.main_v_layout = QVBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_v_layout)

        self.structures = self.filter_structures(structures)

        self.top_buttons_layout = QHBoxLayout()
        self.top_buttons_setup()
        self.main_v_layout.addLayout(self.top_buttons_layout)

        self.settings = ViewDockSettings(self.session, tool_name)

        self.col_display_widget = QWidget()
        self.struct_table = ItemTable(session=self.session, column_control_info=(
            self.col_display_widget, self.settings, {}, True, None, None, True
        ))
        self.table_setup()

        from .mousemode import register_mousemode, NextDockingMouseMode
        if not self.__class__.registered_mousemode:
            register_mousemode(session)
            self.__class__.registered_mousemode = True
        NextDockingMouseMode.vd_instance = self

        self.description_group = QGroupBox()
        self.description_box_setup()



        self.handlers = []
        self.add_handlers()
        self.tool_window.manage('side')

    def filter_structures(self, structures):
        """
        Ensure that ViewDockX structures have the viewdock_data attribute that the ViewDock tool expects.

        Args:
            structures (list): A list of structures.

        Returns:
            list: Only structures that have the viewdock_data attribute.
        """
        for structure in structures:
            if hasattr(structure, 'viewdockx_data'):
                structure.register_attr(self.session, "viewdock_data", "ViewDock")
                structure.viewdock_data = structure.viewdockx_data.copy()
        return [structure for structure in structures if hasattr(structure, 'viewdock_data')]

    def top_buttons_setup(self):
        """
        Create the top buttons for the tool (HBonds and Clashes).
        """
        self.hbonds_button = QPushButton("HBonds")
        self.hbonds_button.clicked.connect(
            lambda: self.popup_callback(
                HBondsGUI, "HBonds", show_model_restrict=False, show_bond_restrict=False
            )
        )
        self.top_buttons_layout.addWidget(self.hbonds_button)

        self.clashes_button = QPushButton("Clashes")
        self.clashes_button.clicked.connect(
            lambda: self.popup_callback(ClashesGUI, "Clashes", has_apply_button=False, show_restrict=False)
        )
        self.top_buttons_layout.addWidget(self.clashes_button)

        self.top_buttons_layout.setAlignment(Qt.AlignLeft)

    def popup_callback(self, gui_class, popup_name, **kwargs):
        """
        Generalized callback function for creating a popup dialog using a specified GUI widget class. This callback
        can be connected to buttons that are supposed to open a dialog for a specific task
        (e.g., HBonds, Clashes...). The GUI Widget class must have a .get_command() implementation that returns a cl
        command that will be run when the OK button is clicked in the dialog.

        Args:
            gui_class: The GUI class to instantiate (e.g., HBondsGUI, ClashesGUI). The class is automatically passed
                the session in its constructor.
            popup_name: The command name to execute (e.g., "hbonds", "clashes").
            **kwargs: Additional keyword arguments to pass to the GUI class constructor. Session is passed to all GUI
                class constructors automatically and should not be specified in this list
        """
        gui_instance = gui_class(self.session, **kwargs)

        # Create a QDialog to act as the popup
        dialog = QDialog(self.tool_window.ui_area)
        dialog.setWindowTitle(f"{self.display_name} {popup_name.capitalize()}")
        layout = QVBoxLayout(dialog)
        dialog.setLayout(layout)

        # Add the GUI widget to the dialog's layout
        layout.addWidget(gui_instance)

        # Add Ok/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        # Connect the Ok button to call gui_instance.get_command()
        def ok_cb():
            # Default behavior for chimerax.ui.widgets
            command = gui_instance.get_command()
            # Binding analysis structures
            mine = concise_model_spec(self.session, self.structures)
            all_structures = self.session.models.list(type=AtomicStructure)
            # All structures that are AtomicStructures but not in the binding analysis structures
            others = concise_model_spec(self.session, set(all_structures) - set(self.structures))
            if others == "#":
                self.session.logger.warning(f"First open a model for {popup_name.capitalize()}.")
            else:
                # command[0] = command name, command[1] = model selection, command[2] = other arguments
                run(self.session, f"{command[0]} {mine} restrict {others} {command[2]}")
            dialog.accept()

        button_box.accepted.connect(ok_cb)
        button_box.rejected.connect(dialog.reject)

        # Show the dialog
        dialog.exec()

    def table_setup(self):
        """
        Create the ItemTable for the structures. Add a for the
        structure ID, a column for the Rating with a custom delegate, and columns for each key in the viewdock_data
        attribute of each structure (e.g., Name, Description, Energy Score...). If a structure does not have a key from
        the set, the cell will be empty.
        """

        table_group = QGroupBox()
        table_group_layout = QVBoxLayout()

        table_group_layout.addWidget(self.struct_table)
        table_group_layout.addWidget(self.col_display_widget)
        table_group.setLayout(table_group_layout)

        # Fixed columns. Generic based on ChimeraX model attribute(s).
        self.struct_table.add_column('ID', lambda s: s.id_string)

        # Custom Rating delegate
        delegate = RatingDelegate(self.struct_table)  # Create the delegate instance
        self.struct_table.add_column('Rating', lambda s: s.viewdock_data.get(RATING_KEY),
                                     data_set = lambda item, value: None,
                                     editable=True)

        # Associate the delegate with the "Rating" column
        rating_column_index = self.struct_table.column_names.index('Rating')
        self.struct_table.setItemDelegateForColumn(rating_column_index, delegate)

        # Set an edit trigger for the table whenever the current selected item changes. Prevents having to click through
        # multiple selections to edit the rating of a structure.
        self.struct_table.setEditTriggers(QAbstractItemView.EditTrigger.CurrentChanged)

        # Collect all unique keys from viewdock_data of all structures and add them as columns
        viewdockx_keys = set()
        for structure in self.structures:
            viewdockx_keys.update(structure.viewdock_data.keys())
        for key in viewdockx_keys:
            if key == RATING_KEY:
                # Rating is already added as a column with a custom delegate, skip it here
                continue
            self.struct_table.add_column(key, lambda s, k=key: s.viewdock_data.get(k, ''))

        # Set the data for the table and launch it
        self.struct_table.data = self.structures
        self.struct_table.launch()

        # Add the table group to the layout
        self.main_v_layout.addWidget(table_group)

    def description_box_setup(self):
        """
        Build the description box at the bottom of the tool which displays all the docking attribute information
        for a selected docking model.
        """

        # Create a group box for the description box
        description_layout = QGridLayout()
        self.description_group.setLayout(description_layout)

        # Set the title alignment to center
        self.description_group.setAlignment(Qt.AlignCenter)

        # Customize the font for the title
        title_font = QFont()
        title_font.setPointSize(16)  # Set font size
        self.description_group.setFont(title_font)

        self.struct_table.selection_changed.connect(
            lambda newly_selected, newly_deselected: self.table_selection_changed(newly_selected, newly_deselected)
        )

        # Add the group box to the main layout
        self.main_v_layout.addWidget(self.description_group)

        if len(self.structures) > 0:
            # Select the first structure in the table to display its data in the description box
            self.struct_table.selected = [self.structures[0]]

    def table_selection_changed(self, newly_selected, newly_deselected):
        """
        Callback for when the selection in the ItemTable changes. Updates what docking structure is visible and the
        description box with the newly selected structure's data.

        Args:
            newly_selected (list): The newly selected structure(s) in the ItemTable.
            newly_deselected (list): The previously selected structure(s) in the ItemTable.
        """
        self.update_structure_displays(newly_selected)
        self.update_model_description(newly_selected)

    def update_structure_displays(self, newly_selected):
        """
        Update the display of the docking structures. The newly selected structures will be shown, and all other docking
        structures will be hidden.

        Args:
            newly_selected (list): The newly selected structure(s) in the ItemTable.
        """
        # Generate arrays for newly selected and the difference
        newly_selected_array = newly_selected
        others_array = [s for s in self.structures if s not in newly_selected_array]

        # Call set_visibility to show newly selected and hide others
        self.set_visibility(newly_selected_array, True)
        self.set_visibility(others_array, False)

    def update_model_description(self, newly_selected):
        """
        Update the description box with the most recently selected structure's data. If more than one structure is
        newly selected, only the first one will be displayed.

        Args:
            newly_selected (list): The newly selected structure(s) in the ItemTable.
        """
        # Create a custom font for the labels
        label_font = QFont()
        label_font.setPointSize(12)  # Set the font size

        if not newly_selected:
            return
        docking_structure = newly_selected[0]
        self.description_group.setTitle(f"ChimeraX Model {docking_structure.atomspec}")

        # Clear the existing layout
        layout = self.description_group.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                if isinstance(widget, QLabel):
                    # Clear the text of QLabel widgets. This prevents layering of old/new widgets if the QT event loop
                    # doesn't delete the old widgets before an event loop pause like opening the context menu.
                    widget.clear()
                widget.deleteLater()

        # Add attributes in a grid layout
        attributes = list(docking_structure.viewdock_data.items())
        total_attributes = len(attributes)
        rows_per_column = (total_attributes + 1) // 2  # Divide attributes evenly over two columns

        for index, (key, value) in enumerate(attributes):
            # Use the column's data_fetch to get the value for attributes appearing in the table
            column = next((col for col in self.struct_table.columns if col.title == key), None)
            if column and column.data_fetch:
                # Fetch the value using the column's data_fetch
                if callable(column.data_fetch):
                    value = column.data_fetch(docking_structure)
                else:
                    # If data_fetch wasn't initialized as a callable, assume data_fetch is a string representing an
                    # attribute path
                    value = docking_structure
                    for attr in column.data_fetch.split('.'):
                        # Loop through nested attributes
                        value = getattr(value, attr, None)
                        if value is None:
                            break

            row = index % rows_per_column
            col = (index // rows_per_column) * 2  # Multiply by 2 to account for key-value pairs

            # Add key label
            key_label = QLabel(f"<b>{key}:</b>") # Use HTML to bold the attr name
            key_label.setFont(label_font)
            layout.addWidget(key_label, row, col)

            # Add value label
            value_label = QLabel(str(value))
            value_label.setFont(label_font)
            layout.addWidget(value_label, row, col + 1)

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
        if not self.structures:
            self.delete()
        else:
            self.struct_table.data = self.structures

    def set_visibility(self, structs, value):
        """
        Shows or hides the structure(s) based on the value.

        Args:
            structs: The structure(s) that is/are being changed. Can be a single structure or a list of structures.
            value: The new value for the display column. True/False for show/hide.
        """
        # Ensure structs is a list
        if not isinstance(structs, (list, tuple)):
            structs = [structs]

        model_spec = concise_model_spec(self.session, structs)
        if value:
            run(self.session, f'show {model_spec} models', log=False)
        else:
            run(self.session, f'hide {model_spec} models', log=False)

    def delete(self):
        """
        Remove all trigger handlers from the tool before deleting.
        """
        for handler in self.handlers:
            self.session.triggers.remove_handler(handler)
        from .mousemode import NextDockingMouseMode
        NextDockingMouseMode.vd_instance = None
        super().delete()

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'structures': self.structures,
            'tool_name': self.tool_name
        }

    @classmethod
    def restore_snapshot(cls, session, snapshot):
        """
        Restore snapshots for the ViewDock tool and the old ViewDockX tool.
        """
        # ViewDockX snapshots
        if '_html_state' in snapshot and 'vdxtable' in snapshot['_html_state']['name']:
            if snapshot['version'] != 2:
                session.logger.warning(
                    f"Incompatible ViewDockX snapshot version {snapshot['version']}. "
                    "Can only convert ViewDockX version 2 tool instances for ViewDock."
                )
                return None
            return cls(session, "ViewDock", snapshot['structures'])
        # ViewDock snapshots
        if snapshot['version'] != 1:
            session.logger.warning(
                f"Incompatible snapshot version {snapshot['version']} for ViewDock tool. "
                "Expected version 1."
            )
            return None

        return cls(session, snapshot['tool_name'], snapshot['structures'])

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
        structure.viewdock_data[RATING_KEY] = new_rating  # Update the rating in the structure's data

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
            # If the cell is selected as part of the row being higlighted, use the highlighted background color
            if option.state & QStyle.State_HasFocus:
                # If the cell is selected, use the highlighted background color
                painter.fillRect(option.rect, option.palette.highlight())
                combo_style.state |= QStyle.State_Selected  # Mark the combo box as selected
                painter.setPen(option.palette.highlightedText().color())  # Set text color to contrast selection
            else:
                # Table is not focused, use the alternate highlight background color
                painter.fillRect(option.rect, option.palette.alternateBase())
                painter.setPen(option.palette.text().color())
        else:
            # If the cell/row is not selected, use the default background color
            painter.fillRect(option.rect, option.palette.base())
            painter.setPen(option.palette.text().color())  # Set text color to default

        # This renders the full combo box frame (border + drop-down arrow) inside the cell
        style.drawComplexControl(QStyle.CC_ComboBox, combo_style, painter)

        # Get the rectangle where the text should be placed inside the combo box
        text_rect = style.subControlRect(QStyle.CC_ComboBox, combo_style, QStyle.SC_ComboBoxEditField, None)

        # Draw the text
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, combo_style.currentText)


class ViewDockSettings(Settings):

    EXPLICIT_SAVE = {ItemTable.DEFAULT_SETTINGS_ATTR: {}}

