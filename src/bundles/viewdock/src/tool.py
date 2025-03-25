from chimerax.core.tools import ToolInstance
from chimerax.ui.widgets import ItemTable
from Qt.QtWidgets import QVBoxLayout
from chimerax.core.commands import run
from chimerax.core.models import REMOVE_MODELS

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
        self.table_setup()
        self.handlers = []
        self.add_handlers()
        self.tool_window.manage('side')

    def table_setup(self):
        """
        Create the ItemTable for the structures. Creates all columns and links all callbacks.
        """
        self.struct_table = ItemTable(session=self.session)
        self.struct_table.add_column('Show', lambda s: s.display, data_set=self.set_visibility, format=ItemTable.COL_FORMAT_BOOLEAN)
        self.struct_table.add_column('ID', lambda s: s.id_string)
        self.struct_table.add_column('Name', lambda s: s.viewdockx_data.get('Name', ''))
        self.struct_table.data = self.structures
        self.struct_table.launch()
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

"""model display changed"""