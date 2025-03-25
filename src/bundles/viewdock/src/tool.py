from chimerax.core.tools import ToolInstance
from chimerax.ui.widgets import ItemTable
from Qt.QtWidgets import QVBoxLayout
from chimerax.core.commands import run

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

"""model display changed"""