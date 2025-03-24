from chimerax.core.tools import ToolInstance
from chimerax.ui.widgets import ItemTable
from Qt.QtWidgets import QVBoxLayout

class ViewDockTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name, structures):
        super().__init__(session, tool_name)
        self.display_name = "ViewDock"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)

        self.setup(structures)

    def setup(self, structures):
        self.main_v_layout = QVBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_v_layout)

        self.struct_table = ItemTable(session=self.session)
        self.struct_table.add_column('ID', lambda s: s.id_string)
        self.struct_table.add_column('Name', lambda s: s.viewdockx_data.get('Name', ''))
        self.struct_table.data = structures
        self.struct_table.launch()
        self.main_v_layout.addWidget(self.struct_table)

        self.tool_window.manage('side')

