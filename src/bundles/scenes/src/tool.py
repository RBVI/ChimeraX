from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QHBoxLayout, QScrollArea, QWidget, QGridLayout, QLabel


class ScenesTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Scenes"
        self.tool_window = MainToolWindow(self)
        self.build_ui()
        self.tool_window.manage('side')

    def build_ui(self):
        self.main_layout = QHBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scenes_widget = ScenesWidget()
        self.scroll_area.setWidget(self.scenes_widget)

        self.main_layout.addWidget(self.scroll_area)


class ScenesWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        self.add_red_boxes()

    def add_red_boxes(self):
        for row in range(3):  # Adjust the number of rows as needed
            for col in range(3):  # Adjust the number of columns as needed
                red_box = QLabel()
                red_box.setStyleSheet("background-color: red; border: 1px solid black;")
                red_box.setFixedSize(100, 100)  # Adjust the size as needed
                self.main_layout.addWidget(red_box, row, col)

