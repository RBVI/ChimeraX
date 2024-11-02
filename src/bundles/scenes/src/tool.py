import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QHBoxLayout, QScrollArea, QWidget, QGridLayout, QLabel, QVBoxLayout
from Qt.QtGui import QPixmap
from Qt.QtCore import Qt


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

        self.scenes_widget = ScenesWidget(self.session)
        self.scroll_area.setWidget(self.scenes_widget)

        self.main_layout.addWidget(self.scroll_area)


class ScenesWidget(QWidget):

    def __init__(self, session):
        super().__init__()
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        self.add_scenes(session)

    def add_scenes(self, session):
        scenes = session.scenes.get_scenes()
        row, col = 0, 0
        for scene in scenes:
            scene_item = SceneItem(scene.get_name(), scene.get_thumbnail())
            scene_item.init_ui()
            self.main_layout.addWidget(scene_item, row, col)
            col += 1
            if col >= 4:  # Adjust the number of columns as needed
                col = 0
                row += 1


class SceneItem(QWidget):
    def __init__(self, scene_name, thumbnail_data, parent=None):
        super().__init__(parent)
        self.name = scene_name
        self.thumbnail_data = thumbnail_data

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Thumbnail
        self.thumbnail_label = QLabel()
        pixmap = QPixmap()
        image_data = base64.b64decode(self.thumbnail_data)
        pixmap.loadFromData(image_data)
        pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
        self.thumbnail_label.setPixmap(pixmap)
        self.thumbnail_label.setPixmap(pixmap)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.thumbnail_label)

        # Label
        self.label = QLabel(self.name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # Set fixed size for the SceneItem
        self.setFixedSize(pixmap.width(), pixmap.height() + self.label.sizeHint().height())
