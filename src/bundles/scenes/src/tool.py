import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QHBoxLayout, QScrollArea, QWidget, QGridLayout, QLabel, QVBoxLayout, QSizePolicy
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

    ITEM_WIDTH = 110

    def __init__(self, session):
        super().__init__()
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        self.add_scenes(session)

    def add_scenes(self, session):
        self.main_layout.setRowStretch(0, 0)
        self.main_layout.setColumnStretch(0, 0)
        scenes = session.scenes.get_scenes()
        self.scene_items = [SceneItem(scene.get_name(), scene.get_thumbnail()) for scene in scenes]
        self.update_layout()

    def resizeEvent(self, event):
        self.update_layout()
        super().resizeEvent(event)

    def update_layout(self):
        # Clear the layout. Before repopulating it in the correct orientation
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)

        width = self.width()
        columns = max(1, width // self.ITEM_WIDTH)  # Adjust to the desired width of each SceneItem
        row, col = 0, 0
        for scene_item in self.scene_items:
            self.main_layout.addWidget(scene_item, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

        # Calculate the required height based on the number of rows
        item_height = self.scene_items[0].height() if self.scene_items else 0
        required_height = (row + 1) * item_height
        self.setMinimumHeight(required_height)


class SceneItem(QWidget):
    def __init__(self, scene_name, thumbnail_data, parent=None):
        super().__init__(parent)
        self.name = scene_name
        self.thumbnail_data = thumbnail_data
        self.init_ui()

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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print(f"SceneItem '{self.name}' clicked")
        super().mousePressEvent(event)
