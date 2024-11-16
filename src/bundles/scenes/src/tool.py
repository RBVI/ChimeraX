import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QHBoxLayout, QScrollArea, QWidget, QGridLayout, QLabel, QVBoxLayout, QSizePolicy
from Qt.QtGui import QPixmap
from Qt.QtCore import Qt
from .triggers import activate_trigger, add_handler, SCENE_SELECTED, EDITED
from chimerax.core.commands import run


class ScenesTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Scenes"
        self.tool_window = MainToolWindow(self)
        self.build_ui()
        self.tool_window.manage('side')

        self.handlers = []
        self.handlers.append(add_handler(SCENE_SELECTED, self.scene_selected_cb))
        self.handlers.append(add_handler(EDITED, self.scene_edited_cb))

    def build_ui(self):
        self.main_layout = QHBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scenes_widget = ScenesWidget(self.session)
        self.scroll_area.setWidget(self.scenes_widget)

        self.main_layout.addWidget(self.scroll_area)

    def scene_selected_cb(self, trigger_name, scene_name):
        run(self.session, f"scene restore {scene_name}")

    def scene_edited_cb(self, trigger_name, scene_name):
        scene_widget = self.scenes_widget.get_scene_item(scene_name)
        if scene_widget:
            scenes_mgr = self.session.scenes
            scene = scenes_mgr.get_scene(scene_name)
            if scene:
                scene_widget.set_thumbnail(scene.get_thumbnail())

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        super().delete()


class ScenesWidget(QWidget):
    ITEM_WIDTH = 110

    def __init__(self, session):
        super().__init__()
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        self.init_scene_item_widgets(session)

    def init_scene_item_widgets(self, session):
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

    def get_scene_item(self, name):
        """
        Get the scene item by name.

        Args:
            name (str): The name of the scene item.

        Returns:
            SceneItem | None: The scene item. None if not found.
        """
        return next((scene_item for scene_item in self.scene_items if scene_item.get_name() == name), None)


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
        self.pixmap = QPixmap()
        self.set_thumbnail(self.thumbnail_data)
        layout.addWidget(self.thumbnail_label)

        # Label
        self.label = QLabel(self.name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # Set fixed size for the SceneItem
        self.setFixedSize(self.pixmap.width(), self.pixmap.height() + self.label.sizeHint().height())

    def set_thumbnail(self, thumbnail_data):
        image_data = base64.b64decode(thumbnail_data)
        self.pixmap.loadFromData(image_data)
        self.pixmap = self.pixmap.scaled(100, 100, Qt.KeepAspectRatio)
        self.thumbnail_label.setPixmap(self.pixmap)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            activate_trigger(SCENE_SELECTED, self.name)
        super().mousePressEvent(event)

    def get_name(self):
        return self.name
