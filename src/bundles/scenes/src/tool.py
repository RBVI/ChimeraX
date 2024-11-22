import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QHBoxLayout, QLineEdit, QScrollArea, QWidget, QGridLayout, QLabel, QVBoxLayout, QGroupBox, QPushButton
from Qt.QtGui import QPixmap
from Qt.QtCore import Qt
from .triggers import activate_trigger, add_handler, SCENE_SELECTED, EDITED, ADDED, SCENE_HIGHLIGHTED, DELETED
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
        self.handlers.append(add_handler(ADDED, self.scene_added_cb))
        self.handlers.append(add_handler(SCENE_HIGHLIGHTED, self.scene_highlighted_cb))
        self.handlers.append(add_handler(DELETED, self.scene_deleted_cb))

        # SceneItem widget that is highlighted
        self.highlighted_scene = None

    def build_ui(self):
        self.main_layout = QVBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_layout)

        self.scroll_area = SceneScrollArea(self.session)

        self.main_layout.addWidget(self.scroll_area)

        self.collapsible_box = CollapsibleBox("Actions")

        self.scene_line_edit_widget = QWidget()
        self.scene_entry_label = QLabel("Scene Name:")
        self.scene_name_entry = QLineEdit()
        self.line_edit_layout = QHBoxLayout()
        self.line_edit_layout.addWidget(self.scene_entry_label)
        self.line_edit_layout.addWidget(self.scene_name_entry)
        self.scene_line_edit_widget.setLayout(self.line_edit_layout)
        self.collapsible_box.add_widget(self.scene_line_edit_widget)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_button_clicked)
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_button_clicked)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_button_clicked)

        self.collapsible_box.add_widget(self.add_button)
        self.collapsible_box.add_widget(self.edit_button)
        self.collapsible_box.add_widget(self.delete_button)

        self.collapsible_box.on_toggled(False)

        self.main_layout.addWidget(self.collapsible_box)

        self.tool_window.ui_area.setMinimumWidth(300)

    def scene_selected_cb(self, trigger_name, scene_name):
        run(self.session, f"scene restore {scene_name}")

    def scene_edited_cb(self, trigger_name, scene_name):
        scene_item_widget = self.scroll_area.get_scene_item(scene_name)
        if scene_item_widget:
            scenes_mgr = self.session.scenes
            scene = scenes_mgr.get_scene(scene_name)
            if scene:
                scene_item_widget.set_thumbnail(scene.get_thumbnail())
                self.scroll_area.set_latest_scene(scene_name)

    def scene_added_cb(self, trigger_name, scene_name):
        scene = self.session.scenes.get_scene(scene_name)
        if scene:
            self.scroll_area.add_scene_item(scene_name, scene.get_thumbnail())

    def scene_highlighted_cb(self, trigger_name, scene_name):
        if self.highlighted_scene:
            if self.highlighted_scene.get_name() != scene_name:
                self.highlighted_scene.set_highlighted(False)
        self.highlighted_scene = self.scroll_area.get_scene_item(scene_name)

    def scene_deleted_cb(self, trigger_name, scene_name):
        self.scroll_area.remove_scene_item(scene_name)
        if self.highlighted_scene and self.highlighted_scene.get_name() == scene_name:
            self.highlighted_scene = None

    def add_button_clicked(self):
        scene_name = self.scene_name_entry.text()
        if not scene_name:
            self.session.logger.warning("Scene name cannot be empty.")
            return
        run(self.session, f"scene save {scene_name}")

    def edit_button_clicked(self):
        if self.highlighted_scene:
            run(self.session, f"scene edit {self.highlighted_scene.get_name()}")
        else:
            self.session.logger.warning("No scene selected to edit")

    def delete_button_clicked(self):
        if self.highlighted_scene:
            run(self.session, f"scene delete {self.highlighted_scene.get_name()}")
        else:
            self.session.logger.warning("No scene selected to delete")

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        super().delete()


class SceneScrollArea(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.container_widget = QWidget()
        self.grid = QGridLayout(self.container_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.grid.setHorizontalSpacing(0)  # Remove horizontal spacing
        self.grid.setVerticalSpacing(0)  # Remove vertical spacing"""
        self.grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Align items to the top-left
        self.setWidget(self.container_widget)
        self.cols = 0
        self.scene_items = []
        self.init_scene_item_widgets(session)

    def init_scene_item_widgets(self, session):
        self.grid.setRowStretch(0, 0)
        self.grid.setColumnStretch(0, 0)
        scenes = session.scenes.get_scenes()
        self.scene_items = [SceneItem(scene.get_name(), scene.get_thumbnail()) for scene in scenes]
        self.update_grid()

    def resizeEvent(self, event):
        required_cols = self.viewport().width() // SceneItem.IMAGE_WIDTH
        if required_cols != self.cols:
            self.update_grid()
        super().resizeEvent(event)

    def update_grid(self):
        # Clear the grid before repopulating it in the correct orientation
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)

        width = self.viewport().width()
        self.cols = max(1, width // SceneItem.IMAGE_WIDTH)  # Adjust to the desired width of each SceneItem
        row, col = 0, 0
        for scene_item in self.scene_items:
            self.grid.addWidget(scene_item, row, col)
            col += 1
            if col >= self.cols:
                self.grid.setRowMinimumHeight(row, scene_item.height())
                col = 0
                row += 1

    def add_scene_item(self, scene_name, thumbnail_data):
        scene_item = SceneItem(scene_name, thumbnail_data)
        self.scene_items.insert(0, scene_item)
        self.update_grid()

    def remove_scene_item(self, scene_name):
        scene_item = self.get_scene_item(scene_name)
        if scene_item:
            self.scene_items.remove(scene_item)
            self.update_grid()

    def set_latest_scene(self, scene_name):
        scene_item = self.get_scene_item(scene_name)
        if scene_item:
            self.scene_items.remove(scene_item)
            self.scene_items.insert(0, scene_item)
            self.update_grid()

    def get_scene_item(self, name):
        return next((scene_item for scene_item in self.scene_items if scene_item.get_name() == name), None)


class SceneItem(QWidget):
    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 100

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
        self.pixmap = self.pixmap.scaled(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, Qt.KeepAspectRatio)
        self.thumbnail_label.setPixmap(self.pixmap)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_highlighted(True)
            activate_trigger(SCENE_HIGHLIGHTED, self.name)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        activate_trigger(SCENE_SELECTED, self.name)
        super().mouseDoubleClickEvent(event)

    def set_highlighted(self, highlighted):
        if highlighted:
            self.setStyleSheet("border: 2px solid green;")
        else:
            self.setStyleSheet("")

    def get_name(self):
        return self.name


class CollapsibleBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self.on_toggled)

        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content.setLayout(self.content_layout)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.content)
        self.setLayout(self.main_layout)

    def on_toggled(self, checked):
        self.content.setVisible(checked)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
