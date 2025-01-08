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

import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import DisclosureArea
from Qt.QtWidgets import QHBoxLayout, QLineEdit, QScrollArea, QWidget, QGridLayout, QLabel, QVBoxLayout, QGroupBox, QPushButton
from Qt.QtGui import QPixmap
from Qt.QtCore import Qt
from .triggers import activate_trigger, add_handler, SCENE_SELECTED, EDITED, SAVED, SCENE_HIGHLIGHTED, DELETED
from chimerax.core.commands import run


class ScenesTool(ToolInstance):
    """
    Main tool for managing scenes. This tool contains a custom scroll area for displaying SceneItem widgets and a
    chimerax.ui DisclosureArea for adding, editing, and deleting scenes. The tool contains handlers for the following
    triggers: SCENE_SELECTED, EDITED, ADDED, SCENE_HIGHLIGHTED, DELETED.
    """

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
        self.handlers.append(add_handler(SAVED, self.scene_saved_cb))
        self.handlers.append(add_handler(SCENE_HIGHLIGHTED, self.scene_highlighted_cb))
        self.handlers.append(add_handler(DELETED, self.scene_deleted_cb))

        # SceneItem widget that is highlighted
        self.highlighted_scene = None

    def build_ui(self):
        self.main_layout = QVBoxLayout()
        self.tool_window.ui_area.setLayout(self.main_layout)

        self.scroll_area = SceneScrollArea(self.session)

        self.main_layout.addWidget(self.scroll_area)

        self.disclosure_area = DisclosureArea(title="Scene Actions")
        self.main_disclosure_layout = QVBoxLayout()

        self.scene_line_edit_widget = QWidget()
        self.scene_entry_label = QLabel("Scene Name:")
        self.scene_name_entry = QLineEdit()
        self.line_edit_layout = QHBoxLayout()
        self.line_edit_layout.addWidget(self.scene_entry_label)
        self.line_edit_layout.addWidget(self.scene_name_entry)
        self.scene_line_edit_widget.setLayout(self.line_edit_layout)
        self.main_disclosure_layout.addWidget(self.scene_line_edit_widget)

        self.disclosure_buttons_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_button_clicked)
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_button_clicked)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_button_clicked)

        self.disclosure_buttons_layout.addWidget(self.save_button)
        self.disclosure_buttons_layout.addWidget(self.edit_button)
        self.disclosure_buttons_layout.addWidget(self.delete_button)

        self.main_disclosure_layout.addLayout(self.disclosure_buttons_layout)

        self.disclosure_area.setContentLayout(self.main_disclosure_layout)
        self.main_layout.addWidget(self.disclosure_area)

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

    def scene_saved_cb(self, trigger_name, scene_name):
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

    def save_button_clicked(self):
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
    """
    The SceneScrollArea is a custom scroll area that contains SceneItem widgets. The scroll area is designed to display
    SceneItem widgets in a grid layout that adjusts to the width of the scroll area.

    Attributes:
        grid (QGridLayout): The grid layout that contains the SceneItem widgets.
        scene_items (list of SceneItem): List of SceneItem widgets.
    """

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
        """
        Update the grid layout with the current SceneItem widgets. This method will clear the grid layout and repopulate
        it with the SceneItem widgets in the correct orientation. The grid layout will adjust the number of columns based
        on the width of the scroll area. This method should be called whenever the scroll area is resized or when a new
        SceneItem widget is added or removed.
        """

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

        # Remove spacing on empty rows from previous grid setups
        for i in range(row, self.grid.rowCount()):
            self.grid.setRowMinimumHeight(i, 0)

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
        """
        Move the SceneItem widget to the top of the grid layout. This method adjusts the ordering of the scene_items
        attribute and updates the grid layout to reflect the new ordering in order to move a recently edited or added
        scene to the top of the grid layout.
        """
        scene_item = self.get_scene_item(scene_name)
        if scene_item:
            self.scene_items.remove(scene_item)
            self.scene_items.insert(0, scene_item)
            self.update_grid()

    def get_scene_item(self, name):
        return next((scene_item for scene_item in self.scene_items if scene_item.get_name() == name), None)


class SceneItem(QWidget):
    """
    The SceneItem widget is a custom widget that displays a thumbnail image and the name of a scene. The widget handles
    mouse events for selecting and highlighting itself as well as activating relevant tool triggers for click events.
    The widget is designed to be used in the SceneScrollArea.
    """

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
        """
        Set the thumbnail image for the SceneItem widget.

        Args:
            thumbnail_data (str): Base64 encoded image data for the thumbnail image.
        """
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
            self.setStyleSheet("border: 2px solid #007BFF;")
        else:
            self.setStyleSheet("")

    def get_name(self):
        return self.name
