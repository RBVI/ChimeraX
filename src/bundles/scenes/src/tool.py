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

SCENE_EVENT_MIME_FORMAT = "application/x-chimerax-scene"

import base64
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow, tool_user_error
from Qt.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QWidget,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QApplication,
)
from Qt.QtGui import QPixmap, QDrag
from Qt.QtCore import Qt, QMimeData, QPoint
from .triggers import (
    activate_trigger,
    add_handler,
    SCENE_SELECTED,
    SAVED,
    RESTORED,
    SCENE_HIGHLIGHTED,
    DELETED,
    RENAMED,
)
from chimerax.core.commands import run, StringArg

"""
This module defines the `ScenesTool` class and related classes for managing scenes within the ChimeraX application.

Classes:
    - ScenesTool: Main tool for managing scenes, including adding, editing, and deleting scenes.
    - SceneScrollArea: Custom scroll area that contains SceneItem widgets in a grid layout.
    - SceneItem: Custom widget that displays a thumbnail image and the name of a scene.

The `ScenesTool` class provides a user interface for managing scenes, including a scroll area for displaying SceneItem
widgets and an area for scene actions. The tool handles various triggers for scene selection, editing, saving,
 highlighting, and deletion.

The `SceneScrollArea` class manages the layout and display of SceneItem widgets, adjusting the grid layout based on the
scroll area width.

The `SceneItem` class represents an individual scene with a thumbnail and name, handling mouse events for selection and
activation of relevant triggers.
"""


class ScenesTool(ToolInstance):
    """
    Main tool for managing scenes. This tool contains a custom scroll area for displaying SceneItem widgets and an
    area for adding, editing, and deleting scenes. The tool contains handlers for the following
    triggers: SCENE_SELECTED, ADDED, SCENE_HIGHLIGHTED, DELETED.
    """

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Scenes"
        self.tool_window = MainToolWindow(self)
        self.build_ui()
        self.tool_window.manage("side")

        self.handlers = []
        self.handlers.append(add_handler(SCENE_SELECTED, self.scene_selected_cb))
        self.handlers.append(add_handler(SAVED, self.scene_saved_cb))
        self.handlers.append(add_handler(RESTORED, self.scene_restored_cb))
        self.handlers.append(add_handler(SCENE_HIGHLIGHTED, self.scene_highlighted_cb))
        self.handlers.append(add_handler(DELETED, self.scene_deleted_cb))
        self.handlers.append(add_handler(RENAMED, self.scene_renamed_cb))

    def build_ui(self):
        """
        Build the UI for the ScenesTool. The UI consists of a SceneScrollArea containing SceneItem widgets and a
        DisclosureArea for adding, editing, and deleting scenes.
        """

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(5)
        self.tool_window.ui_area.setLayout(self.main_layout)

        # The scroll area contains the SceneItem widgets ordered in a grid layout
        self.scroll_area = SceneScrollArea(self.session)
        self.main_layout.addWidget(self.scroll_area)

        # Set up the line edit for entering the scene name
        self.scene_entry_label = QLabel("Scene Name:")
        self.scene_name_entry = QLineEdit()
        self.scene_name_entry.returnPressed.connect(self.rename_button_clicked)
        self.user_changed_scene_name = False
        self.scene_name_entry.textEdited.connect(lambda *args, self=self:
            setattr(self, "user_changed_scene_name", True))
        self.line_edit_layout = QHBoxLayout()
        self.line_edit_layout.setSpacing(10)
        self.line_edit_layout.addWidget(self.scene_entry_label)
        self.line_edit_layout.addWidget(self.scene_name_entry)
        self.main_layout.addLayout(self.line_edit_layout)

        # Layout for buttons
        self.action_buttons_layout = QHBoxLayout()
        self.action_buttons_layout.setSpacing(5)

        # Create buttons for saving, editing, and deleting scenes and connect them to their respective methods
        self.save_button = QPushButton("Add")
        self.save_button.clicked.connect(self.save_button_clicked)
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_button_clicked)
        self.update_button.setEnabled(False)
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_button_clicked)
        self.rename_button.setEnabled(False)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_button_clicked)
        self.delete_button.setEnabled(False)

        # Add the buttons to the action buttons layout
        self.action_buttons_layout.addWidget(self.save_button)
        self.action_buttons_layout.addWidget(self.update_button)
        self.action_buttons_layout.addWidget(self.rename_button)
        self.action_buttons_layout.addWidget(self.delete_button)

        self.main_layout.addLayout(self.action_buttons_layout)

        self.tool_window.ui_area.setMinimumWidth(300)

    def scene_selected_cb(self, trigger_name, scene_name):
        """
        Callback for the SELECTED trigger. Restore the scene with the given name.
        """
        run(self.session, f"scene restore {StringArg.unparse(scene_name)}")

    def scene_saved_cb(self, trigger_name, scene_name):
        """
        Callback for the SAVED trigger. Get the newly added scene from the session and add it to the scroll area.
        """
        scene = self.session.scenes.get_scene(scene_name)
        scene_item_widget = self.scroll_area.get_scene_item(scene_name)
        if scene_item_widget:
            scene_item_widget.set_thumbnail(scene.get_thumbnail())
            self.scroll_area.set_latest_scene(scene_name)
        else:
            self.scroll_area.add_scene_item(scene_name, scene.get_thumbnail())

    def scene_restored_cb(self, trigger_name, scene_name):
        """
        Callback for the RESTORED trigger. Get the scene from the session put its name in the text area
        """
        scene_item_widget = self.scroll_area.get_scene_item(scene_name)
        if scene_item_widget:
            self.scroll_area.set_highlighted_scene(scene_item_widget)

    def scene_renamed_cb(self, trigger_name, names):
        """
        Callback for the SAVED trigger. Get the newly added scene from the session and add it to the scroll area.
        """
        old_scene_name, new_scene_name = names
        scene_item_widget = self.scroll_area.get_scene_item(old_scene_name)
        if scene_item_widget:
            scene_item_widget.set_name(new_scene_name)

    def scene_highlighted_cb(self, trigger_name, scene_name):
        """
        Callback for the HIGHLIGHTED trigger. This is now mostly handled by SceneScrollArea directly.
        """
        highlighting = scene_name is not None
        self.scene_name_entry.setText(scene_name if highlighting else "")
        self.user_changed_scene_name = False
        self.update_button.setEnabled(highlighting)
        self.rename_button.setEnabled(highlighting)
        self.delete_button.setEnabled(highlighting)

    def scene_deleted_cb(self, trigger_name, scene_name):
        """
        Callback for the DELETED trigger. Remove the scene from the scroll area and clear the highlighted scene if it is
        the deleted scene.
        """
        self.scroll_area.remove_scene_item(scene_name)
        highlighted_item = self.scroll_area.highlighted_scene
        self.scene_name_entry.setText(highlighted_item.name if highlighted_item else "")
        self.user_changed_scene_name = False
        self.update_button.setEnabled(bool(highlighted_item))
        self.rename_button.setEnabled(bool(highlighted_item))
        self.delete_button.setEnabled(bool(highlighted_item))

    def save_button_clicked(self):
        """
        Save the current scene with the name in the line edit widget.
        """
        scene_name = self.scene_name_entry.text().strip()
        if scene_name:
            if scene_name in self.session.scenes.scene_names:
                if self.user_changed_scene_name:
                    tool_user_error(f"Scene named {StringArg.unparse(scene_name)} already exists;"
                        " either use Update to update that scene or supply a new scene name.")
                    return
                scene_name = ""
        run(self.session, f"scene save {StringArg.unparse(scene_name)}")

    def update_button_clicked(self):
        """
        Save the current scene with the name in the line edit widget.
        """
        highlighted_scene = self.scroll_area.get_highlighted_scene()
        # 'Update' button only enabled if there's a highlighted scene
        scene_name = highlighted_scene.get_name()
        run(self.session, f"scene save {StringArg.unparse(scene_name)}")

    def rename_button_clicked(self):
        """
        Rename the highlighted scene.
        """
        highlighted_scene = self.scroll_area.get_highlighted_scene()
        if highlighted_scene:
            new_scene_name = self.scene_name_entry.text().strip()
            if not new_scene_name:
                tool_user_error(f"New scene name is blank")
            run(
                self.session,
                f"scene rename {StringArg.unparse(highlighted_scene.get_name())} {StringArg.unparse(new_scene_name)}",
            )

    def delete_button_clicked(self):
        """
        Delete the highlighted scene.
        """
        highlighted_scene = self.scroll_area.get_highlighted_scene()
        if highlighted_scene:
            run(
                self.session,
                f"scene delete {StringArg.unparse(highlighted_scene.name)}",
            )
        else:
            tool_user_error(f"No scene chosen in scene list")

    def delete(self):
        """
        Remove all handlers before deleting the tool instance.
        """
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def take_snapshot(self, session, flags):
        """
        Save the active highlighted scene
        """
        data = super().take_snapshot(session, flags)
        highlighted_scene = self.scroll_area.get_highlighted_scene()
        data["highlighted_scene"] = (
            highlighted_scene.get_name() if highlighted_scene else None
        )
        data["scroll_area"] = self.scroll_area.take_snapshot(session, flags)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        """
        Restore highlighted SceneItem in the tools scroll area when needed.
        """
        ti = super().restore_snapshot(session, data)  # get tool instance
        highlighted_scene_name = data.get("highlighted_scene")
        if highlighted_scene_name:
            scene_item = ti.scroll_area.get_scene_item(highlighted_scene_name)
            if scene_item:
                ti.scroll_area.set_highlighted_scene(scene_item)
        ti.scroll_area.set_state_from_snapshot(session, data.get("scroll_area"))
        return ti


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
        self.session = session
        self.container_widget = QWidget()
        self.grid = QGridLayout(self.container_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.grid.setHorizontalSpacing(0)  # Remove horizontal spacing
        self.grid.setVerticalSpacing(0)  # Remove vertical spacing"""
        self.grid.setAlignment(
            Qt.AlignLeft | Qt.AlignTop
        )  # Align items to the top-left
        self.setWidget(self.container_widget)
        self.cols = 0
        self.scene_items = []
        self.highlighted_scene = None
        self.init_scene_item_widgets(session)
        self.setAcceptDrops(True)

    def init_scene_item_widgets(self, session):
        """
        Initialize the SceneItem widgets in the scroll area. This method retrieves the scenes from the session and
        creates a SceneItem widget for each scene. The SceneItem widgets are added to the grid layout and the layout is
        updated to reflect the new widgets.
        """
        self.grid.setRowStretch(0, 0)
        self.grid.setColumnStretch(0, 0)
        scenes = session.scenes.get_scenes()
        self.scene_items = [
            SceneItem(scene.get_name(), scene.get_thumbnail()) for scene in scenes
        ]
        self.update_grid()

    def resizeEvent(self, event):
        """
        Override the resize event to update the grid layout when the scroll area is resized.
        """
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
        self.cols = max(
            1, width // SceneItem.IMAGE_WIDTH
        )  # Adjust to the desired width of each SceneItem
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
        """
        Add a new SceneItem widget to the scroll area. This method creates a new SceneItem widget with the given scene
        name and thumbnail data and appends it to the end of the scene_items list. The grid layout is then updated
        to reflect the new SceneItem widget.

        Args:
            scene_name (str): The name of the scene.
            thumbnail_data (str): Base64 encoded image data for the thumbnail image.
        """
        scene_item = SceneItem(scene_name, thumbnail_data)
        self.scene_items.append(scene_item)
        self.update_grid()

    def remove_scene_item(self, scene_name):
        """
        Remove a SceneItem widget from the scroll area. This method removes the SceneItem widget with the given scene
        name from the scene_items list and updates the grid layout to reflect the removal of the widget.
        """
        scene_item = self.get_scene_item(scene_name)
        if scene_item:
            if self.highlighted_scene == scene_item:
                self.highlighted_scene = None
            self.scene_items.remove(scene_item)
            self.update_grid()

    def set_latest_scene(self, scene_name):
        """
        Used to:
        Move the SceneItem widget to the top of the grid layout. This method adjusts the ordering of the scene_items
        attribute and updates the grid layout to reflect the new ordering in order to move a recently edited or added
        scene to the top of the grid layout.

        Now:
        No-op
        """
        # scene_item = self.get_scene_item(scene_name)
        # if scene_item:
        #    self.scene_items.remove(scene_item)
        #    self.scene_items.insert(0, scene_item)
        #    self.update_grid()

    def get_scene_item(self, name):
        return next(
            (
                scene_item
                for scene_item in self.scene_items
                if scene_item.get_name() == name
            ),
            None,
        )

    def set_highlighted_scene(self, scene_item):
        """
        Set the highlighted scene. If None, dehighlight current highlight.
        Clicking on an already highlighted scene keeps it highlighted.
        """
        if scene_item is None:
            if self.highlighted_scene:
                self.highlighted_scene.set_highlighted(False)
            self.highlighted_scene = None
        elif self.highlighted_scene is None:
            self.highlighted_scene = scene_item
            scene_item.set_highlighted(True)
        elif self.highlighted_scene == scene_item:
            # Keep it highlighted (no toggle)
            pass
        else:
            self.highlighted_scene.set_highlighted(False)
            scene_item.set_highlighted(True)
            self.highlighted_scene = scene_item

        activate_trigger(
            SCENE_HIGHLIGHTED,
            None
            if self.highlighted_scene is None
            else self.highlighted_scene.get_name(),
        )

    def get_highlighted_scene(self):
        """
        Get the currently highlighted scene item.
        """
        return self.highlighted_scene

    def take_snapshot(self, session, flags):
        """
        Save the order of the scenes in the grid for the snapshot data.
        """
        data = {
            "scene_items": [scene_item.get_name() for scene_item in self.scene_items]
        }
        return data

    def set_state_from_snapshot(self, session, data):
        """
        Restore the order of the scenes in the SceneScrollArea from the snapshot data.
        """
        ordered_scene_names = data.get("scene_items", [])
        # Create a dictionary for quick lookup of SceneItem by name from scene_items attribute.
        scene_item_dict = {
            scene_item.get_name(): scene_item for scene_item in self.scene_items
        }
        # Reorder self.scene_items based on snapshot data
        self.scene_items = [
            scene_item_dict[name]
            for name in ordered_scene_names
            if name in scene_item_dict
        ]
        self.update_grid()  # Make sure the grid layout reflects the new ordering

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            event.acceptProposedAction()
            import json
            scene_data_bytes = event.mimeData().data(SCENE_EVENT_MIME_FORMAT)
            scene_data_str = scene_data_bytes.data().decode("utf-8")
            try:
                scene_data = json.loads(scene_data_str)
                scene_name = scene_data.get("name")
            except json.JSONDecodeError:
                # Fallback: treat as plain scene name
                scene_name = scene_data_str
            self._reorder_from_drop_pos(scene_name, event.position())

    def mousePressEvent(self, event):
        """
        Handle mouse press events on the scroll area. If clicked on blank area, deselect highlighted scene.
        """
        if event.button() == Qt.LeftButton:
            # Check if click is on a SceneItem widget
            clicked_widget = self.childAt(event.pos())
            scene_item_clicked = False

            # Walk up the widget hierarchy to see if we clicked on a SceneItem
            widget = clicked_widget
            while widget and widget != self:
                if isinstance(widget, SceneItem):
                    scene_item_clicked = True
                    break
                widget = widget.parent()

            # If we didn't click on a SceneItem, deselect the highlighted scene
            if not scene_item_clicked:
                self.set_highlighted_scene(None)

        super().mousePressEvent(event)

    def _reorder_from_drop_pos(self, scene_name, drop_pos):
        drop_x, drop_y = drop_pos.x(), drop_pos.y()
        row_items = []
        for si in self.scene_items:
            si_x, si_y = si.pos().x(), si.pos().y()
            si_w, si_h = si.size().width(), si.size().height()
            if drop_y < si_y or drop_y > si_y + si_h:
                continue
            row_items.append(si)
        if not row_items:
            return
        for si in row_items:
            if drop_x < si.pos().x() + si.size().width()/2:
                try:
                    left = self.scene_items[self.scene_items.index(si)-1]
                except IndexError:
                    left = None
                right = si
                break
        else:
            # right of the entire row
            left = row_items[-1]
            try:
                right = self.scene_items[self.scene_items.index(left)+1]
            except IndexError:
                right = None
        if (left and left.name == scene_name) or (right and right.name == scene_name):
            # dropped into same place
            return
        drop_si = [si for si in self.scene_items if si.name == scene_name][0]
        self.scene_items.remove(drop_si)
        for i, si in enumerate(self.scene_items):
            if si == left:
                self.scene_items = self.scene_items[:i+1] + [drop_si] + self.scene_items[i+1:]
                break
            if si == right:
                self.scene_items = self.scene_items[:i] + [drop_si] + self.scene_items[i:]
                break
        self.update_grid()


class SceneItem(QWidget):
    """
    The SceneItem widget is a custom widget that displays a thumbnail image and the name of a scene. The widget handles
    mouse events for selecting and highlighting itself, activating the SELECTED trigger on single click, and initiating
    drag operations when the mouse is moved during a press. The widget is designed to be used in the SceneScrollArea.
    """

    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 100

    def __init__(self, scene_name, thumbnail_data, parent=None):
        super().__init__(parent)
        self.name = scene_name
        self.thumbnail_data = thumbnail_data
        self.drag_start_position = None
        self.drag_initiated = False
        self.init_ui()

    def init_ui(self):
        """
        Initialize the UI for the SceneItem widget. The UI consists of a thumbnail label and a name label.
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Label
        self.label = QLabel(self.name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.pixmap = QPixmap()
        self.set_thumbnail(self.thumbnail_data)
        layout.addWidget(self.thumbnail_label)

        # Set fixed size for the SceneItem
        self.setFixedSize(
            self.pixmap.width(), self.pixmap.height() + self.label.sizeHint().height()
        )

    def set_thumbnail(self, thumbnail_data):
        """
        Set the thumbnail image for the SceneItem widget.

        Args:
            thumbnail_data (str): Base64 encoded image data for the thumbnail image.
        """
        image_data = base64.b64decode(thumbnail_data)
        self.pixmap.loadFromData(image_data)
        self.pixmap = self.pixmap.scaled(
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, Qt.KeepAspectRatio
        )
        self.thumbnail_label.setPixmap(self.pixmap)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)

    def set_name(self, new_name):
        self.name = new_name
        self.label.setText(new_name)

    def mousePressEvent(self, event):
        """
        Handle mouse press events for selecting and highlighting the SceneItem widget.
        """
        if event.button() == Qt.LeftButton:
            # Store the position for potential drag operation and reset drag flag
            self.drag_start_position = event.pos()
            self.drag_initiated = False

            # Find the parent SceneScrollArea and tell it to highlight this item
            parent = self.parent()
            while parent and not isinstance(parent, SceneScrollArea):
                parent = parent.parent()
            if parent:
                parent.set_highlighted_scene(self)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events to initiate drag operations.
        """
        if not (event.buttons() & Qt.LeftButton):
            return

        if not self.drag_start_position:
            return

        # Check if we've moved far enough to start a drag
        if (
            event.pos() - self.drag_start_position
        ).manhattanLength() < QApplication.startDragDistance():
            return

        # Mark that we've initiated a drag to prevent click activation
        self.drag_initiated = True
        self.start_drag()
        super().mouseMoveEvent(event)

    def start_drag(self):
        """
        Start a drag operation with the scene data.
        """
        drag = QDrag(self)
        mime_data = QMimeData()

        # Set basic scene name
        mime_data.setText(self.name)

        # Get scene from session and encode comprehensive data
        parent = self.parent()
        while parent and not isinstance(parent, SceneScrollArea):
            parent = parent.parent()

        if parent and parent.session:
            scene = parent.session.scenes.get_scene(self.name)
            if scene:
                import json

                scene_data = {
                    "name": self.name,
                    "thumbnail": self.thumbnail_data,
                    "models": [],  # Will be populated from scene data
                }

                # Extract model positions from the scene if available
                try:
                    # This may vary depending on ChimeraX scene format
                    # For now, we'll pass the scene name and let the drop handler get the scene
                    pass
                except:
                    pass

                mime_data.setData(
                    SCENE_EVENT_MIME_FORMAT,
                    json.dumps(scene_data).encode("utf-8"),
                )
            else:
                # Fallback to just the name
                mime_data.setData(
                    SCENE_EVENT_MIME_FORMAT, self.name.encode("utf-8")
                )
        else:
            # Fallback to just the name
            mime_data.setData(SCENE_EVENT_MIME_FORMAT, self.name.encode("utf-8"))

        drag.setMimeData(mime_data)

        # Use the thumbnail as the drag pixmap
        drag.setPixmap(self.pixmap)
        drag.setHotSpot(QPoint(self.pixmap.width() // 2, self.pixmap.height() // 2))

        # Execute the drag
        drag.exec(Qt.CopyAction)

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events. If no drag was initiated, restore the scene.
        """
        if event.button() == Qt.LeftButton and not self.drag_initiated:
            # This was a click (not a drag), so restore the scene
            activate_trigger(SCENE_SELECTED, self.name)
        else:
            super().mouseReleaseEvent(event)

    def set_highlighted(self, highlighted):
        """
        Set the SceneItem widget as highlighted or not highlighted. This method updates the style of the widget to
        indicate whether it is highlighted or not.
        """
        if highlighted:
            self.setStyleSheet("border: 2px solid #007BFF;")
        else:
            self.setStyleSheet("")

    def get_name(self):
        return self.name
