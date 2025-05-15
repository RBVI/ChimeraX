"""
This module defines the KeyframeEditorWidget class and related classes for visualizing and interacting with the
Animations manager from the animations bundle in ChimeraX.

Classes:
    KeyframeEditorWidget: A widget for visualizing and interacting with keyframes in an animation.
    KFEGraphicsView: A custom QGraphicsView for the keyframe editor.
    KeyframeEditorScene: A custom QGraphicsScene for the keyframe editor.
    Timeline: A QGraphicsItemGroup representing the timeline.
    KeyframeItem: A QGraphicsPixmapItem representing a keyframe.
    TimelineCursor: A QGraphicsLineItem representing the timeline cursor.
    TickMarkItem: A QGraphicsLineItem representing a tick mark on the timeline.
    TickIntervalLabel: A QGraphicsTextItem representing a label for tick intervals.
"""

from Qt.QtWidgets import (QGraphicsPixmapItem, QGraphicsItem, QGraphicsView, QGraphicsScene,
                          QVBoxLayout, QWidget, QGraphicsTextItem, QGraphicsLineItem, QGraphicsItemGroup, QPushButton,
                          QSizePolicy, QLabel,
                          QHBoxLayout, QStyle, QGraphicsRectItem)
from Qt.QtCore import QByteArray, Qt, QPointF, QLineF, QObject, Signal, QSize, QTimer, QRectF
from Qt.QtGui import QPixmap, QPen, QTransform, QBrush
from .animation import format_time
from .triggers import (MGR_KF_ADDED, MGR_KF_DELETED, MGR_KF_EDITED, MGR_LENGTH_CHANGED, MGR_PREVIEWED, KF_ADD,
                       KF_DELETE, KF_EDIT, LENGTH_CHANGE, PREVIEW, PLAY, add_handler, activate_trigger,
                       MGR_FRAME_PLAYED, RECORD, STOP_PLAYING, REMOVE_TIME, INSERT_TIME, remove_handler, STOP_RECORDING,
                       MGR_RECORDING_STOP, MGR_RECORDING_START)


class KeyframeEditorWidget(QWidget):
    """
    A widget for editing keyframes in an animation.

    Attributes:
        layout (QVBoxLayout): The main layout of the widget.
        handlers (list): List of handlers for triggers.
        time_label_layout (QHBoxLayout): Layout for the time label.
        time_label (QLabel): Label displaying the current time.
        kfe_view (KFEGraphicsView): Graphics view for the keyframe editor.
        kfe_scene (KeyframeEditorScene): Graphics scene for the keyframe editor.
        button_layout (QHBoxLayout): Layout for navigation buttons.
        time_buttons_layout (QHBoxLayout): Layout for time adjustment buttons.
    """

    def __init__(self, length, keyframes):
        """
        Initialize the KeyframeEditorWidget.

        Args:
            length (int | float): Length of the timeline in seconds.
            keyframes (list): List of animation.Keyframe objects.
        """
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.handlers = []

        # Time label
        self.time_label_layout = QHBoxLayout()
        self.time_label = QLabel()
        self.time_label_layout.addWidget(self.time_label)
        self.layout.addLayout(self.time_label_layout)

        # Keyframe editor graphics view widget
        self.kfe_view = KFEGraphicsView(self)
        self.kfe_scene = KeyframeEditorScene(length, keyframes)
        self.kfe_view.setScene(self.kfe_scene)
        self.layout.addWidget(self.kfe_view)

        # Connect time label triggers. Must be done after the KeyframeEditor widget is created because depends on
        # the keyframe editor scene.
        self.update_time_label(0)
        self.handlers.append(add_handler(MGR_PREVIEWED, lambda trigger_name, time: self.update_time_label(time)))
        self.handlers.append(add_handler(MGR_FRAME_PLAYED, lambda trigger_name, time: self.update_time_label(time)))
        # This handler is needed for when the cursor is moved but there is no manager preview call. This happens if
        # There are no keyframes in the animation.
        self.handlers.append(add_handler(PREVIEW, lambda trigger_name, time: self.update_time_label(time)))

        # Horizontal layout for navigation buttons
        self.button_layout = QHBoxLayout()

        # Rewind button
        self.rewind_button = QPushButton()
        self.rewind_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.button_layout.addWidget(self.rewind_button)
        self.rewind_button.clicked.connect(self.rewind)

        # Play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.button_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(
            lambda: activate_trigger(PLAY, (self.kfe_scene.get_cursor().get_time(), False)))

        # Pause button
        self.pause_button = QPushButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.button_layout.addWidget(self.pause_button)
        self.pause_button.clicked.connect(lambda: activate_trigger(STOP_PLAYING, None))

        # Fast forward button
        self.fast_forward_button = QPushButton()
        self.fast_forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.button_layout.addWidget(self.fast_forward_button)
        self.fast_forward_button.clicked.connect(self.fast_forward)

        # Record button
        self.record_button = QPushButton("Record")
        self.record_button.setCheckable(True)
        self.button_layout.addWidget(self.record_button)
        self.record_button.clicked.connect(self.recording_toggle)
        self.handlers.append(
            add_handler(MGR_RECORDING_START, lambda trigger_name, data: self.record_button.setChecked(True)))
        self.handlers.append(
            add_handler(MGR_RECORDING_STOP, lambda trigger_name, data: self.record_button.setChecked(False)))

        # Add button
        self.add_button = QPushButton("Add")
        self.button_layout.addWidget(self.add_button)
        self.add_button.clicked.connect(lambda: activate_trigger(KF_ADD, self.kfe_scene.get_cursor().get_time()))

        # Delete button
        self.delete_button = QPushButton("Delete")
        self.button_layout.addWidget(self.delete_button)
        self.delete_button.clicked.connect(lambda: self.delete_keyframes())

        self.layout.addLayout(self.button_layout)

        # Layout for all the time adjustment buttons
        self.time_buttons_layout = QHBoxLayout()

        # Time adjustment button values
        remove_large = -5
        remove_medium = -2
        remove_small = -0.5

        insert_small = 0.5
        insert_medium = 2
        insert_large = 5

        # Remove buttons
        for adjustment in [remove_large, remove_medium, remove_small, insert_small, insert_medium, insert_large]:
            self.add_time_adjustment_button(adjustment)

        self.layout.addLayout(self.time_buttons_layout)

    def add_time_adjustment_button(self, d_time):
        """
        Add a button for adjusting the timeline by a specified amount of time.

        Args:
            d_time (float): The amount of time for the button to adjust the timeline by in seconds.
        """
        button = QPushButton(f"{d_time}s")
        if d_time < 0:
            trigger_activation = lambda: activate_trigger(
                REMOVE_TIME, (self.kfe_scene.get_cursor().get_time(), d_time * -1))
        else:
            trigger_activation = lambda: activate_trigger(
                INSERT_TIME, (self.kfe_scene.get_cursor().get_time(), d_time))
        button.clicked.connect(trigger_activation)
        self.time_buttons_layout.addWidget(button)

    def update_time_label(self, time):
        """
        Update the time label to display the specified time.

        Args:
            time (int | float): The time to display in seconds.
        """
        self.time_label.setText(f"{format_time(time)} / {format_time(self.kfe_scene.timeline.get_time_length())}")

    def rewind(self):
        """
        Rewind the timeline to the beginning.
        """
        activate_trigger(STOP_PLAYING, None)
        cursor = self.kfe_scene.get_cursor()
        cursor.set_pos_from_time(0)
        self.kfe_view.horizontalScrollBar().setValue(0)
        self.kfe_scene.get_cursor().activate_preview_trigger()

    def fast_forward(self):
        """
        Fast-forward the timeline to the end.
        """
        activate_trigger(STOP_PLAYING, None)
        cursor = self.kfe_scene.get_cursor()
        timeline_len = self.kfe_scene.timeline.get_time_length()
        cursor.set_pos_from_time(timeline_len)
        self.kfe_view.horizontalScrollBar().setValue(self.kfe_view.horizontalScrollBar().maximum())
        self.kfe_scene.get_cursor().activate_preview_trigger()

    def delete_keyframes(self):
        """
        Delete selected keyframes.
        """
        keyframes = self.kfe_scene.get_selected_keyframes()
        for keyframe in keyframes:
            activate_trigger(KF_DELETE, keyframe.get_name())

    def recording_toggle(self):
        """
        Toggle recording on or off.
        """
        if self.record_button.isChecked():
            # Clicking on the button flips the checked state automatically. We don't want automatic because we only
            # want it checked if the animation has actually started recording. It is important that this reset happens
            # before we trigger a start or stop recording because the handlers for the managers start stop recording
            # will set the checked state properly and that code is run as soon as the trigger is activated. If the
            # manager start stop doesn't get called we need to make sure that our button is set to unchecked.
            self.record_button.setChecked(False)
            activate_trigger(RECORD, None)
        else:
            # Same reset situation here as above
            self.record_button.setChecked(False)
            activate_trigger(STOP_RECORDING, None)

    def remove_handlers(self):
        """
        Remove all trigger handlers.
        """
        for handler in self.handlers:
            remove_handler(handler)
        self.kfe_scene.remove_handlers()


class KFEGraphicsView(QGraphicsView):
    """
    A custom QGraphicsView for the keyframe editor.

    Attributes:
        scroll_timer (QTimer): Timer for auto-scrolling functionality.
        _is_dragging_cursor (bool): Flag indicating if the cursor is being dragged.
    """
    def __init__(self, scene):
        """
        Initialize the KFEGraphicsView.

        Args:
            scene (QGraphicsScene): The scene to be displayed in the view.
        """
        super().__init__(scene)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self.auto_scroll)
        self.scroll_timer.start(50)  # Check every 50 ms
        self._is_dragging_cursor = False

    def sizeHint(self):
        """
        Provide a recommended size for the view based on the scene rectangle width and height.

        Returns:
            QSize: The recommended size for the view.
        """
        # Get the current scene rectangle
        scene_rect = self.scene().sceneRect()
        width = min(700, int(scene_rect.width()))
        height = int(scene_rect.height() + 400)
        return QSize(width, height)

    def auto_scroll(self):
        """
        Automatically scroll the view if necessary when the cursor is near the side of the view.
        """
        if not self._is_dragging_cursor:
            return

        cursor_pos = self.mapFromGlobal(self.cursor().pos())

        # slow scroll margin must be > fast scroll margin. Refers to how many pixels from the edge into the view
        slow_scroll_margin = 20  # Margin in pixels to start slow scrolling
        fast_scroll_margin = -20  # Margin in pixels to start fast scrolling

        scroll_speed = 10  # Pixels per scroll
        fast_scroll_speed = 50  # Pixels per scroll

        if cursor_pos.x() < fast_scroll_margin:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - fast_scroll_speed)
        if cursor_pos.x() < slow_scroll_margin:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - scroll_speed)
        elif cursor_pos.x() > self.viewport().width() - fast_scroll_margin:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + fast_scroll_speed)
        elif cursor_pos.x() > self.viewport().width() - slow_scroll_margin:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + scroll_speed)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging_cursor = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging_cursor = False
        super().mouseReleaseEvent(event)


class KeyframeEditorScene(QGraphicsScene):
    """
    A custom QGraphicsScene for the keyframe editor.

    Attributes:
        handlers (list): List of handlers for triggers.
        timeline (Timeline): The timeline graphics item in the scene.
        cursor (TimelineCursor): The cursor graphics item in the scene.
        keyframes (dict): Dictionary of keyframe names to KeyframeItem objects.
        selection_box (QGraphicsRectItem): The selection box for drag selection.
        selection_start_pos (QPointF): The starting position of the selection box.
    """
    def __init__(self, length, keyframes):
        """
        Initialize the KeyframeEditorScene.

        Args:
            length (int | float): Length of the timeline in seconds.
            keyframes (list): List of animation.Keyframe objects.
        """
        super().__init__()
        self.handlers = []
        self.timeline = Timeline(time_length=length)
        self.addItem(self.timeline)
        self.cursor = TimelineCursor(QPointF(0, 0), 70, self.timeline)
        self.addItem(self.cursor)
        self.keyframes = {}  # Dictionary of keyframe name to KeyframeItem

        # Box for highlight drag.
        self.selection_box = None
        self.selection_start_pos = None

        for kf in keyframes:
            self.add_kf_item(kf)

        # Connect triggers from the animation manager in the session to the keyframe editor
        self.handlers.append(add_handler(MGR_KF_ADDED, lambda trigger_name, data: self.add_kf_item(data)))
        self.handlers.append(add_handler(MGR_KF_EDITED, lambda trigger_name, data: self.handle_keyframe_edit(data)))
        self.handlers.append(add_handler(MGR_KF_DELETED, lambda trigger_name, data: self.delete_kf_item(data)))
        self.handlers.append(
            add_handler(MGR_LENGTH_CHANGED, lambda trigger_name, data: self.animation_len_changed(data)))
        self.handlers.append(add_handler(MGR_PREVIEWED, lambda trigger_name, data: self.cursor.set_pos_from_time(data)))
        self.handlers.append(
            add_handler(MGR_FRAME_PLAYED, lambda trigger_name, data: self.cursor.set_pos_from_time(data)))

        self.selectionChanged.connect(self.on_selection_changed)

    def add_kf_item(self, kf):
        """
        Add a keyframe item to the scene.

        Args:
            kf (animation.Keyframe): The keyframe to add.
        """
        thumbnail_bytes = kf.get_thumbnail()
        pixmap = QPixmap()
        pixmap.loadFromData(thumbnail_bytes, "JPEG")
        pixmap = pixmap.scaled(KeyframeItem.SIZE, KeyframeItem.SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        kf_item_x = self.timeline.get_pos_for_time(kf.get_time()) - KeyframeItem.SIZE / 2
        keyframe_item = KeyframeItem(kf.get_name(), pixmap, QPointF(kf_item_x, 0),
                                     self.timeline)
        # Need to update the info label on the keyframe graphic item. Placing on the timeline
        # automatically does a pos to time conversion based on the timeline.
        # However, 1 pixel change may be a 0.02 change in time.
        keyframe_item.set_info_time(kf.get_time())
        self.keyframes[kf.get_name()] = keyframe_item
        self.addItem(keyframe_item)

    def handle_keyframe_edit(self, kf):
        """
        Handle editing of a keyframe and updating position of KeyframeItem. Expected to be called from a trigger.

        Args:
            kf (animation.Keyframe): The keyframe to edit.
        """
        keyframe_item = self.keyframes[kf.get_name()]
        if keyframe_item is None:
            raise ValueError(f"Keyframe graphics item with name {kf.get_name()} not found.")
        keyframe_item.set_position_from_time(kf.get_time())
        self.cursor.activate_preview_trigger()

    def delete_kf_item(self, kf):
        """
        Delete a keyframe item from the scene. Expected to be called from a trigger.

        Args:
            kf (str): The name of the keyframe to delete.
        """
        keyframe_item = self.keyframes.pop(kf.get_name())
        self.removeItem(keyframe_item)

    def update_scene_size(self):
        """
        Update the size of the scene based on the timeline length.
        """
        margin = 10  # Margin in pixels on each side
        scene_width = self.timeline.get_pix_length() + 2 * margin  # Total width including margins
        self.setSceneRect(-margin, 0, scene_width, self.height())

    def animation_len_changed(self, length):
        """
        Handle changes to the animation length. Expected to be called from a trigger.

        Args:
            length (int | float): The new length of the animation in seconds.
        """
        self.timeline.set_time_length(length)
        if self.cursor.x() > self.timeline.x() + self.timeline.get_pix_length():
            self.cursor.setX(self.timeline.x() + self.timeline.get_pix_length())
        self.cursor.activate_preview_trigger()
        self.update_scene_size()

    def mousePressEvent(self, event):
        """
        When the mouse is pressed on the scene handle moving the cursor, selecting a keyframe, or starting a drag
        selection box.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse press event.
        """
        if event.button() == Qt.LeftButton:
            clicked_pos = event.scenePos()
            if not isinstance(self.itemAt(clicked_pos, QTransform()), KeyframeItem):
                self.clearSelection()
            if self.timeline.contains(clicked_pos):
                self.cursor.setPos(clicked_pos)

            # Activate a selection box if click happened in blank space.
            if not self.itemAt(clicked_pos, QTransform()):
                self.start_selection_box(clicked_pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events. Adjust the selection box and select any keyframes that intersect with the box.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse move event.
        """
        # Adjust the selection box if it has been started.
        if self.selection_box:
            current_pos = event.scenePos()
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_box.setRect(rect)
            for keyframe_item in self.keyframes.values():
                if rect.intersects(keyframe_item.sceneBoundingRect()):
                    keyframe_item.setSelected(True)
                else:
                    keyframe_item.setSelected(False)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event, QGraphicsSceneMouseEvent=None):
        """
        Handle mouse release events. Trigger keyframe editing and end the selection box if it has been started.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse release event.
        """
        if event.button() == Qt.LeftButton:
            keyframes = self.get_selected_keyframes()
            for keyframes in keyframes:
                keyframes.trigger_for_edit()

            # End the selection box if it has been started.
            if self.selection_box:
                self.end_selection_box(event.scenePos())
        super().mouseReleaseEvent(event)

    def start_selection_box(self, pos):
        """
        Start the selection box for drag selection.

        Args:
            pos (QPointF): The starting position of the selection box.
        """
        self.selection_start_pos = pos
        self.selection_box = QGraphicsRectItem(QRectF(pos, pos))
        self.selection_box.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
        self.selection_box.setBrush(QBrush(Qt.transparent))
        self.addItem(self.selection_box)

    def end_selection_box(self, pos):
        """
        End the selection box for drag selection.

        Args:
            pos (QPointF): The ending position of the selection box.
        """
        self.removeItem(self.selection_box)
        self.selection_box = None
        self.selection_start_pos = None

    def on_selection_changed(self):
        """
        Handle changes to the selection. Make sure that keyframes display their time info when they are selected.
        """
        selected_keyframes = self.get_selected_keyframes()
        for item in self.items():
            if isinstance(item, KeyframeItem):
                if item not in selected_keyframes:
                    item.hide_info()
                else:
                    item.show_info()

    def remove_handlers(self):
        """
        Remove all trigger handlers.
        """
        for handler in self.handlers:
            remove_handler(handler)

    def get_selected_keyframes(self):
        """
        Get the selected graphical keyframes.

        Returns:
            list: List of selected KeyframeItem objects.
        """
        selected_keyframes = []
        for item in self.selectedItems():
            if isinstance(item, KeyframeItem):
                selected_keyframes.append(item)
        return selected_keyframes

    def get_cursor(self):
        """
        Get the timeline cursor.

        Returns:
            TimelineCursor: The timeline cursor.
        """
        return self.cursor


class Timeline(QGraphicsItemGroup):
    """
    A QGraphicsItemGroup representing the timeline markings.

    Attributes:
        SCALE (int): Scale factor in pixels per second.
        time_length (int | float): Length of the timeline in seconds.
        pix_length (int): Length of the timeline in pixels.
        interval (float): Interval between tick marks in seconds.
        major_interval (float): Interval between major tick marks in seconds.
        tick_length (int): Length of the tick marks in pixels.
        major_tick_length (int): Length of the major tick marks in pixels.
    """

    SCALE = 60  # Scale factor. Pixels per second.

    def __init__(self, time_length=5, interval=0.1, major_interval=1, tick_length=10, major_tick_length=20):
        """
        Initialize the Timeline.

        Args:
            time_length (int | float): Length of the timeline in seconds.
            interval (float): Interval between tick marks in seconds.
            major_interval (float): Interval between major tick marks in seconds.
            tick_length (int): Length of the tick marks in pixels.
            major_tick_length (int): Length of the major tick marks in pixels.
        """
        super().__init__()
        self.time_length = time_length  # Length of the timeline in seconds
        # Length of the timeline in pixels. Can only be a whole number of pixels.
        self.pix_length = round(time_length * self.SCALE)
        self.interval = interval
        self.major_interval = major_interval
        self.tick_length = tick_length
        self.major_tick_length = major_tick_length
        self.update_tick_marks()

    def update_tick_marks(self):
        """
        Update the tick marks on the timeline based on the current time length and tick interval fields.
        """

        y_position = 60  # Top positions of the tick marks and labels on the y-axis

        pixel_interval = int(self.interval * self.SCALE)
        pixel_major_interval = int(self.major_interval * self.SCALE)

        # Clear existing tick marks and labels
        for item in self.childItems():
            self.removeFromGroup(item)
            self.scene().removeItem(item)

        for i in range(0, self.pix_length + 1, pixel_interval):
            position = QPointF(i, y_position)
            if i % pixel_major_interval == 0:
                tick_mark = QGraphicsLineItem(
                    QLineF(position, QPointF(position.x(), position.y() + self.major_tick_length)))
                tick_mark.setPen(QPen(Qt.gray, 1))
                self.addToGroup(tick_mark)

                time_label = QGraphicsTextItem(f"{i // pixel_major_interval}")
                text_rect = time_label.boundingRect()
                time_label.setPos(position.x() - text_rect.width() / 2, y_position + self.major_tick_length)
                self.addToGroup(time_label)
            else:
                tick_mark = QGraphicsLineItem(QLineF(position, QPointF(position.x(), position.y() + self.tick_length)))
                tick_mark.setPen(QPen(Qt.gray, 1))
                self.addToGroup(tick_mark)

    def get_pix_length(self):
        """
        Get the length of the timeline in pixels.

        Returns:
            int: The length of the timeline in pixels.
        """
        return self.pix_length

    def get_pos_for_time(self, time):
        """
        Get the x position on the timeline for a given time.

        Args:
            time (int | float): The time in seconds.

        Returns:
            float: The x position on the timeline in pixels.
        """
        return round(time * self.SCALE)

    def get_time_for_pos(self, pos_x):
        """
        Get the time for a given position on the timeline.

        Args:
            pos_x (float): The position on the timeline in pixels.

        Returns:
            float: The time in seconds.
        """
        calc_time = pos_x / self.SCALE
        if calc_time < 0:
            return 0
        elif calc_time > self.time_length:
            return self.time_length
        else:
            return calc_time

    def set_time_length(self, length):
        """
        Set the time length, the pixel length and call for tick markings update.

        Args:
            length (int | float): The new length of the timeline in seconds.
        """
        self.time_length = length
        self.pix_length = round(self.time_length * self.SCALE)
        self.update_tick_marks()

    def get_time_length(self):
        """
        Get the length of the timeline.

        Returns:
            int | float: The length of the timeline in seconds.
        """
        return self.time_length


class KeyframeItem(QGraphicsPixmapItem):
    """
    A QGraphicsPixmapItem representing a keyframe in the timeline.

    Attributes:
        SIZE (int): The size of the keyframe item in pixels. The keyframe is square.
        name (str): The name of the keyframe.
        timeline (Timeline): Reference to the timeline object.
        tick_mark (TickMarkItem): The tick mark item at the bottom of the keyframe.
        hover_info (QGraphicsTextItem): The hover information text item.
    """

    SIZE = 50

    def __init__(self, name, pixmap, position, timeline: Timeline):
        """
        Initialize the KeyframeItem.

        Args:
            name (str): The name of the keyframe.
            pixmap (QPixmap): The pixmap representing the keyframe.
            position (QPointF): The initial position of the keyframe.
            timeline (Timeline): The timeline object.
        """
        super().__init__(pixmap)
        self.setFlags(
            QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.name = name
        # hold onto a reference to the timeline so that update each keyframe based on the timeline.
        # Timeline must be initialized before any itemChange like position is made.
        self.timeline = timeline

        # Add tick mark at the bottom middle of the keyframe box
        half_width = self.boundingRect().width() / 2
        tick_position = QPointF(half_width, self.boundingRect().height())
        self.tick_mark = TickMarkItem(tick_position, 10)  # Length of the tick mark
        self.tick_mark.setPen(QPen(Qt.red, 1))
        self.tick_mark.setParentItem(self)

        # Create hover info item
        self.hover_info = QGraphicsTextItem(f"info text", self)
        self.hover_info.setDefaultTextColor(Qt.white)
        self.hover_info.setPos(0, -20)  # Position above the keyframe
        self.hover_info.hide()  # Hide initially

        self.setPos(position)

    def hoverEnterEvent(self, event):
        """
        Handle hover enter events to show hover info.

        Args:
            event (QGraphicsSceneHoverEvent): The hover event.
        """
        self.show_info()  # Show hover info
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """
        Handle hover leave events to hide hover info.

        Args:
            event (QGraphicsSceneHoverEvent): The hover event.
        """
        if not self.isSelected():
            self.hide_info()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """
        Handle item position changes, avoid keyframe collision, and update time labels.

        Args:
            change (QGraphicsItem.GraphicsItemChange): The type of change.
            value (QVariant): The new value.

        Returns:
            QVariant: The processed value.
        """
        if change == QGraphicsItem.ItemPositionChange:
            half_width = self.boundingRect().width() / 2
            if value.x() < self.timeline.x() - half_width:
                new_x = self.timeline.x() - half_width
            elif value.x() > self.timeline.x() + self.timeline.get_pix_length() - half_width:
                new_x = self.timeline.x() + self.timeline.get_pix_length() - half_width
            else:
                new_x = value.x()

            new_x = self.avoid_collision(new_x, self.x())

            # Update the info text
            time = self.timeline.get_time_for_pos(new_x + half_width)
            self.hover_info.setPlainText(format_time(time))

            return QPointF(new_x, 0)

        return super().itemChange(change, value)

    def avoid_collision(self, new_x, old_x):
        """
        Avoid collision with other keyframe items.

        Args:
            new_x (float): The new x position.
            old_x (float): The old x position.
        """
        # First time this is called is from the constructor so this GraphicsItem has not been added to the scene yet.
        if self.scene() is None:
            return new_x

        # Iterate over all items in the scene
        for item in self.scene().items():
            # Check if the item is a KeyframeItem and is not the current item
            if isinstance(item, KeyframeItem) and item is not self:
                # Check if the x position is taken by another item
                if item.x() == new_x:
                    return old_x
        return new_x

    def trigger_for_edit(self):
        """
        Activate a trigger to signal that a keyframe needs an update.
        """
        new_time = (float(self.timeline.get_time_for_pos(self.x() + self.boundingRect().width() / 2)))
        activate_trigger(KF_EDIT, (self.name, new_time))

    def show_info(self):
        """
        Show the hover info.
        """
        self.hover_info.show()

    def hide_info(self):
        """
        Hide the hover info.
        """
        self.hover_info.hide()

    def set_position_from_time(self, time):
        """
        Set the position of the keyframe based on the time.

        Args:
            time (int | float): The time in seconds.
        """
        new_x = self.timeline.get_pos_for_time(time) - self.boundingRect().width() / 2
        self.setX(new_x)

    def set_info_time(self, time: int | float):
        """
        Set the hover info text based on the time.

        Args:
            time (int | float): The time in seconds.
        """
        self.hover_info.setPlainText(format_time(time))

    def get_name(self):
        """
        Get the name of the keyframe.

        Returns:
            str: The name of the keyframe.
        """
        return self.name


class TimelineCursor(QGraphicsLineItem):
    def __init__(self, position, length, timeline: Timeline):
        super().__init__(QLineF(position, QPointF(position.x(), position.y() + length)))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setPen(QPen(Qt.red, 2))
        self.timeline = timeline
        self.setZValue(1)  # Render in front of other items

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # Called before the position has changed.

            # Clamp the cursor to the timeline
            rtn_point = QPointF(value.x(), self.y())
            if value.x() < self.timeline.x():
                rtn_point = QPointF(self.timeline.x(), self.y())
            elif value.x() > self.timeline.x() + self.timeline.get_pix_length():
                rtn_point = QPointF(self.timeline.x() + self.timeline.get_pix_length(), self.y())
            return rtn_point

        return super().itemChange(change, value)

    def set_pos_from_time(self, time):
        new_x = self.timeline.get_pos_for_time(time)
        self.setX(new_x)

    def get_time(self):
        return self.timeline.get_time_for_pos(self.x())

    def mouseReleaseEvent(self, event):
        self.activate_preview_trigger()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event, QGraphicsSceneMouseEvent=None):
        self.activate_preview_trigger()
        super().mouseMoveEvent(event)

    def activate_preview_trigger(self):
        activate_trigger(PREVIEW, round(self.get_time(), 2))


class TickMarkItem(QGraphicsLineItem):
    def __init__(self, position, length):
        super().__init__(QLineF(position, QPointF(position.x(), position.y() + length)))
        self.setPen(QPen(Qt.gray, 1))


class TickIntervalLabel(QGraphicsTextItem):
    def __init__(self, text):
        super().__init__(text)
