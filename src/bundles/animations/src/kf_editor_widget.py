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

from dataclasses import dataclass, field
from typing import List, Optional

from Qt.QtCore import (
    QByteArray,
    QEvent,
    QLineF,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from Qt.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QPolygonF,
    QResizeEvent,
    QTransform,
)
from Qt.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneResizeEvent,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .animation import format_time
from .triggers import (
    INSERT_TIME,
    KF_ADD,
    KF_DELETE,
    KF_EDIT,
    LENGTH_CHANGE,
    MGR_FRAME_PLAYED,
    MGR_KF_ADDED,
    MGR_KF_DELETED,
    MGR_KF_EDITED,
    MGR_LENGTH_CHANGED,
    MGR_PREVIEWED,
    MGR_RECORDING_START,
    MGR_RECORDING_STOP,
    PLAY,
    PREVIEW,
    RECORD,
    REMOVE_TIME,
    STOP_PLAYING,
    STOP_RECORDING,
    activate_trigger,
    add_handler,
    remove_handler,
)


@dataclass
class Keyframe:
    time: int

@dataclass
class Track:
    keyframes: List[Keyframe] = field(default_factory=list)

FPS = 30
PX_PER_FRAME = 10
_TIMELINE_OFFSET_PIXELS = 100
_TIMELINE_MINOR_TICK_HEIGHT = 10
_TIMELINE_MAJOR_TICK_HEIGHT = 1.7 * _TIMELINE_MINOR_TICK_HEIGHT
_PIXELS_PER_SECOND = 60

_TIMELINE_HEIGHT = 48
_TRACK_HEIGHT = 48
MIN_TIMELINE_WIDTH = 200
EDGE_PAD_PIXELS = 50
DRAG_THRESHOLD = 5

_COLORS = {
    "white": QColor("#FFFFFF"),
    "black": QColor("#000000"),
    "red":   QColor("#FF0000"),
    "grid":  QColor("#D3D3D3"),
    "text":  QColor("#808080"),
}

COLORS = {
    "playhead": _COLORS["red"],
}

class TimelineModel(QObject):
    """Stores tracks/keyframes and notifies observers when data mutate."""

    changed = Signal()

    def __init__(self, tracks: int = 3):
        super().__init__()
        self.tracks: List[Track] = [Track() for _ in range(tracks)]

    def add_keyframe(self, track: int, frame: int, pix: QPixmap | None = None):
        self.tracks[track].keyframes.append(Keyframe(frame, pix))
        self.tracks[track].keyframes.sort(key=lambda k: k.time)
        self.changed.emit()

    def delete_keyframe(self, kf: Keyframe, track_idx: int):
        track = self.tracks[track_idx]
        if kf in track.keyframes:
            track.keyframes.remove(kf)
            self.changed.emit()

    def add_track(self, name, object):
        #new_track = Track(object)
        new_label = Label(name)
        self.tracks.append(Track())
        self.labels.append(new_label)
        self.name_to_label[name] = new_label
        self.label_to_object[new_label] = object
        self.changed.emit()

    def delete_track(self, idx: int = -1):
        if len(self.tracks) <= 1:
            return  # maintain at least one track
        self.tracks.pop(idx)
        self.changed.emit()




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

    def __init__(self, keyframes):
        """
        Initialize the KeyframeEditorWidget.

        Args:
            length (int | float): Length of the timeline in seconds.
            keyframes (list): List of animation.Keyframe objects.
        """
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.handlers = []

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
        #self.play_button.clicked.connect(
        #    lambda: activate_trigger(PLAY, (self.kfe_scene.get_cursor().get_time(), False)))

        # Pause button
        self.pause_button = QPushButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.button_layout.addWidget(self.pause_button)
        #self.pause_button.clicked.connect(lambda: activate_trigger(STOP_PLAYING, None))

        # Fast forward button
        self.fast_forward_button = QPushButton()
        self.fast_forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.button_layout.addWidget(self.fast_forward_button)
        self.fast_forward_button.clicked.connect(self.fast_forward)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        self.button_layout.addWidget(separator)

        # Record button
        #self.start_recording_icon = QIcon(
        #    os.path.join(os.path.dirname(__file__), "resources", "red_circle.png")
        #)
        #self.stop_recording_icon = QIcon(
        #    os.path.join(os.path.dirname(__file__), "resources", "red_square.png")
        #)
        #self.record_button = QPushButton(
        #    parent, text="Record Command", icon=self.start_recording_icon
        #)
        self.record_button = QPushButton()
        self.record_button.setIcon(self.style().standardIcon(QStyle.SP_DialogNoButton))
        self.button_layout.addWidget(self.record_button)
        self.record_button.clicked.connect(self.recording_toggle)

        self.handlers.append(
            add_handler(MGR_RECORDING_START, lambda trigger_name, data: self.record_button.setChecked(True)))
        self.handlers.append(
            add_handler(MGR_RECORDING_STOP, lambda trigger_name, data: self.record_button.setChecked(False)))

        self.button_layout.addStretch ()

        self.layout.addLayout(self.button_layout)

        # Keyframe editor graphics view widget
        self.kfe_view = KFEGraphicsView(self)
        # TODO: Calculate how many seconds fit into the bounding box of whatever
        # size we are and use that instead of 5
        self.kfe_scene = KeyframeEditorScene(keyframes)
        self.kfe_view.setScene(self.kfe_scene)
        self.kfe_scene.initialize()
        self.layout.addWidget(self.kfe_view)

        #button = QPushButton("click me")
        #proxy = QGraphicsProxyWidget()
        #proxy.setWidget(button)
        #proxy.setPos(10,10)
        #self.kfe_scene.addItem(proxy)

        # Connect time label triggers. Must be done after the KeyframeEditor widget is created because depends on
        # the keyframe editor scene.
        # This handler is needed for when the cursor is moved but there is no manager preview call. This happens if
        # There are no keyframes in the animation.

    def event(self, event):
        if event.type() == QEvent.Type.Resize:
            view_size = self.kfe_view.viewport().size()
            self.kfe_scene.setSceneRect(0, 0, max(view_size.width(), self.kfe_scene.itemsBoundingRect().width()), max(view_size.height(), self.kfe_scene.itemsBoundingRect().height()))
            self.kfe_scene.resizeEvent()
        return QWidget.event(self, event)

    def rewind(self):
        """
        Rewind the timeline to the beginning.
        """
        #activate_trigger(STOP_PLAYING, None)
        cursor = self.kfe_scene.get_cursor()
        cursor.set_pos_from_time(0)
        self.kfe_view.horizontalScrollBar().setValue(0)
        self.kfe_scene.get_cursor().activate_preview_trigger()

    def fast_forward(self):
        """
        Fast-forward the timeline to the end.
        """
        #activate_trigger(STOP_PLAYING, None)
        cursor = self.kfe_scene.get_cursor()
        timeline_len = self.kfe_scene.timeline.get_time_length()
        cursor.set_pos_from_time(timeline_len)
        self.kfe_view.horizontalScrollBar().setValue(self.kfe_view.horizontalScrollBar().maximum())
        self.kfe_scene.get_cursor().activate_preview_trigger()

    def delete_keyframes(self):
        """
        Delete selected keyframes.
        """
        ...
        #keyframes = self.kfe_scene.get_selected_keyframes()
        #for keyframe in keyframes:
        #    activate_trigger(KF_DELETE, keyframe.get_name())

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
            #activate_trigger(RECORD, None)
        else:
            # Same reset situation here as above
            self.record_button.setChecked(False)
            #activate_trigger(STOP_RECORDING, None)

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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

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
    def __init__(self, keyframes):
        """
        Initialize the KeyframeEditorScene.

        Args:
            length (int | float): Length of the timeline in seconds.
            keyframes (list): List of animation.Keyframe objects.
        """
        super().__init__()
        self.handlers = []
        self.timeline = Timeline()
        self.addItem(self.timeline)
        self.cursor = TimelineCursor(QPointF(_TIMELINE_OFFSET_PIXELS, 0), self.height(), self.timeline)
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

    def initialize(self):
        #self.timeline.update_tick_marks(int(self.width()))
        self.timeline.add_track()
        self.timeline.add_track()

    def resizeEvent(self):
        #self.timeline.update_tick_marks(int(self.width()))
        self.cursor.set_height(self.height())

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
        self.setSceneRect(-margin, 0, self.width(), self.height())

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
        if event.button() == Qt.MouseButton.LeftButton:
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


class TimelineTrack(QGraphicsItemGroup):
    """
    one horizontal strip: [ label | timeline lane ]
    emits clicked(model) when either sub-item is pressed
    """
    clicked = Signal(object)

    def __init__(self, model, y_pos=0., initial_width: int = 10, parent=None):
        super().__init__(parent)
        self.model = model

        # label
        txt = QGraphicsTextItem(model, self)
        txt.setPos(0, 0)
        txt.setTextWidth(_TIMELINE_OFFSET_PIXELS)  # wraps if needed
        self.addToGroup(txt)

        # lane
        lane_rect = QGraphicsRectItem(
            QRectF(_TIMELINE_OFFSET_PIXELS, 0, initial_width - _TIMELINE_OFFSET_PIXELS, _TRACK_HEIGHT), self)
        #lane_rect.setBrush(QBrush(QColor("#1e1e1e")))
        lane_rect.setBrush(QBrush(_COLORS["red"]))
        lane_rect.setPen(QPen(Qt.NoPen))
        self.addToGroup(lane_rect)

        # group housekeeping
        #self.setHandlesChildEvents(False)  # we capture clicks ourselves
        self.setPos(0, y_pos)

    # mouse
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.model)
            event.accept()
        else:
            super().mousePressEvent(event)

class Timeline(QGraphicsItemGroup):
    """
    A QGraphicsItemGroup representing the timeline markings.

    Attributes:
        pix_length (int): Length of the timeline in pixels.
        interval (float): Interval between tick marks in seconds.
        major_interval (float): Interval between major tick marks in seconds.
    """

    def __init__(self, interval=0.1, major_interval=1):
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
        self.tracks = []
        self.time_length = 10  # Length of the timeline in seconds
        # Length of the timeline in pixels. Can only be a whole number of pixels.
        # TODO: Just use the time of the last keyframe as the length of the timeline.
        # When users drag keyframes to the right the timeline should expand automatically.
        self.pix_length = round(self.time_length * _PIXELS_PER_SECOND)
        self.interval = interval
        self.major_interval = major_interval

    def add_track(self):
        track = TimelineTrack("Track %s" % (len(self.tracks) + 1), y_pos=(len(self.tracks) + 1) * _TRACK_HEIGHT, initial_width=int(self.scene().width()))
        self.tracks.append(track)
        self.addToGroup(track)

    def update_tick_marks(self, width):
        """
        Update the tick marks on the timeline based on the current time length and tick interval fields.
        """

        y_position = 0  # Top positions of the tick marks and labels on the y-axis

        pixel_interval = int(self.interval * _PIXELS_PER_SECOND)
        pixel_major_interval = int(self.major_interval * _PIXELS_PER_SECOND)

        # Clear existing tick marks and labels
        for item in self.childItems():
            self.removeFromGroup(item)
            self.scene().removeItem(item)

        i = _TIMELINE_OFFSET_PIXELS
        for i in range(_TIMELINE_OFFSET_PIXELS, width, pixel_interval):
            position = QPointF(i, y_position)
            if (i - _TIMELINE_OFFSET_PIXELS) % pixel_major_interval == 0:
                tick_mark = QGraphicsLineItem(
                    QLineF(position, QPointF(position.x(), position.y() + _TIMELINE_MAJOR_TICK_HEIGHT)))
                tick_mark.setPen(QPen(Qt.gray, 1))
                self.addToGroup(tick_mark)

                time_label = QGraphicsTextItem(f"{i // pixel_major_interval}")
                text_rect = time_label.boundingRect()
                time_label.setPos(position.x() - text_rect.width() / 2, y_position + _TIMELINE_MAJOR_TICK_HEIGHT)
                self.addToGroup(time_label)
            else:
                tick_mark = QGraphicsLineItem(QLineF(position, QPointF(position.x(), position.y() + _TIMELINE_MINOR_TICK_HEIGHT)))
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
        return round(time * _PIXELS_PER_SECOND)

    def get_time_for_pos(self, pos_x):
        """
        Get the time for a given position on the timeline.

        Args:
            pos_x (float): The position on the timeline in pixels.

        Returns:
            float: The time in seconds.
        """
        calc_time = pos_x / _PIXELS_PER_SECOND
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
        self.pix_length = round(self.time_length * _PIXELS_PER_SECOND)
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
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
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
        #activate_trigger(KF_EDIT, (self.name, new_time))

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


class TimelineCursor(QGraphicsItemGroup):
    def __init__(self, position, length, timeline: Timeline):
        super().__init__()
        self.line = QGraphicsLineItem(QLineF(position, QPointF(position.x(), position.y() + length)))
        self.addToGroup(self.line)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.line.setPen(QPen(COLORS["playhead"], 2))

        poly = QPolygonF([
            QPointF(-6 + position.x(), 0),   # left top
            QPointF( 6 + position.x(), 0),   # right top
            QPointF( 6 + position.x(), 10),  # right shoulder
            QPointF( 0 + position.x(), 18),  # bottom tip
            QPointF(-6 + position.x(), 10),  # left shoulder
        ])
        head = QGraphicsPolygonItem(poly)
        head.setBrush(COLORS["playhead"])
        #head.setPen(Qt.PenStyle.NoPen)
        self.addToGroup(head)
        self.timeline = timeline
        self.setZValue(1)  # Render in front of other items

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Called before the position has changed.

            # Clamp the cursor to the timeline
            rtn_point = QPointF(value.x(), self.y())
            if value.x() < self.timeline.x():
                rtn_point = QPointF(self.timeline.x(), self.y())
            elif value.x() > self.timeline.x() + self.timeline.get_pix_length():
                rtn_point = QPointF(self.timeline.x() + self.timeline.get_pix_length(), self.y())
            return rtn_point

        return super().itemChange(change, value)

    def set_height(self, height):
        line = self.line.line()                 # copy
        line.setP2(QPointF(line.p2().x(), height))
        self.line.setLine(line)

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
        ...
        #activate_trigger(PREVIEW, round(self.get_time(), 2))


class TickMarkItem(QGraphicsLineItem):
    def __init__(self, position, length):
        super().__init__(QLineF(position, QPointF(position.x(), position.y() + length)))
        self.setPen(QPen(Qt.gray, 1))


class TickIntervalLabel(QGraphicsTextItem):
    def __init__(self, text):
        super().__init__(text)
