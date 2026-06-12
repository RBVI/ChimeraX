# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
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

"""Keyframe timeline implementation for the animations tool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from Qt.QtCore import QEasingCurve, QObject, QPointF, QRectF, Qt, QTimer, QSize, Signal, Slot
from Qt.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from Qt.QtWidgets import (
    QAbstractItemView,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSizePolicy,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFrame,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QGridLayout,
)

from chimerax.scenes.tool import SCENE_EVENT_MIME_FORMAT

__all__ = ["KeyframeTimelineWidget", "KeyframeEditorWidget"]

TRACK_HEIGHT = 24
RULER_HEIGHT = 18
FRAME_WIDTH = 12

KEYFRAME_SIZE = 8
KEYFRAME_BRUSH = QBrush(QColor("#FFD24C"))
KEYFRAME_PEN = QPen(QColor("#A67C00"))

CLIP_BRUSH = QBrush(QColor(0, 170, 255, 120))
CLIP_PEN = QPen(QColor(0, 120, 180))

GRID_PEN = QPen(QColor("#888"))
GRID_PEN.setStyle(Qt.DashLine)
GRID_PEN.setWidthF(0.5)

SEPARATOR_PEN = QPen(QColor("#444"))
SEPARATOR_PEN.setWidth(1)


@dataclass
class KeyframeData:
    frame: int
    value: any  # noqa: ANN401


@dataclass
class ClipData:
    start: int
    end: int
    payload: any  # noqa: ANN401


class KeyframeItem(QGraphicsRectItem):
    """Visual representation of a keyframe."""

    def __init__(self, data: KeyframeData, parent: Optional[QGraphicsItem] = None):
        super().__init__(-KEYFRAME_SIZE / 2, -KEYFRAME_SIZE / 2, KEYFRAME_SIZE, KEYFRAME_SIZE, parent)
        self.setBrush(KEYFRAME_BRUSH)
        self.setPen(KEYFRAME_PEN)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(10)
        self.data = data
        self._drag_start_frame: int | None = None

    def mousePressEvent(self, event):  # noqa: D401, N802
        self._drag_start_frame = self.data.frame
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401, N802
        super().mouseMoveEvent(event)

    def itemChange(self, change, value):  # noqa: D401, N802
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_x = value.x()
            if hasattr(self, "_track_row") and self._track_row:
                track_y = self._track_row.index * TRACK_HEIGHT + RULER_HEIGHT + TRACK_HEIGHT / 2
                value.setY(track_y)

            frame_width = getattr(self.scene(), "frame_width", FRAME_WIDTH)
            frame = round(new_x / frame_width)
            frame = max(frame, 0)
            value.setX(frame * frame_width)
            self.data.frame = frame

            if hasattr(self.scene(), "extend_timeline") and frame > self.scene().num_frames:
                self.scene().extend_timeline(frame + 24)

            return value
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.isSelected():
            selection_pen = QPen(QColor("#FFFFFF"))
            selection_pen.setWidth(2)
            painter.setPen(selection_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(-1, -1, 1, 1))


class ClipItem(QGraphicsRectItem):
    """Visual representation of a clip spanning multiple frames."""

    def __init__(self, data: ClipData, parent: Optional[QGraphicsItem] = None):
        x = data.start * FRAME_WIDTH
        width = (data.end - data.start) * FRAME_WIDTH
        super().__init__(x, 0, width, TRACK_HEIGHT, parent)
        self.setBrush(CLIP_BRUSH)
        self.data = data
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)

    def itemChange(self, change, value):  # noqa: D401, N802
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_x = value.x()
            frame_width = getattr(self.scene(), "frame_width", FRAME_WIDTH)
            frame = round(new_x / frame_width)
            frame = max(frame, 0)
            duration = self.data.end - self.data.start
            self.data.start = frame
            self.data.end = frame + duration
            value.setX(frame * frame_width)
            return value
        return super().itemChange(change, value)


class TrackRow(QGraphicsItemGroup):
    """Group holding all graphics items for a single track row."""

    def __init__(self, index: int, name: str, scene_width: int):
        super().__init__()
        self.index = index
        self.name = name
        self.track_name = name

        y = index * TRACK_HEIGHT + RULER_HEIGHT
        self.setPos(0, y)

        self.bg = QGraphicsRectItem(0, 0, scene_width, TRACK_HEIGHT, self)
        self.bg.setBrush(QColor("#202020"))
        self.bg.setPen(QPen(Qt.NoPen))
        self.bg.setZValue(-1)
        self.bg.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.bg.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.bg.setAcceptedMouseButtons(Qt.NoButton)

        self.sep = QGraphicsRectItem(0, TRACK_HEIGHT - 1, scene_width, 1, self)
        self.sep.setBrush(SEPARATOR_PEN.color())
        self.sep.setPen(QPen(Qt.NoPen))
        self.sep.setZValue(-1)
        self.sep.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.sep.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.sep.setAcceptedMouseButtons(Qt.NoButton)

        self.addToGroup(self.bg)
        self.addToGroup(self.sep)
        self._highlighted = False
        self._hovered = False
        self.setAcceptHoverEvents(True)

    def set_highlighted(self, highlighted: bool):
        self._highlighted = highlighted
        self._update_background()

    def set_hovered(self, hovered: bool):
        self._hovered = hovered
        self._update_background()

    def _update_background(self):
        if self._highlighted:
            self.bg.setBrush(QColor("#3D5A80"))
        elif self._hovered:
            self.bg.setBrush(QColor("#2A3F5F"))
        else:
            self.bg.setBrush(QColor("#202020"))

    def hoverEnterEvent(self, event):
        if hasattr(self.scene(), "views") and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, "track_hovered"):
                view.track_hovered.emit(self.index, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if hasattr(self.scene(), "views") and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, "track_hovered"):
                view.track_hovered.emit(self.index, False)
        super().hoverLeaveEvent(event)

    def add_keyframe(self, kf_data: KeyframeData):
        kf = KeyframeItem(kf_data)
        frame_width = getattr(self.scene(), "frame_width", FRAME_WIDTH) if self.scene() else FRAME_WIDTH
        track_y = self.pos().y()
        kf.setPos(kf_data.frame * frame_width, track_y + TRACK_HEIGHT / 2)
        if self.scene():
            self.scene().addItem(kf)
        kf._track_row = self
        return kf

    def add_clip(self, clip_data: ClipData):
        clip = ClipItem(clip_data, self)
        frame_width = getattr(self.scene(), "frame_width", FRAME_WIDTH) if self.scene() else FRAME_WIDTH
        track_y = self.pos().y()
        clip.setPos(clip_data.start * frame_width, track_y)
        self.addToGroup(clip)
        return clip


class TimelineScene(QGraphicsScene):
    """Scene containing timeline ruler and track rows."""

    def __init__(self, num_frames: int = 240, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.num_frames = num_frames
        self.zoom_factor = 1.0
        self.base_frame_width = 12
        self.track_rows: List[TrackRow] = []
        self.current_frame = 0
        self.playhead_line = None
        self.ruler_items = []
        self.grid_items = []
        self.setBackgroundBrush(QColor("#303030"))
        self._draw_ruler()
        self._draw_frame_grid()
        self._draw_playhead()

    @property
    def frame_width(self):
        return self.base_frame_width * self.zoom_factor

    def set_zoom(self, factor: float):
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(factor, 10.0))
        if old_zoom != self.zoom_factor:
            self._redraw_timeline()

    def zoom_in(self):
        self.set_zoom(self.zoom_factor * 1.2)

    def zoom_out(self):
        self.set_zoom(self.zoom_factor / 1.2)

    def extend_timeline(self, new_num_frames: int):
        if new_num_frames > self.num_frames:
            self.num_frames = new_num_frames
            self._redraw_timeline()

    def _redraw_timeline(self):
        for item in self.ruler_items:
            self.removeItem(item)
        for item in self.grid_items:
            self.removeItem(item)
        self.ruler_items.clear()
        self.grid_items.clear()

        self._draw_ruler()
        self._draw_frame_grid()
        self._update_scene_rect()
        self._update_track_positions()
        self._draw_playhead()

    def _draw_ruler(self):
        font = QFont()
        font.setPointSize(8)
        if self.zoom_factor >= 2.0:
            tick_interval = 1
        elif self.zoom_factor >= 0.5:
            tick_interval = 5
        else:
            tick_interval = 10

        for frame in range(0, self.num_frames + 1, tick_interval):
            x = frame * self.frame_width
            tick = QGraphicsRectItem(x, 0, 1, 6)
            tick.setBrush(QColor("#CCCCCC"))
            tick.setPen(QPen(Qt.NoPen))
            self.addItem(tick)
            self.ruler_items.append(tick)

            if frame % (tick_interval * 2) == 0 or tick_interval == 1:
                label = self.addText(str(frame), font)
                label.setDefaultTextColor(QColor("#CCCCCC"))
                label.setPos(x + 2, 0)
                self.ruler_items.append(label)

    def _draw_frame_grid(self):
        if self.zoom_factor >= 2.0:
            grid_interval = 1
        elif self.zoom_factor >= 1.0:
            grid_interval = 5
        else:
            grid_interval = 10

        for frame in range(0, self.num_frames + 1, grid_interval):
            x = frame * self.frame_width
            tick_height = 8 if frame % (grid_interval * 2) == 0 else 4
            line = self.addLine(x, RULER_HEIGHT - tick_height, x, RULER_HEIGHT, GRID_PEN)
            line.setZValue(-1)
            self.grid_items.append(line)

    def _update_scene_rect(self):
        visible_track_count = sum(1 for track in self.track_rows if track.isVisible())
        height = RULER_HEIGHT + visible_track_count * TRACK_HEIGHT
        width = self.num_frames * self.frame_width
        self.setSceneRect(0, 0, width, height)

    def _update_track_positions(self):
        for track_row in self.track_rows:
            for item in self.items():
                if isinstance(item, KeyframeItem) and hasattr(item, "_track_row") and item._track_row == track_row:
                    frame = item.data.frame
                    track_y = track_row.index * TRACK_HEIGHT + RULER_HEIGHT + TRACK_HEIGHT / 2
                    item.setPos(frame * self.frame_width, track_y)
                elif isinstance(item, ClipItem) and item.parentItem() == track_row:
                    clip_data = item.data
                    clip_x = clip_data.start * self.frame_width
                    clip_width = (clip_data.end - clip_data.start) * self.frame_width
                    clip_y = track_row.index * TRACK_HEIGHT + RULER_HEIGHT
                    item.setPos(clip_x, clip_y)
                    item.setRect(0, 0, clip_width, TRACK_HEIGHT)

    def add_track(self, name: str):
        row = TrackRow(len(self.track_rows), name, self.num_frames * self.frame_width)
        self.track_rows.append(row)
        self.addItem(row)
        self._draw_frame_grid()
        self._update_scene_rect()
        self._draw_playhead()
        return row

    def insert_track(self, position: int, name: str):
        row = TrackRow(position, name, self.num_frames * self.frame_width)
        self.track_rows.insert(position, row)

        for i in range(position, len(self.track_rows)):
            self.track_rows[i].index = i
            new_y = i * TRACK_HEIGHT + RULER_HEIGHT
            self.track_rows[i].setPos(0, new_y)

        self.addItem(row)
        self._draw_frame_grid()
        self._update_scene_rect()
        self._draw_playhead()
        return row

    def set_track_visible(self, track_index: int, visible: bool):
        if track_index < len(self.track_rows):
            track_row = self.track_rows[track_index]
            track_row.setVisible(visible)
            self._update_track_positions_for_visibility()
            self._update_scene_rect()
            self._draw_frame_grid()
            self._draw_playhead()
            self.update()
            for view in self.views():
                view.update()
                view.viewport().update()

    def _update_track_positions_for_visibility(self):
        visible_track_count = 0
        for track_row in self.track_rows:
            if track_row.isVisible():
                new_y_pos = visible_track_count * TRACK_HEIGHT + RULER_HEIGHT
                track_row.setPos(0, new_y_pos)
                track_row.update()
                track_row.visual_index = visible_track_count

                for item in self.items():
                    if (
                        isinstance(item, KeyframeItem)
                        and hasattr(item, "_track_row")
                        and item._track_row == track_row
                        and item.isVisible()
                    ):
                        kf_y = new_y_pos + TRACK_HEIGHT / 2
                        item.setPos(item.pos().x(), kf_y)

                visible_track_count += 1
            else:
                track_row.setPos(0, -1000)

    def _draw_playhead(self):
        if self.playhead_line is not None:
            self.removeItem(self.playhead_line)

        x = self.current_frame * self.frame_width
        visible_track_count = sum(1 for track in self.track_rows if track.isVisible())
        height = RULER_HEIGHT + visible_track_count * TRACK_HEIGHT

        playhead_pen = QPen(QColor("#FF6B6B"))
        playhead_pen.setWidth(2)
        self.playhead_line = self.addLine(x, 0, x, height, playhead_pen)
        self.playhead_line.setZValue(10)

    def set_current_frame(self, frame: int):
        frame = max(0, min(frame, self.num_frames))
        if frame != self.current_frame:
            self.current_frame = frame
            self._draw_playhead()

    def get_current_frame(self) -> int:
        return self.current_frame


class TimelineView(QGraphicsView):
    row_clicked = Signal(int)
    track_hovered = Signal(int, bool)
    frame_changed = Signal(int)
    keyframes_deleted = Signal(list)
    scene_dropped = Signal(str, int)
    track_deleted = Signal(int)

    def __init__(self, scene: TimelineScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameStyle(QGraphicsView.NoFrame)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)
        self._dragging_playhead = False
        self.setAcceptDrops(True)
        self._last_clicked_track = -1

    def wheelEvent(self, event):  # noqa: N802, D401
        if event.modifiers() & Qt.ControlModifier or True:
            angle_delta = event.angleDelta().y()
            if angle_delta > 0:
                self.scene().zoom_in()  # type: ignore[attr-defined]
            else:
                self.scene().zoom_out()  # type: ignore[attr-defined]
        else:
            super().wheelEvent(event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            item_at_pos = self.itemAt(event.position().toPoint())
            if isinstance(item_at_pos, KeyframeItem):
                super().mousePressEvent(event)
                return

            if 0 <= pos.y() <= RULER_HEIGHT:
                frame_width = self.scene().frame_width  # type: ignore[attr-defined]
                frame = round(pos.x() / frame_width)
                frame = max(0, min(frame, self.scene().num_frames))  # type: ignore[attr-defined]
                self.scene().set_current_frame(frame)  # type: ignore[attr-defined]
                self.frame_changed.emit(frame)
                self._dragging_playhead = True
                return

            scene_height = RULER_HEIGHT + len(self.scene().track_rows) * TRACK_HEIGHT  # type: ignore[arg-type]
            if RULER_HEIGHT < pos.y() <= scene_height:
                y = pos.y() - RULER_HEIGHT
                index = int(y // TRACK_HEIGHT)
                if 0 <= index < len(self.scene().track_rows):  # type: ignore[arg-type]
                    self._last_clicked_track = index
                    self.row_clicked.emit(index)
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_playhead:
            pos = self.mapToScene(event.position().toPoint())
            frame_width = self.scene().frame_width  # type: ignore[attr-defined]
            frame = round(pos.x() / frame_width)
            frame = max(0, min(frame, self.scene().num_frames))  # type: ignore[attr-defined]
            self.scene().set_current_frame(frame)  # type: ignore[attr-defined]
            self.frame_changed.emit(frame)

            if frame > self.scene().num_frames - 12:  # type: ignore[attr-defined]
                self.scene().extend_timeline(frame + 24)  # type: ignore[attr-defined]
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            selected_items = self.scene().selectedItems()
            keyframes_to_delete = []

            for item in selected_items:
                if isinstance(item, KeyframeItem):
                    keyframes_to_delete.append(item)

            if keyframes_to_delete:
                self._delete_keyframes(keyframes_to_delete)
                self.keyframes_deleted.emit(keyframes_to_delete)
            else:
                if hasattr(self, "_last_clicked_track") and self._last_clicked_track >= 0:
                    self.track_deleted.emit(self._last_clicked_track)
        else:
            super().keyPressEvent(event)

    def _delete_keyframes(self, keyframes):
        for keyframe in keyframes:
            self.scene().removeItem(keyframe)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            pos = self.mapToScene(event.position().toPoint())
            frame_width = self.scene().frame_width
            frame = max(0, round(pos.x() / frame_width))

            scene_data_bytes = event.mimeData().data(SCENE_EVENT_MIME_FORMAT)
            try:
                import json

                scene_data = json.loads(scene_data_bytes.data().decode("utf-8"))
                scene_name = scene_data.get("name", "Unknown Scene")
            except Exception:
                scene_name = scene_data_bytes.data().decode("utf-8")

            self.scene_dropped.emit(scene_name, frame)
            event.acceptProposedAction()
        else:
            event.ignore()


class TrackHeaderView(QListWidget):
    track_selected = Signal(int)
    track_hovered = Signal(int, bool)
    track_collapsed = Signal(int)
    track_deleted = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.setFixedWidth(120)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameStyle(QListWidget.NoFrame)
        self.setViewportMargins(0, RULER_HEIGHT, 0, 0)
        self.setStyleSheet(
            """
            QListWidget::item:selected {
                background-color: #3D5A80;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2A3F5F;
            }
            """
        )
        self.setMouseTracking(True)
        self._last_hovered_index = -1
        self.track_widgets = {}

    def add_track(self, name: str, is_parent: bool = False):
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, TRACK_HEIGHT))
        self.addItem(item)

        track_index = self.count() - 1
        track_widget = TrackItemWidget(track_index, name, is_parent)
        track_widget.expand_clicked.connect(self._on_track_collapsed)

        self.setItemWidget(item, track_widget)
        self.track_widgets[track_index] = track_widget
        return item

    def add_subtrack(self, name: str, parent_index: int):
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, TRACK_HEIGHT))
        self.addItem(item)

        track_index = self.count() - 1
        track_widget = TrackItemWidget(track_index, name, is_parent=False)

        self.setItemWidget(item, track_widget)
        self.track_widgets[track_index] = track_widget
        return item

    def insert_track(self, position: int, name: str, is_parent: bool = False):
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, TRACK_HEIGHT))
        self.insertItem(position, item)

        old_widgets = dict(self.track_widgets)
        self.track_widgets.clear()

        for old_index, widget in old_widgets.items():
            if old_index >= position:
                new_index = old_index + 1
                widget.track_index = new_index
                self.track_widgets[new_index] = widget
            else:
                self.track_widgets[old_index] = widget

        track_widget = TrackItemWidget(position, name, is_parent)
        track_widget.expand_clicked.connect(self._on_track_collapsed)

        self.setItemWidget(item, track_widget)
        self.track_widgets[position] = track_widget
        return item

    def set_track_as_parent(self, track_index: int):
        if track_index in self.track_widgets:
            widget = self.track_widgets[track_index]
            if not widget.is_parent:
                item = self.item(track_index)
                name = widget.name_label.text()
                new_widget = TrackItemWidget(track_index, name, is_parent=True)
                new_widget.expand_clicked.connect(self._on_track_collapsed)
                self.setItemWidget(item, new_widget)
                self.track_widgets[track_index] = new_widget

    def _on_track_collapsed(self, track_index: int):
        self.track_collapsed.emit(track_index)

    def set_track_expanded(self, track_index: int, expanded: bool):
        if track_index in self.track_widgets:
            self.track_widgets[track_index].set_expanded(expanded)

    def set_track_visible(self, track_index: int, visible: bool):
        if track_index < self.count():
            item = self.item(track_index)
            if item:
                if visible:
                    item.setHidden(False)
                    item.setSizeHint(QSize(0, TRACK_HEIGHT))
                else:
                    item.setHidden(True)
                    item.setSizeHint(QSize(0, 0))

    @Slot()
    def _on_selection_changed(self):
        selected_indexes = self.selectedIndexes()
        if selected_indexes:
            self.track_selected.emit(selected_indexes[0].row())

    def mouseMoveEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item:
            index = self.row(item)
            if index != self._last_hovered_index:
                if self._last_hovered_index >= 0:
                    self.track_hovered.emit(self._last_hovered_index, False)
                self.track_hovered.emit(index, True)
                self._last_hovered_index = index
        else:
            if self._last_hovered_index >= 0:
                self.track_hovered.emit(self._last_hovered_index, False)
                self._last_hovered_index = -1
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._last_hovered_index >= 0:
            self.track_hovered.emit(self._last_hovered_index, False)
            self._last_hovered_index = -1
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            current_row = self.currentRow()
            if current_row >= 0:
                self.track_deleted.emit(current_row)
        else:
            super().keyPressEvent(event)


class TrackItemWidget(QWidget):
    """Custom widget for track items with expand/collapse button."""

    expand_clicked = Signal(int)

    def __init__(self, track_index: int, name: str, is_parent: bool = False):
        super().__init__()
        self.track_index = track_index
        self.is_parent = is_parent
        self.is_expanded = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)

        if is_parent:
            self.expand_btn = QPushButton("▼")
            self.expand_btn.setFixedSize(16, 16)
            self.expand_btn.setStyleSheet(
                """
                QPushButton {
                    border: none;
                    background: transparent;
                    color: #CCCCCC;
                    font-size: 10px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 2px;
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.2);
                }
                """
            )
            self.expand_btn.clicked.connect(self._on_expand_clicked)
            layout.addWidget(self.expand_btn)
        else:
            layout.addSpacing(20)

        self.name_label = QLabel(name)
        layout.addWidget(self.name_label)
        layout.addStretch()

    def _on_expand_clicked(self):
        self.is_expanded = not self.is_expanded
        self.expand_btn.setText("▼" if self.is_expanded else "▶")
        self.expand_clicked.emit(self.track_index)

    def set_expanded(self, expanded: bool):
        self.is_expanded = expanded
        if hasattr(self, "expand_btn"):
            self.expand_btn.setText("▼" if expanded else "▶")


class ModelSelectionPanel(QWidget):
    """Panel for selecting models and adding tracks to the timeline."""

    track_requested = Signal(object)

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel("Model Animation")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        model_frame = QFrame()
        model_layout = QVBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Select Model:"))

        from chimerax.ui.widgets import ModelMenuButton

        self.model_menu = ModelMenuButton(self.session, no_value_button_text="Choose model...")
        self.model_menu.value_changed.connect(self.on_model_changed)
        model_layout.addWidget(self.model_menu)

        layout.addWidget(model_frame)

        self.add_track_button = QPushButton("Add Track")
        self.add_track_button.clicked.connect(self.add_track)
        self.add_track_button.setEnabled(False)
        layout.addWidget(self.add_track_button)
        layout.addStretch()

    def on_model_changed(self):
        model = self.model_menu.value
        self.add_track_button.setEnabled(model is not None)

    def add_track(self):
        model = self.model_menu.value
        if model:
            self.track_requested.emit(model)


class PlaceEditorWidget(QWidget):
    """Widget for editing Place objects (position, rotation, scale)."""

    place_changed = Signal(object)
    keyframe_requested = Signal(str, object)

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self._place = None
        self._updating = False
        self._handlers = []
        self.setup_ui()
        self._setup_handlers()

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._check_position_update)
        self._update_timer.start(100)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout(pos_group)

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-9999, 9999)
        self.x_spin.setDecimals(2)
        self.x_spin.valueChanged.connect(self._on_position_changed)

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-9999, 9999)
        self.y_spin.setDecimals(2)
        self.y_spin.valueChanged.connect(self._on_position_changed)

        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-9999, 9999)
        self.z_spin.setDecimals(2)
        self.z_spin.valueChanged.connect(self._on_position_changed)

        pos_layout.addWidget(QLabel("X:"), 0, 0)
        pos_layout.addWidget(self.x_spin, 0, 1)
        pos_layout.addWidget(QLabel("Y:"), 1, 0)
        pos_layout.addWidget(self.y_spin, 1, 1)
        pos_layout.addWidget(QLabel("Z:"), 2, 0)
        pos_layout.addWidget(self.z_spin, 2, 1)

        rot_group = QGroupBox("Rotation")
        rot_layout = QGridLayout(rot_group)

        self.rx_spin = QDoubleSpinBox()
        self.rx_spin.setRange(-180, 180)
        self.rx_spin.setDecimals(1)
        self.rx_spin.setSuffix("°")
        self.rx_spin.valueChanged.connect(self._on_rotation_changed)

        self.ry_spin = QDoubleSpinBox()
        self.ry_spin.setRange(-180, 180)
        self.ry_spin.setDecimals(1)
        self.ry_spin.setSuffix("°")
        self.ry_spin.valueChanged.connect(self._on_rotation_changed)

        self.rz_spin = QDoubleSpinBox()
        self.rz_spin.setRange(-180, 180)
        self.rz_spin.setDecimals(1)
        self.rz_spin.setSuffix("°")
        self.rz_spin.valueChanged.connect(self._on_rotation_changed)

        rot_layout.addWidget(QLabel("X:"), 0, 0)
        rot_layout.addWidget(self.rx_spin, 0, 1)
        rot_layout.addWidget(QLabel("Y:"), 1, 0)
        rot_layout.addWidget(self.ry_spin, 1, 1)
        rot_layout.addWidget(QLabel("Z:"), 2, 0)
        rot_layout.addWidget(self.rz_spin, 2, 1)

        layout.addWidget(pos_group)
        layout.addWidget(rot_group)

        button_group = QGroupBox("Keyframes")
        button_layout = QVBoxLayout(button_group)

        self.keyframe_btn = QPushButton("Create Keyframe")
        self.keyframe_btn.clicked.connect(self._create_keyframe)
        button_layout.addWidget(self.keyframe_btn)

        layout.addWidget(button_group)
        layout.addStretch()
        self.setEnabled(False)

    def _setup_handlers(self):
        handler = self.session.triggers.add_handler("graphics update", self._on_graphics_update)
        self._handlers.append(handler)

        for trigger_name in ["model position changed", "models changed", "frame"]:
            try:
                handler = self.session.triggers.add_handler(trigger_name, self._on_model_position_changed)
                self._handlers.append(handler)
            except Exception:
                pass

    def _on_graphics_update(self, trigger_name, view):
        if not self._updating and self._place is not None:
            if hasattr(self, "_is_camera") and self._is_camera and hasattr(self, "_current_camera"):
                current_camera_pos = self._current_camera.position
                if not self._places_equal(current_camera_pos, self._place):
                    self.set_place(current_camera_pos)

    def _on_model_position_changed(self, trigger_name, model):
        if not self._updating and self._place is not None:
            if hasattr(self, "_current_model") and self._current_model == model:
                if not self._places_equal(model.position, self._place):
                    self.set_place(model.position)

    def _places_equal(self, place1, place2, tolerance=1e-6):
        if place1 is None or place2 is None:
            return place1 is place2
        import numpy as np

        return np.allclose(place1.matrix, place2.matrix, atol=tolerance)

    def _check_position_update(self):
        if self._updating or self._place is None:
            return

        current_position = None
        if hasattr(self, "_current_model") and self._current_model is not None:
            current_position = self._current_model.position
        elif hasattr(self, "_current_camera") and self._current_camera is not None:
            current_position = self._current_camera.position

        if current_position is not None and not self._places_equal(current_position, self._place):
            self.set_place(current_position)

    def cleanup(self):
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()

        if hasattr(self, "_update_timer"):
            self._update_timer.stop()

    def set_place(self, place):
        self._place = place
        self._updating = True

        if place is not None:
            position = place.translation()
            self.x_spin.setValue(position[0])
            self.y_spin.setValue(position[1])
            self.z_spin.setValue(position[2])

            axis, angle = place.rotation_axis_and_angle()
            if abs(axis[0]) > 0.7:
                self.rx_spin.setValue(angle if axis[0] > 0 else -angle)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(0)
            elif abs(axis[1]) > 0.7:
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(angle if axis[1] > 0 else -angle)
                self.rz_spin.setValue(0)
            elif abs(axis[2]) > 0.7:
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(angle if axis[2] > 0 else -angle)
            else:
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(0)

            self.setEnabled(True)
        else:
            self.setEnabled(False)

        self._updating = False

    def set_object(self, obj):
        if hasattr(obj, "position") and not hasattr(obj, "view"):
            self._current_model = obj
            self._current_camera = None
            self._is_camera = False
            self.set_place(obj.position)
        elif hasattr(obj, "position"):
            self._current_model = None
            self._current_camera = obj
            self._is_camera = True
            self.set_place(obj.position)
        else:
            self._current_model = None
            self._current_camera = None
            self._is_camera = False
            self.set_place(None)

    def _on_position_changed(self):
        if self._updating or self._place is None:
            return

        from chimerax.geometry import Place

        new_position = [self.x_spin.value(), self.y_spin.value(), self.z_spin.value()]
        new_place = Place(axes=self._place.axes(), origin=new_position)

        self._place = new_place
        if hasattr(self, "_current_model") and self._current_model is not None:
            self._current_model.position = new_place
        elif hasattr(self, "_current_camera") and self._current_camera is not None:
            self._current_camera.position = new_place

        self.place_changed.emit(new_place)

    def _on_rotation_changed(self):
        if self._updating or self._place is None:
            return

        from chimerax.geometry import Place, rotation

        current_pos = self._place.translation()
        rx = rotation([1, 0, 0], self.rx_spin.value())
        ry = rotation([0, 1, 0], self.ry_spin.value())
        rz = rotation([0, 0, 1], self.rz_spin.value())
        combined_rotation = rz * ry * rx
        new_place = Place(axes=combined_rotation.axes(), origin=current_pos)

        self._place = new_place
        if hasattr(self, "_current_model") and self._current_model is not None:
            self._current_model.position = new_place
        elif hasattr(self, "_current_camera") and self._current_camera is not None:
            self._current_camera.position = new_place

        self.place_changed.emit(new_place)

    def _create_keyframe(self):
        if self._place is not None:
            self.keyframe_requested.emit("transform", self._place)


class TrackDetailView(QWidget):
    """Detail view showing track info and property editors."""

    keyframe_requested = Signal(str, object)

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.label = QLabel("no track selected", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.place_editor = PlaceEditorWidget(self.session, self)
        self.place_editor.place_changed.connect(self._on_place_changed)
        self.place_editor.keyframe_requested.connect(self.keyframe_requested)
        layout.addWidget(self.place_editor)
        layout.addStretch()

    @Slot(str)
    def set_track(self, name: str):
        self.label.setText(name)

    def set_model(self, model):
        self.current_model = model
        self.place_editor.set_object(model)

    @Slot(object)
    def _on_place_changed(self, new_place):
        if self.current_model and hasattr(self.current_model, "position"):
            self.current_model.position = new_place

    def cleanup(self):
        if hasattr(self.place_editor, "cleanup"):
            self.place_editor.cleanup()


class KeyframeTimelineWidget(QWidget):
    """Detailed keyframe-mode widget extracted from the dual-mode editor."""

    keyframeMoved = Signal(int, int, int)
    clipResized = Signal(int, int, int)
    preferences_requested = Signal()

    def __init__(self, session=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.track_models = {}
        self.track_subtracks = {}
        self.track_parents = {}
        self.collapsed_tracks = set()
        self.is_playing = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)

        from .settings import get_settings

        self.fps = get_settings(session).playback_fps
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.model_selection_panel = ModelSelectionPanel(self.session)
        self.model_selection_panel.track_requested.connect(self.add_model_track)
        self.model_selection_panel.setFixedWidth(200)

        self.track_header = TrackHeaderView()
        self.timeline_scene = TimelineScene()
        self.timeline_view = TimelineView(self.timeline_scene)
        self.track_detail_view = TrackDetailView(self.session)
        self.track_detail_view.setFixedWidth(160)

        self.transport_controls = self._create_transport_controls()
        main_layout.addWidget(self.transport_controls)

        timeline_layout = QHBoxLayout()
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.addWidget(self.model_selection_panel)
        timeline_layout.addWidget(self.track_header)
        timeline_layout.addWidget(self.timeline_view)
        timeline_layout.addWidget(self.track_detail_view)
        main_layout.addLayout(timeline_layout)

        self.track_header.verticalScrollBar().valueChanged.connect(
            self.timeline_view.verticalScrollBar().setValue
        )
        self.timeline_view.verticalScrollBar().valueChanged.connect(
            self.track_header.verticalScrollBar().setValue
        )
        self.track_header.track_selected.connect(self._on_label_selected)
        self.timeline_view.row_clicked.connect(self.track_header.setCurrentRow)
        self.track_header.track_selected.connect(self._on_track_selected)
        self.track_header.track_hovered.connect(self._on_track_hovered)
        self.timeline_view.track_hovered.connect(self._on_track_hovered)
        self.track_header.track_collapsed.connect(self._on_track_collapsed)
        self.track_detail_view.keyframe_requested.connect(self._on_keyframe_requested)
        self.timeline_view.frame_changed.connect(self._on_frame_changed)
        self.timeline_view.frame_changed.connect(lambda f: self.frame_label.setText(f"Frame: {f}"))
        self.timeline_view.keyframes_deleted.connect(self._on_keyframes_deleted)
        self.timeline_view.scene_dropped.connect(self._on_scene_dropped)
        self.track_header.track_deleted.connect(self._on_track_deleted)
        self.timeline_view.track_deleted.connect(self._on_track_deleted)

        self.add_camera_track()

    def cleanup(self):
        self.playback_timer.stop()
        self.track_detail_view.cleanup()

    def _create_transport_controls(self):
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_frame.setFixedHeight(40)

        layout = QHBoxLayout(controls_frame)
        layout.setContentsMargins(5, 5, 5, 5)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self.play_pause_btn.setFixedSize(60, 30)
        layout.addWidget(self.play_pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setFixedSize(60, 30)
        layout.addWidget(self.stop_btn)

        self.frame_label = QLabel("Frame: 0")
        layout.addWidget(self.frame_label)

        layout.addWidget(QLabel("FPS:"))
        from Qt.QtWidgets import QComboBox

        self.fps_combo = QComboBox()
        for fps_val in [24, 48, 60, 120]:
            self.fps_combo.addItem(str(fps_val), fps_val)
        idx = self.fps_combo.findData(self.fps)
        if idx >= 0:
            self.fps_combo.setCurrentIndex(idx)
        self.fps_combo.currentIndexChanged.connect(
            lambda i: self._on_fps_changed(self.fps_combo.itemData(i))
        )
        self.fps_combo.setFixedWidth(70)
        layout.addWidget(self.fps_combo)

        layout.addWidget(QLabel("Zoom:"))

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_out_btn.setFixedSize(30, 30)
        layout.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_in_btn.setFixedSize(30, 30)
        layout.addWidget(zoom_in_btn)

        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_fit_btn.setFixedSize(40, 30)
        layout.addWidget(zoom_fit_btn)

        layout.addStretch()

        from Qt.QtWidgets import QToolButton
        from chimerax.ui.icons import get_qt_icon

        self.preferences_btn = QToolButton()
        self.preferences_btn.setIcon(get_qt_icon("gear"))
        self.preferences_btn.setToolTip("Preferences")
        self.preferences_btn.setFixedSize(30, 30)
        self.preferences_btn.clicked.connect(self.preferences_requested.emit)
        layout.addWidget(self.preferences_btn)

        return controls_frame

    @Slot(object)
    def add_model_track(self, model):
        track_name = f"{model.name} (#{model.id_string})"
        track_index = len(self.timeline_scene.track_rows)
        self.track_models[track_index] = model
        self.add_track(track_name, is_parent=True)

    def add_camera_track(self):
        track_name = "Camera"
        track_index = len(self.timeline_scene.track_rows)
        self.track_models[track_index] = self.session.view.camera
        self.add_track(track_name, is_parent=True)

    @Slot(int)
    def _on_label_selected(self, index: int):
        y = RULER_HEIGHT + index * TRACK_HEIGHT
        self.timeline_view.ensureVisible(0, y, 1, TRACK_HEIGHT)

    @Slot(int)
    def _on_track_selected(self, index: int):
        for row in self.timeline_scene.track_rows:
            row.set_highlighted(False)

        for i in range(self.track_header.count()):
            item = self.track_header.item(i)
            if item:
                item.setBackground(QBrush())

        if 0 <= index < len(self.timeline_scene.track_rows):
            self.timeline_scene.track_rows[index].set_highlighted(True)

        if index in self.track_models:
            model = self.track_models[index]
            if hasattr(model, "id_string"):
                display_text = f"Model: {model.name}\nID: #{model.id_string}"
            else:
                display_text = "Camera"
            self.track_detail_view.set_track(display_text)
            self.track_detail_view.set_model(model)
        else:
            self.track_detail_view.set_track("No model selected")
            self.track_detail_view.set_model(None)

    @Slot(int, bool)
    def _on_track_hovered(self, index: int, is_hovered: bool):
        if 0 <= index < len(self.timeline_scene.track_rows):
            self.timeline_scene.track_rows[index].set_hovered(is_hovered)

        if 0 <= index < self.track_header.count():
            item = self.track_header.item(index)
            if item:
                if is_hovered:
                    item.setBackground(QBrush(QColor("#2A3F5F")))
                else:
                    if self.track_header.currentRow() == index:
                        item.setBackground(QBrush(QColor("#3D5A80")))
                    else:
                        item.setBackground(QBrush())

    @Slot(str, object)
    def _on_keyframe_requested(self, property_name: str, value: object):
        current_track = self.track_header.currentRow()
        if current_track < 0:
            return

        root_track = self._get_root_parent_track(current_track)
        subtrack_index = self._get_or_create_subtrack(root_track, property_name)
        current_frame = self.timeline_scene.get_current_frame()
        self._remove_keyframe_at_frame(subtrack_index, current_frame)
        self.insert_keyframe(subtrack_index, current_frame, value)

    def _get_root_parent_track(self, track_index: int):
        if track_index in self.track_parents:
            return self.track_parents[track_index]
        return track_index

    def _get_or_create_subtrack(self, parent_track_index: int, property_name: str):
        if parent_track_index not in self.track_subtracks:
            self.track_subtracks[parent_track_index] = {}

        if property_name in self.track_subtracks[parent_track_index]:
            return self.track_subtracks[parent_track_index][property_name]

        parent_model = self.track_models.get(parent_track_index)
        subtrack_name = f"  └─ {property_name}"
        insertion_position = self._find_subtrack_insertion_position(parent_track_index)
        self._shift_track_indices_after(insertion_position)

        subtrack_index = insertion_position
        self.track_subtracks[parent_track_index][property_name] = subtrack_index
        self.track_models[subtrack_index] = parent_model
        self.track_parents[subtrack_index] = parent_track_index
        self.insert_track(insertion_position, subtrack_name)
        return subtrack_index

    def _find_subtrack_insertion_position(self, parent_track_index: int):
        insertion_pos = parent_track_index + 1
        if parent_track_index in self.track_subtracks:
            for existing_subtrack_index in self.track_subtracks[parent_track_index].values():
                if existing_subtrack_index >= insertion_pos:
                    insertion_pos = existing_subtrack_index + 1
        return insertion_pos

    def _shift_track_indices_after(self, insertion_position: int):
        old_models = dict(self.track_models)
        self.track_models.clear()
        for track_index, model in old_models.items():
            if track_index >= insertion_position:
                self.track_models[track_index + 1] = model
            else:
                self.track_models[track_index] = model

        old_subtracks = dict(self.track_subtracks)
        self.track_subtracks.clear()
        for parent_index, subtracks in old_subtracks.items():
            new_parent_index = parent_index + 1 if parent_index >= insertion_position else parent_index
            self.track_subtracks[new_parent_index] = {}
            for property_name, subtrack_index in subtracks.items():
                new_subtrack_index = subtrack_index + 1 if subtrack_index >= insertion_position else subtrack_index
                self.track_subtracks[new_parent_index][property_name] = new_subtrack_index

        old_parents = dict(self.track_parents)
        self.track_parents.clear()
        for subtrack_index, parent_index in old_parents.items():
            new_subtrack_index = subtrack_index + 1 if subtrack_index >= insertion_position else subtrack_index
            new_parent_index = parent_index + 1 if parent_index >= insertion_position else parent_index
            self.track_parents[new_subtrack_index] = new_parent_index

    @Slot(int)
    def _on_track_collapsed(self, track_index: int):
        if track_index in self.collapsed_tracks:
            self.collapsed_tracks.remove(track_index)
            self._show_subtracks(track_index)
            self.track_header.set_track_expanded(track_index, True)
        else:
            self.collapsed_tracks.add(track_index)
            self._hide_subtracks(track_index)
            self.track_header.set_track_expanded(track_index, False)

    def _hide_subtracks(self, parent_track_index: int):
        if parent_track_index in self.track_subtracks:
            for subtrack_index in self.track_subtracks[parent_track_index].values():
                self.track_header.set_track_visible(subtrack_index, False)
                self.timeline_scene.set_track_visible(subtrack_index, False)
                self._hide_subtrack_keyframes(subtrack_index)

    def _show_subtracks(self, parent_track_index: int):
        if parent_track_index in self.track_subtracks:
            for subtrack_index in self.track_subtracks[parent_track_index].values():
                self.track_header.set_track_visible(subtrack_index, True)
                self.timeline_scene.set_track_visible(subtrack_index, True)
                self._show_subtrack_keyframes(subtrack_index)

    def _hide_subtrack_keyframes(self, track_index: int):
        if track_index >= len(self.timeline_scene.track_rows):
            return
        for item in self.timeline_scene.items():
            if (
                isinstance(item, KeyframeItem)
                and hasattr(item, "_track_row")
                and item._track_row.index == track_index
            ):
                item.setVisible(False)

    def _show_subtrack_keyframes(self, track_index: int):
        if track_index >= len(self.timeline_scene.track_rows):
            return
        for item in self.timeline_scene.items():
            if (
                isinstance(item, KeyframeItem)
                and hasattr(item, "_track_row")
                and item._track_row.index == track_index
            ):
                item.setVisible(True)

    @Slot(int)
    def _on_frame_changed(self, frame: int):
        self.frame_label.setText(f"Frame: {frame}")
        self._evaluate_animation_at_frame(frame)

    @Slot()
    def _toggle_playback(self):
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self.is_playing = True
        self.play_pause_btn.setText("Pause")
        interval = int(1000 / self.fps)
        self.playback_timer.start(interval)

    def _pause_playback(self):
        self.is_playing = False
        self.play_pause_btn.setText("Play")
        self.playback_timer.stop()

    @Slot()
    def _stop_playback(self):
        self._pause_playback()
        self.timeline_scene.set_current_frame(0)
        self.timeline_view.frame_changed.emit(0)

    def _advance_frame(self):
        current_frame = self.timeline_scene.get_current_frame()
        next_frame = current_frame + 1
        if next_frame > self.timeline_scene.num_frames:
            next_frame = 0

        self.timeline_scene.set_current_frame(next_frame)
        self.timeline_view.frame_changed.emit(next_frame)

    @Slot(int)
    def _on_fps_changed(self, fps: int):
        self.fps = fps
        if self.is_playing:
            interval = int(1000 / self.fps)
            self.playback_timer.start(interval)

        from .settings import get_settings

        settings = get_settings(self.session)
        settings.playback_fps = fps

    def _evaluate_animation_at_frame(self, frame: int):
        for track_index, model in self.track_models.items():
            if model is None:
                continue

            if track_index in self.track_subtracks:
                for property_name, subtrack_index in self.track_subtracks[track_index].items():
                    if property_name == "transform":
                        interpolated_value = self._interpolate_keyframes(subtrack_index, frame)

                        if interpolated_value is not None and hasattr(model, "positions"):
                            try:
                                from chimerax.geometry import Place, Places

                                if isinstance(interpolated_value, Places):
                                    model.positions = interpolated_value
                                elif isinstance(interpolated_value, Place):
                                    model.positions = Places([interpolated_value])
                                elif hasattr(interpolated_value, "__iter__") and all(
                                    isinstance(p, Place) for p in interpolated_value
                                ):
                                    model.positions = Places(interpolated_value)
                            except Exception:
                                pass

    def _interpolate_keyframes(self, track_index: int, frame: int):
        if track_index >= len(self.timeline_scene.track_rows):
            return None

        keyframes = []
        for item in self.timeline_scene.items():
            if (
                isinstance(item, KeyframeItem)
                and hasattr(item, "_track_row")
                and item._track_row.index == track_index
            ):
                keyframes.append((item.data.frame, item.data.value))

        if not keyframes:
            return None

        keyframes.sort(key=lambda x: x[0])
        prev_kf = None
        next_kf = None

        for kf_frame, kf_value in keyframes:
            if kf_frame <= frame:
                prev_kf = (kf_frame, kf_value)
            if kf_frame >= frame and next_kf is None:
                next_kf = (kf_frame, kf_value)
                break

        if prev_kf and prev_kf[0] == frame:
            return prev_kf[1]
        if next_kf and next_kf[0] == frame:
            return next_kf[1]

        if not prev_kf and next_kf:
            return next_kf[1]
        if prev_kf and not next_kf:
            return prev_kf[1]
        if not prev_kf and not next_kf:
            return None

        if prev_kf and next_kf and prev_kf[0] != next_kf[0]:
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            t = max(0, min(1, t))

            from chimerax.geometry import Place, Places

            if (
                (isinstance(prev_kf[1], (Place, Places)) and isinstance(next_kf[1], (Place, Places)))
                or (
                    hasattr(prev_kf[1], "__iter__")
                    and hasattr(next_kf[1], "__iter__")
                    and all(isinstance(p, Place) for p in prev_kf[1])
                    and all(isinstance(p, Place) for p in next_kf[1])
                )
            ):
                try:
                    prev_positions = (
                        prev_kf[1]
                        if isinstance(prev_kf[1], Places)
                        else Places([prev_kf[1]])
                        if isinstance(prev_kf[1], Place)
                        else Places(list(prev_kf[1]))
                    )
                    next_positions = (
                        next_kf[1]
                        if isinstance(next_kf[1], Places)
                        else Places([next_kf[1]])
                        if isinstance(next_kf[1], Place)
                        else Places(list(next_kf[1]))
                    )

                    if len(prev_positions) != len(next_positions):
                        return prev_kf[1]

                    if track_index in self.track_models:
                        model = self.track_models[track_index]
                        if hasattr(model, "view") or model == self.session.view.camera:
                            center = self._get_scene_center()
                        else:
                            center = self._get_model_center(model)
                    else:
                        center = [0, 0, 0]

                    import numpy as np

                    if not isinstance(center, np.ndarray):
                        center = np.array(center, dtype=np.float32)

                    interpolated_positions = []
                    for i in range(len(prev_positions)):
                        result = prev_positions[i].interpolate(next_positions[i], center, t)
                        if isinstance(result, Place):
                            interpolated_positions.append(result)
                        else:
                            interpolated_positions.append(prev_positions[i])

                    if len(interpolated_positions) == 1 and isinstance(prev_kf[1], Place):
                        return interpolated_positions[0]
                    return Places(interpolated_positions)
                except Exception:
                    return prev_kf[1]

        return prev_kf[1] if prev_kf else next_kf[1]

    def _get_scene_center(self):
        if not self.session:
            return [0, 0, 0]

        try:
            if hasattr(self.session.view, "center_of_rotation"):
                cor = self.session.view.center_of_rotation
                if cor is not None:
                    return [cor[0], cor[1], cor[2]]
        except Exception:
            pass

        try:
            from chimerax.atomic import AtomicStructure

            bounds_list = []
            for model in self.session.models:
                if isinstance(model, AtomicStructure) and model.visible:
                    try:
                        bounds = model.bounds()
                        if bounds is not None:
                            bounds_list.append(bounds)
                    except Exception:
                        continue
                elif hasattr(model, "bounds") and model.visible:
                    try:
                        bounds = model.bounds()
                        if bounds is not None:
                            bounds_list.append(bounds)
                    except Exception:
                        continue

            if bounds_list:
                from chimerax.geometry import union_bounds

                overall_bounds = union_bounds(bounds_list)
                if overall_bounds is not None:
                    center = overall_bounds.center()
                    return [center[0], center[1], center[2]]
        except Exception:
            pass

        return [0, 0, 0]

    def _get_model_center(self, model):
        if not model:
            return [0, 0, 0]

        try:
            if hasattr(model, "bounds"):
                bounds = model.bounds()
                if bounds is not None:
                    center = bounds.center()
                    return [center[0], center[1], center[2]]
        except Exception:
            pass

        try:
            if hasattr(model, "position"):
                pos = model.position.translation()
                return [pos[0], pos[1], pos[2]]
        except Exception:
            pass

        return [0, 0, 0]

    def _interpolate_camera_direct(self, place1, place2, t):
        from chimerax.geometry import Place
        import numpy as np

        pos1 = place1.translation()
        pos2 = place2.translation()
        interp_pos = pos1 + t * (pos2 - pos1)

        rot1 = place1.axes()
        rot2 = place2.axes()
        q1 = self._matrix_to_quaternion(rot1)
        q2 = self._matrix_to_quaternion(rot2)
        interp_q = self._slerp_quaternion(q1, q2, t)
        interp_rot = self._quaternion_to_matrix(interp_q)
        return Place(axes=interp_rot, origin=interp_pos)

    def _matrix_to_quaternion(self, matrix):
        import numpy as np

        trace = np.trace(matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                w = (matrix[2, 1] - matrix[1, 2]) / s
                x = 0.25 * s
                y = (matrix[0, 1] + matrix[1, 0]) / s
                z = (matrix[0, 2] + matrix[2, 0]) / s
            elif matrix[1, 1] > matrix[2, 2]:
                s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                w = (matrix[0, 2] - matrix[2, 0]) / s
                x = (matrix[0, 1] + matrix[1, 0]) / s
                y = 0.25 * s
                z = (matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                w = (matrix[1, 0] - matrix[0, 1]) / s
                x = (matrix[0, 2] + matrix[2, 0]) / s
                y = (matrix[1, 2] + matrix[2, 1]) / s
                z = 0.25 * s

        return np.array([w, x, y, z])

    def _slerp_quaternion(self, q1, q2, t):
        import numpy as np

        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q1 + s1 * q2

    def _quaternion_to_matrix(self, q):
        import numpy as np

        w, x, y, z = q
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0:
            return np.eye(3)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    @Slot(list)
    def _on_keyframes_deleted(self, deleted_keyframes):
        pass

    def _remove_keyframe_at_frame(self, track_index: int, frame: int):
        if track_index >= len(self.timeline_scene.track_rows):
            return

        items_to_remove = []
        for item in self.timeline_scene.items():
            if (
                isinstance(item, KeyframeItem)
                and hasattr(item, "_track_row")
                and item._track_row.index == track_index
                and item.data.frame == frame
            ):
                items_to_remove.append(item)

        for item in items_to_remove:
            self.timeline_scene.removeItem(item)

    def add_track(self, name: str, is_parent: bool = False):
        row = self.timeline_scene.add_track(name)
        self.track_header.add_track(name, is_parent)
        return row

    def insert_track(self, position: int, name: str, is_parent: bool = False):
        row = self.timeline_scene.insert_track(position, name)
        self.track_header.insert_track(position, name, is_parent)
        return row

    def insert_keyframe(self, track_index: int, frame: int, value: any):  # noqa: ANN401
        kf_data = KeyframeData(frame, value)
        row = self.timeline_scene.track_rows[track_index]
        return row.add_keyframe(kf_data)

    def insert_clip(self, track_index: int, start: int, end: int, payload: any):  # noqa: ANN401
        clip_data = ClipData(start, end, payload)
        row = self.timeline_scene.track_rows[track_index]
        return row.add_clip(clip_data)

    def _zoom_in(self):
        self.timeline_scene.zoom_in()
        self._update_zoom_label()

    def _zoom_out(self):
        self.timeline_scene.zoom_out()
        self._update_zoom_label()

    def _zoom_fit(self):
        view_width = self.timeline_view.viewport().width()
        scene_width = self.timeline_scene.num_frames * self.timeline_scene.base_frame_width
        if scene_width > 0:
            fit_zoom = view_width / scene_width * 0.9
            self.timeline_scene.set_zoom(fit_zoom)
            self._update_zoom_label()

    def _update_zoom_label(self):
        zoom_percent = int(self.timeline_scene.zoom_factor * 100)
        self.zoom_label.setText(f"{zoom_percent}%")

    @Slot(str, int)
    def _on_scene_dropped(self, scene_name: str, frame: int):
        if not self.session:
            return

        scene = self.session.scenes.get_scene(scene_name)
        if not scene:
            return

        scene_models = list(scene.named_view.positions.keys())

        for model in scene_models:
            if model not in self.session.models.list():
                continue

            existing_track = None
            for track_index, track_model in self.track_models.items():
                if track_model == model:
                    existing_track = track_index
                    break

            if existing_track is None:
                if hasattr(model, "name") and hasattr(model, "id_string"):
                    track_name = f"{model.name} (#{model.id_string})"
                else:
                    track_name = f"Model {id(model)}"

                track_index = len(self.timeline_scene.track_rows)
                self.track_models[track_index] = model
                self.add_track(track_name, is_parent=True)
                existing_track = track_index

            subtrack_index = self._get_or_create_subtrack(existing_track, "transform")
            self._remove_keyframe_at_frame(subtrack_index, frame)
            scene_positions = scene.named_view.positions[model]
            self.insert_keyframe(subtrack_index, frame, scene_positions)

        camera_track = self._get_or_create_camera_track()
        if camera_track is not None:
            camera_subtrack = self._get_or_create_subtrack(camera_track, "camera")
            self._remove_keyframe_at_frame(camera_subtrack, frame)

            if scene.main_view_data and "camera" in scene.main_view_data:
                camera_data = scene.main_view_data["camera"]
                self.insert_keyframe(camera_subtrack, frame, camera_data)
            else:
                camera = self.session.view.camera
                camera_data = {
                    "position": camera.position,
                    "field_of_view": getattr(camera, "field_of_view", None),
                    "field_width": getattr(camera, "field_width", None),
                }
                self.insert_keyframe(camera_subtrack, frame, camera_data)

    def _get_or_create_camera_track(self):
        for track_index, model in self.track_models.items():
            if hasattr(model, "view"):
                return track_index

        camera_track_index = None
        for i, row in enumerate(self.timeline_scene.track_rows):
            if row.track_name.lower() == "camera":
                camera_track_index = i
                break

        if camera_track_index is not None:
            self.track_models[camera_track_index] = self.session.view
            return camera_track_index

        return None

    @Slot(int)
    def _on_track_deleted(self, track_index: int):
        if track_index < 0 or track_index >= len(self.timeline_scene.track_rows):
            return

        if track_index in self.track_models:
            model = self.track_models[track_index]
            if hasattr(model, "view") or model == self.session.view.camera:
                return

        if track_index in self.track_subtracks:
            subtracks_to_delete = list(self.track_subtracks[track_index].values())
            subtracks_to_delete.sort(reverse=True)
            for subtrack_index in subtracks_to_delete:
                self._delete_track_at_index(subtrack_index)

        self._delete_track_at_index(track_index)

    def _delete_track_at_index(self, track_index: int):
        if track_index < 0 or track_index >= len(self.timeline_scene.track_rows):
            return

        items_to_remove = []
        for item in self.timeline_scene.items():
            if (
                isinstance(item, KeyframeItem)
                and hasattr(item, "_track_row")
                and item._track_row.index == track_index
            ):
                items_to_remove.append(item)

        for item in items_to_remove:
            self.timeline_scene.removeItem(item)

        track_row = self.timeline_scene.track_rows[track_index]
        self.timeline_scene.removeItem(track_row)
        self.timeline_scene.track_rows.pop(track_index)
        self.track_header.takeItem(track_index)
        self._shift_track_indices_after_deletion(track_index)
        self.timeline_scene._update_scene_rect()
        self.timeline_scene._draw_playhead()

    def _shift_track_indices_after_deletion(self, deleted_index: int):
        old_models = dict(self.track_models)
        self.track_models.clear()
        for track_index, model in old_models.items():
            if track_index < deleted_index:
                self.track_models[track_index] = model
            elif track_index > deleted_index:
                self.track_models[track_index - 1] = model

        old_subtracks = dict(self.track_subtracks)
        self.track_subtracks.clear()
        for parent_index, subtracks in old_subtracks.items():
            if parent_index == deleted_index:
                continue

            new_parent_index = parent_index - 1 if parent_index > deleted_index else parent_index
            self.track_subtracks[new_parent_index] = {}
            for property_name, subtrack_index in subtracks.items():
                if subtrack_index == deleted_index:
                    continue
                new_subtrack_index = subtrack_index - 1 if subtrack_index > deleted_index else subtrack_index
                self.track_subtracks[new_parent_index][property_name] = new_subtrack_index

        old_parents = dict(self.track_parents)
        self.track_parents.clear()
        for subtrack_index, parent_index in old_parents.items():
            if subtrack_index == deleted_index or parent_index == deleted_index:
                continue

            new_subtrack_index = subtrack_index - 1 if subtrack_index > deleted_index else subtrack_index
            new_parent_index = parent_index - 1 if parent_index > deleted_index else parent_index
            self.track_parents[new_subtrack_index] = new_parent_index

        old_collapsed = set(self.collapsed_tracks)
        self.collapsed_tracks.clear()
        for collapsed_index in old_collapsed:
            if collapsed_index == deleted_index:
                continue
            new_collapsed_index = collapsed_index - 1 if collapsed_index > deleted_index else collapsed_index
            self.collapsed_tracks.add(new_collapsed_index)

        for i, track_row in enumerate(self.timeline_scene.track_rows):
            track_row.index = i
            new_y = i * TRACK_HEIGHT + RULER_HEIGHT
            track_row.setPos(0, new_y)

            for item in self.timeline_scene.items():
                if (
                    isinstance(item, KeyframeItem)
                    and hasattr(item, "_track_row")
                    and item._track_row == track_row
                ):
                    kf_y = new_y + TRACK_HEIGHT / 2
                    item.setPos(item.pos().x(), kf_y)

        old_track_widgets = dict(self.track_header.track_widgets)
        self.track_header.track_widgets.clear()
        for old_index, widget in old_track_widgets.items():
            if old_index == deleted_index:
                continue
            new_index = old_index - 1 if old_index > deleted_index else old_index
            widget.track_index = new_index
            self.track_header.track_widgets[new_index] = widget


KeyframeEditorWidget = KeyframeTimelineWidget
