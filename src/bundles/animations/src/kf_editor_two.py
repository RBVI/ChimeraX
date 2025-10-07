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

"""
This module defines the KeyframeEditorWidget class and related graphics items to build
an interactive keyframe/timeline editor widget using PySide6.QtWidgets and
PySide6.QtCore frameworks.

The widget is composed of:
    - TrackHeaderView: A header area listing track names and allowing track-level
      operations.
    - TimelineView: A QGraphicsView that displays a QGraphicsScene containing
      timeline ruler, keyframes, and clips.
    - KeyframeItem: A QGraphicsRectItem representing an individual keyframe.
    - ClipItem: A QGraphicsRectItem representing a clip spanning multiple
      keyframes or time range.

Key Components
--------------
KeyframeEditorWidget
^^^^^^^^^^^^^^^^^^^^
A composite widget that aligns TrackHeaderView and TimelineView horizontally so
that track rows line up with the timeline content. It provides high‑level APIs
for adding/removing tracks, inserting keyframes, and querying selected items.

TrackHeaderView
^^^^^^^^^^^^^^^
A QListWidget subclass displaying track names with optional mute/solo buttons.
Selecting a track in the header highlights the corresponding row in the
timeline. Right‑click context menu supports renaming and deleting tracks.

TimelineView & TimelineScene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimelineView subclasses QGraphicsView, hosting a custom TimelineScene. It
creates a timeline ruler at the top, draws vertical grid lines for frames, and
manages TrackRow groups that hold KeyframeItems and ClipItems.

TrackRow (QGraphicsItemGroup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each track row is a QGraphicsItemGroup containing a background rect, horizontal
separator line, and child items (keyframes/clips). TrackRow keeps its Y
position synced with the corresponding row in TrackHeaderView.

KeyframeItem
^^^^^^^^^^^^
A small square/diamond shaped QGraphicsRectItem that can be dragged along the X
axis to change its frame. Double‑clicking opens a value editor dialog (handled
externally). Shift‑dragging duplicates keyframes.

ClipItem
^^^^^^^^
A rectangular QGraphicsRectItem spanning a time range. It can be resized via
handles or moved along the timeline. Dragging with Alt duplicates the clip.

Signals
^^^^^^^
KeyframeEditorWidget emits high‑level signals such as keyframeMoved(int, int,
int) and clipResized(int, int, int) with arguments (track_index, old_frame,
new_frame). Client code can connect to these to update the underlying data
model.

Usage Example
-------------
    >>> editor = KeyframeEditorWidget()
    >>> editor.add_track("Position")
    >>> editor.insert_keyframe(0, frame=12, value=(0, 0, 0))

The widget is designed to be embedded in larger applications like animation
tools or video editors. It purposely avoids any domain‑specific logic so that
clients can adapt it to their data models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Qt.QtCore import (QEasingCurve, QPointF, QRectF, Qt, QTimer, QSize,
                            Signal, Slot)
from Qt.QtGui import (QBrush, QColor, QFont, QPainter, QPen, QPixmap,
                           QPolygonF)
from Qt.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QGraphicsItem,
                               QGraphicsItemGroup, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMenu, QSizePolicy,
                               QStyleOptionGraphicsItem, QVBoxLayout, QWidget,
                               QPushButton, QFrame, QDoubleSpinBox, QSpinBox, QFormLayout,
                               QGroupBox, QGridLayout, QStackedWidget, QButtonGroup,
                               QDialog, QComboBox, QLineEdit)

__all__ = [
    "KeyframeEditorWidget",
]

# Import scene timeline components for dual-mode support
from .scene_timeline import SceneTimelineWidget
from .scene_animation import SceneAnimation


class CompactStackedWidget(QStackedWidget):
    """Custom QStackedWidget that only uses the size of the current widget"""

    def sizeHint(self):
        """Return the size hint of the current widget only, not the largest"""
        current_widget = self.currentWidget()
        if current_widget:
            return current_widget.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        """Return the minimum size hint of the current widget only"""
        current_widget = self.currentWidget()
        if current_widget:
            return current_widget.minimumSizeHint()
        return super().minimumSizeHint()

TRACK_HEIGHT = 24
RULER_HEIGHT = 18
FRAME_WIDTH = 12  # pixels per frame at 100% zoom

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
    value: any  # noqa: ANN401 – allow arbitrary data type

@dataclass
class ClipData:
    start: int
    end: int
    payload: any  # noqa: ANN401 – allow arbitrary data type

class KeyframeItem(QGraphicsRectItem):
    """Visual representation of a keyframe."""

    def __init__(self, data: KeyframeData, parent: Optional[QGraphicsItem] = None):
        super().__init__(-KEYFRAME_SIZE / 2, -KEYFRAME_SIZE / 2, KEYFRAME_SIZE, KEYFRAME_SIZE, parent)
        self.setBrush(KEYFRAME_BRUSH)
        self.setPen(KEYFRAME_PEN)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)  # Explicitly accept left mouse button
        self.setZValue(10)  # Put keyframes above background elements
        self.data = data
        # Cache original frame when dragging
        self._drag_start_frame: int | None = None

    def mousePressEvent(self, event):  # noqa: D401, N802
        self._drag_start_frame = self.data.frame
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401, N802
        super().mouseMoveEvent(event)

    def itemChange(self, change, value):  # noqa: D401, N802
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_x = value.x()
            # Keep Y position fixed to track center (don't allow vertical movement)
            if hasattr(self, '_track_row') and self._track_row:
                # Calculate correct Y position for this track
                track_y = self._track_row.index * TRACK_HEIGHT + RULER_HEIGHT + TRACK_HEIGHT / 2
                value.setY(track_y)

            # Snap to frame boundaries
            frame_width = getattr(self.scene(), 'frame_width', FRAME_WIDTH)
            frame = round(new_x / frame_width)
            # Clamp frame to non‑negative
            frame = max(frame, 0)
            value.setX(frame * frame_width)

            # Update model data
            self.data.frame = frame

            # Extend timeline if needed
            if hasattr(self.scene(), 'extend_timeline') and frame > self.scene().num_frames:
                self.scene().extend_timeline(frame + 24)  # Add some padding

            return value
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        """Custom paint to show selection state."""
        # Draw the keyframe
        super().paint(painter, option, widget)

        # Draw selection border if selected
        if self.isSelected():
            selection_pen = QPen(QColor("#FFFFFF"))
            selection_pen.setWidth(2)
            painter.setPen(selection_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(-1, -1, 1, 1))


class ClipItem(QGraphicsRectItem):
    """Visual representation of a clip spanning multiple frames."""

    def __init__(self, data: ClipData, parent: Optional[QGraphicsItem] = None):
        # Initial positioning - will be updated by scene
        x = data.start * FRAME_WIDTH
        width = (data.end - data.start) * FRAME_WIDTH
        super().__init__(x, 0, width, TRACK_HEIGHT, parent)
        self.setBrush(CLIP_BRUSH)
        #self.setPen(CLIP_PEN)
        self.data = data
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        # Handles would be added here for resizing

    def itemChange(self, change, value):  # noqa: D401, N802
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_x = value.x()
            # Snap to frame grid
            frame_width = getattr(self.scene(), 'frame_width', FRAME_WIDTH)
            frame = round(new_x / frame_width)
            frame = max(frame, 0)
            self.data.start = frame
            self.data.end = frame + self.data.end - self.data.start
            value.setX(frame * frame_width)
            return value
        return super().itemChange(change, value)


class TrackRow(QGraphicsItemGroup):
    """Group holding all graphics items for a single track row."""

    def __init__(self, index: int, name: str, scene_width: int):
        super().__init__()
        self.index = index
        self.name = name

        # Position the entire group at the track's location
        y = index * TRACK_HEIGHT + RULER_HEIGHT
        self.setPos(0, y)

        # Background rect - positioned relative to group (0,0)
        self.bg = QGraphicsRectItem(0, 0, scene_width, TRACK_HEIGHT, self)
        self.bg.setBrush(QColor("#202020"))
        self.bg.setPen(QPen(Qt.NoPen))
        self.bg.setZValue(-1)  # Put background behind keyframes
        # Make background non-interactive so clicks pass through to keyframes
        self.bg.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.bg.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.bg.setAcceptedMouseButtons(Qt.NoButton)

        # Bottom separator line - positioned relative to group
        self.sep = QGraphicsRectItem(0, TRACK_HEIGHT - 1, scene_width, 1, self)
        self.sep.setBrush(SEPARATOR_PEN.color())
        self.sep.setPen(QPen(Qt.NoPen))
        self.sep.setZValue(-1)  # Put separator behind keyframes
        # Make separator non-interactive too
        self.sep.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.sep.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.sep.setAcceptedMouseButtons(Qt.NoButton)

        self.addToGroup(self.bg)
        self.addToGroup(self.sep)
        self._highlighted = False
        self._hovered = False

        # Enable hover events
        self.setAcceptHoverEvents(True)

    def set_highlighted(self, highlighted: bool):
        """Set the highlight state of this track row."""
        self._highlighted = highlighted
        self._update_background()

    def set_hovered(self, hovered: bool):
        """Set the hover state of this track row."""
        self._hovered = hovered
        self._update_background()

    def _update_background(self):
        """Update background color based on highlight and hover states."""
        if self._highlighted:
            self.bg.setBrush(QColor("#3D5A80"))  # Blue highlight
        elif self._hovered:
            self.bg.setBrush(QColor("#2A3F5F"))  # Hover color
        else:
            self.bg.setBrush(QColor("#202020"))  # Normal background

    def hoverEnterEvent(self, event):
        """Handle mouse enter for hover effect."""
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'track_hovered'):
                view.track_hovered.emit(self.index, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse leave for hover effect."""
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'track_hovered'):
                view.track_hovered.emit(self.index, False)
        super().hoverLeaveEvent(event)

    def add_keyframe(self, kf_data: KeyframeData):
        kf = KeyframeItem(kf_data)  # Don't set parent to group
        frame_width = getattr(self.scene(), 'frame_width', FRAME_WIDTH) if self.scene() else FRAME_WIDTH
        # Position keyframe relative to the track's absolute position
        track_y = self.pos().y()  # Get current track position
        kf.setPos(kf_data.frame * frame_width, track_y + TRACK_HEIGHT / 2)
        # Don't add to group - add directly to scene
        if self.scene():
            self.scene().addItem(kf)
        # Store reference to track for position calculations
        kf._track_row = self
        return kf

    def add_clip(self, clip_data: ClipData):
        clip = ClipItem(clip_data, self)

        frame_width = getattr(self.scene(), 'frame_width', FRAME_WIDTH) if self.scene() else FRAME_WIDTH
        # Position clip relative to the track's current position
        track_y = self.pos().y()  # Get current track position
        clip.setPos(clip_data.start * frame_width, track_y)
        self.addToGroup(clip)
        return clip

class TimelineScene(QGraphicsScene):
    """Scene containing timeline ruler and track rows."""

    def __init__(self, num_frames: int = 240, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.num_frames = num_frames
        self.zoom_factor = 1.0  # Current zoom level
        self.base_frame_width = 12  # Base pixels per frame
        self.track_rows: List[TrackRow] = []
        self.current_frame = 0
        self.playhead_line = None
        self.ruler_items = []  # Store ruler items for clearing
        self.grid_items = []   # Store grid items for clearing
        self.setBackgroundBrush(QColor("#303030"))
        self._draw_ruler()
        self._draw_frame_grid()
        self._draw_playhead()

    @property
    def frame_width(self):
        """Get current frame width based on zoom."""
        return self.base_frame_width * self.zoom_factor

    def set_zoom(self, factor: float):
        """Set zoom factor and redraw timeline."""
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(factor, 10.0))  # Clamp zoom between 0.1x and 10x

        if old_zoom != self.zoom_factor:
            self._redraw_timeline()

    def zoom_in(self):
        """Zoom in by 20%."""
        self.set_zoom(self.zoom_factor * 1.2)

    def zoom_out(self):
        """Zoom out by 20%."""
        self.set_zoom(self.zoom_factor / 1.2)

    def extend_timeline(self, new_num_frames: int):
        """Extend timeline to accommodate more frames."""
        if new_num_frames > self.num_frames:
            self.num_frames = new_num_frames
            self._redraw_timeline()

    def _redraw_timeline(self):
        """Redraw entire timeline with current zoom and frame count."""
        # Clear existing ruler and grid items
        for item in self.ruler_items:
            self.removeItem(item)
        for item in self.grid_items:
            self.removeItem(item)
        self.ruler_items.clear()
        self.grid_items.clear()

        # Redraw everything
        self._draw_ruler()
        self._draw_frame_grid()
        self._update_scene_rect()
        self._update_track_positions()
        self._draw_playhead()

    def _draw_ruler(self):
        font = QFont()
        font.setPointSize(8)
        # Adaptive tick spacing based on zoom
        if self.zoom_factor >= 2.0:
            tick_interval = 1  # Show every frame when zoomed in
        elif self.zoom_factor >= 0.5:
            tick_interval = 5  # Show every 5 frames at medium zoom
        else:
            tick_interval = 10  # Show every 10 frames when zoomed out

        for frame in range(0, self.num_frames + 1, tick_interval):
            x = frame * self.frame_width
            tick = QGraphicsRectItem(x, 0, 1, 6)
            tick.setBrush(QColor("#CCCCCC"))
            tick.setPen(QPen(Qt.NoPen))
            self.addItem(tick)
            self.ruler_items.append(tick)

            # Only add labels for major ticks to avoid clutter
            if frame % (tick_interval * 2) == 0 or tick_interval == 1:
                label = self.addText(str(frame), font)
                label.setDefaultTextColor(QColor("#CCCCCC"))
                label.setPos(x + 2, 0)
                self.ruler_items.append(label)

    def _draw_frame_grid(self):
        # Adaptive grid spacing based on zoom
        if self.zoom_factor >= 2.0:
            grid_interval = 1  # Show every frame when heavily zoomed in
        elif self.zoom_factor >= 1.0:
            grid_interval = 5  # Show every 5 frames when zoomed in
        else:
            grid_interval = 10  # Show every 10 frames when zoomed out

        # Only draw frame markers in the ruler area, not extending to tracks
        for frame in range(0, self.num_frames + 1, grid_interval):
            x = frame * self.frame_width
            # Draw short tick marks in the ruler area only
            tick_height = 8 if frame % (grid_interval * 2) == 0 else 4  # Taller for major ticks
            line = self.addLine(x, RULER_HEIGHT - tick_height, x, RULER_HEIGHT, GRID_PEN)
            line.setZValue(-1)
            self.grid_items.append(line)

    def _update_scene_rect(self):
        # Count only visible tracks for height calculation
        visible_track_count = sum(1 for track in self.track_rows if track.isVisible())
        height = RULER_HEIGHT + visible_track_count * TRACK_HEIGHT
        width = self.num_frames * self.frame_width
        self.setSceneRect(0, 0, width, height)

    def _update_track_positions(self):
        """Update positions of all keyframes and clips when zoom changes."""
        for track_row in self.track_rows:
            # Update keyframes directly in scene (since they're not in groups)
            for item in self.items():
                if isinstance(item, KeyframeItem) and hasattr(item, '_track_row') and item._track_row == track_row:
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
        # Extend grid lines
        self._draw_frame_grid()
        self._update_scene_rect()
        # Update playhead to account for new height
        self._draw_playhead()
        return row

    def insert_track(self, position: int, name: str):
        """Insert a track at a specific position."""
        # Create the track row with temporary index
        row = TrackRow(position, name, self.num_frames * self.frame_width)

        # Insert into track_rows list at the specified position
        self.track_rows.insert(position, row)

        # Update indices for all tracks after the insertion point
        for i in range(position, len(self.track_rows)):
            self.track_rows[i].index = i
            # Update the track position
            new_y = i * TRACK_HEIGHT + RULER_HEIGHT
            self.track_rows[i].setPos(0, new_y)

        self.addItem(row)
        # Extend grid lines
        self._draw_frame_grid()
        self._update_scene_rect()
        # Update playhead to account for new height
        self._draw_playhead()
        return row

    def set_track_visible(self, track_index: int, visible: bool):
        """Show or hide a track row in the timeline."""
        if track_index < len(self.track_rows):
            track_row = self.track_rows[track_index]
            track_row.setVisible(visible)
            # Update track positions to close gaps left by hidden tracks
            self._update_track_positions_for_visibility()
            # Update grid and scene after visibility change
            self._update_scene_rect()
            self._draw_frame_grid()
            self._draw_playhead()
            # Force scene update
            self.update()
            # Also force view updates
            for view in self.views():
                view.update()
                view.viewport().update()

    def _update_track_positions_for_visibility(self):
        """Update track positions to account for hidden tracks."""
        visible_track_count = 0
        for track_index, track_row in enumerate(self.track_rows):
            if track_row.isVisible():
                # Position this track at the next available visible position
                new_y_pos = visible_track_count * TRACK_HEIGHT + RULER_HEIGHT

                # Force update the track row position
                track_row.setPos(0, new_y_pos)
                track_row.update()  # Force the item to update its visual representation

                # Also update the internal index for calculations
                track_row.visual_index = visible_track_count

                # Update keyframes for this track to match new position
                for item in self.items():
                    if (isinstance(item, KeyframeItem) and
                        hasattr(item, '_track_row') and
                        item._track_row == track_row and
                        item.isVisible()):
                        kf_y = new_y_pos + TRACK_HEIGHT / 2
                        item.setPos(item.pos().x(), kf_y)

                visible_track_count += 1
            else:
                # Hidden tracks get moved out of view
                track_row.setPos(0, -1000)  # Move off-screen

    def _draw_playhead(self):
        """Draw the playhead line."""
        if self.playhead_line is not None:
            self.removeItem(self.playhead_line)

        x = self.current_frame * self.frame_width
        # Calculate height based on visible tracks only
        visible_track_count = sum(1 for track in self.track_rows if track.isVisible())
        height = RULER_HEIGHT + visible_track_count * TRACK_HEIGHT

        # Create playhead line
        playhead_pen = QPen(QColor("#FF6B6B"))  # Red playhead
        playhead_pen.setWidth(2)
        self.playhead_line = self.addLine(x, 0, x, height, playhead_pen)
        self.playhead_line.setZValue(10)  # Always on top

    def set_current_frame(self, frame: int):
        """Set the current frame and update playhead position."""
        frame = max(0, min(frame, self.num_frames))
        if frame != self.current_frame:
            self.current_frame = frame
            self._draw_playhead()

    def get_current_frame(self) -> int:
        """Get the current frame."""
        return self.current_frame


class TimelineView(QGraphicsView):

    row_clicked = Signal(int)
    track_hovered = Signal(int, bool)  # track_index, is_hovered
    frame_changed = Signal(int)  # current_frame
    keyframes_deleted = Signal(list)  # list of deleted keyframes  
    scene_dropped = Signal(str, int)  # scene_name, frame
    track_deleted = Signal(int)  # track_index

    def __init__(self, scene: TimelineScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameStyle(QGraphicsView.NoFrame)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Enable focus to receive key events
        self.setFocusPolicy(Qt.StrongFocus)
        # Track playhead dragging state
        self._dragging_playhead = False
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        # Track last clicked track for deletion
        self._last_clicked_track = -1

    def wheelEvent(self, event):  # noqa: N802, D401
        # Zoom with Ctrl+Wheel or just wheel
        if event.modifiers() & Qt.ControlModifier or True:  # Allow zooming with just wheel
            angle_delta = event.angleDelta().y()
            if angle_delta > 0:
                self.scene().zoom_in()  # type: ignore[attr-defined]
            else:
                self.scene().zoom_out()  # type: ignore[attr-defined]
        else:
            super().wheelEvent(event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # ensure the view starts at frame 0
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint())

            # First check if we clicked on a keyframe
            item_at_pos = self.itemAt(event.position().toPoint())
            if isinstance(item_at_pos, KeyframeItem):
                # Let keyframe handle the event
                super().mousePressEvent(event)
                return

            # Check if clicking in ruler area to set playhead
            if 0 <= pos.y() <= RULER_HEIGHT:
                frame_width = self.scene().frame_width  # type: ignore[attr-defined]
                frame = round(pos.x() / frame_width)
                frame = max(0, min(frame, self.scene().num_frames))  # type: ignore[attr-defined]
                self.scene().set_current_frame(frame)  # type: ignore[attr-defined]
                self.frame_changed.emit(frame)
                # Start playhead dragging
                self._dragging_playhead = True
                return

            # Check if clicking on track rows for track selection
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
        """Handle mouse move events for playhead dragging."""
        if self._dragging_playhead:
            pos = self.mapToScene(event.position().toPoint())
            frame_width = self.scene().frame_width  # type: ignore[attr-defined]
            frame = round(pos.x() / frame_width)
            frame = max(0, min(frame, self.scene().num_frames))  # type: ignore[attr-defined]
            self.scene().set_current_frame(frame)  # type: ignore[attr-defined]
            self.frame_changed.emit(frame)

            # Extend timeline if dragging beyond current end
            if frame > self.scene().num_frames - 12:  # type: ignore[attr-defined]
                self.scene().extend_timeline(frame + 24)  # type: ignore[attr-defined]
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # First check if we have selected keyframes
            selected_items = self.scene().selectedItems()
            keyframes_to_delete = []

            for item in selected_items:
                if isinstance(item, KeyframeItem):
                    keyframes_to_delete.append(item)

            if keyframes_to_delete:
                # Delete selected keyframes
                self._delete_keyframes(keyframes_to_delete)
                self.keyframes_deleted.emit(keyframes_to_delete)
            else:
                # No keyframes selected, check if a track row is highlighted
                # We'll use the last clicked track for this
                if hasattr(self, '_last_clicked_track') and self._last_clicked_track >= 0:
                    self.track_deleted.emit(self._last_clicked_track)
        else:
            super().keyPressEvent(event)

    def _delete_keyframes(self, keyframes):
        """Delete the specified keyframes from the scene."""
        for keyframe in keyframes:
            # Since keyframes are now added directly to scene, just remove them
            self.scene().removeItem(keyframe)

    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasFormat("application/x-chimerax-scene"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move events."""
        if event.mimeData().hasFormat("application/x-chimerax-scene"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop events."""
        if event.mimeData().hasFormat("application/x-chimerax-scene"):
            # Get the drop position
            pos = self.mapToScene(event.position().toPoint())
            
            # Calculate frame from X position
            frame_width = self.scene().frame_width
            frame = max(0, round(pos.x() / frame_width))
            
            # Extract scene data
            scene_data_bytes = event.mimeData().data("application/x-chimerax-scene")
            try:
                import json
                scene_data = json.loads(scene_data_bytes.data().decode('utf-8'))
                scene_name = scene_data.get('name', 'Unknown Scene')
            except:
                # Fallback to treating it as plain text
                scene_name = scene_data_bytes.data().decode('utf-8')
            
            # Emit signal to handle the scene drop
            self.scene_dropped.emit(scene_name, frame)
            event.acceptProposedAction()
        else:
            event.ignore()


class TrackHeaderView(QListWidget):
    track_selected = Signal(int)
    track_hovered = Signal(int, bool)  # track_index, is_hovered
    track_collapsed = Signal(int)  # track_index
    track_deleted = Signal(int)  # track_index

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.setFixedWidth(120)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameStyle(QListWidget.NoFrame)
        self.setViewportMargins(0, RULER_HEIGHT, 0, 0)

        # Set custom selection colors
        self.setStyleSheet("""
            QListWidget::item:selected {
                background-color: #3D5A80;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2A3F5F;
            }
        """)

        # Enable mouse tracking for hover events
        self.setMouseTracking(True)
        self._last_hovered_index = -1
        self.track_widgets = {}  # track_index -> TrackItemWidget

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
        """Add a subtrack under a parent track."""
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, TRACK_HEIGHT))
        self.addItem(item)

        track_index = self.count() - 1
        track_widget = TrackItemWidget(track_index, name, is_parent=False)

        self.setItemWidget(item, track_widget)
        self.track_widgets[track_index] = track_widget
        return item

    def insert_track(self, position: int, name: str, is_parent: bool = False):
        """Insert a track at a specific position."""
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, TRACK_HEIGHT))
        self.insertItem(position, item)

        # Update track widget indices for all items after insertion point
        # First, shift existing track_widgets
        old_widgets = dict(self.track_widgets)
        self.track_widgets.clear()

        for old_index, widget in old_widgets.items():
            if old_index >= position:
                new_index = old_index + 1
                widget.track_index = new_index
                self.track_widgets[new_index] = widget
            else:
                self.track_widgets[old_index] = widget

        # Create widget for the new track
        track_widget = TrackItemWidget(position, name, is_parent)
        track_widget.expand_clicked.connect(self._on_track_collapsed)

        self.setItemWidget(item, track_widget)
        self.track_widgets[position] = track_widget
        return item

    def set_track_as_parent(self, track_index: int):
        """Convert an existing track to a parent track with expand/collapse button."""
        if track_index in self.track_widgets:
            widget = self.track_widgets[track_index]
            if not widget.is_parent:
                # Replace with new parent widget
                item = self.item(track_index)
                name = widget.name_label.text()
                new_widget = TrackItemWidget(track_index, name, is_parent=True)
                new_widget.expand_clicked.connect(self._on_track_collapsed)
                self.setItemWidget(item, new_widget)
                self.track_widgets[track_index] = new_widget

    def _on_track_collapsed(self, track_index: int):
        """Handle track expand/collapse."""
        self.track_collapsed.emit(track_index)

    def set_track_expanded(self, track_index: int, expanded: bool):
        """Set the expanded state of a track."""
        if track_index in self.track_widgets:
            self.track_widgets[track_index].set_expanded(expanded)

    def set_track_visible(self, track_index: int, visible: bool):
        """Show or hide a track."""
        if track_index < self.count():
            item = self.item(track_index)
            if item:
                if visible:
                    # Show the item
                    item.setHidden(False)
                    item.setSizeHint(QSize(0, TRACK_HEIGHT))
                else:
                    # Hide the item and collapse its space
                    item.setHidden(True)
                    item.setSizeHint(QSize(0, 0))

    @Slot()
    def _on_selection_changed(self):
        selected_indexes = self.selectedIndexes()
        if selected_indexes:
            self.track_selected.emit(selected_indexes[0].row())

    def mouseMoveEvent(self, event):
        """Handle mouse move for hover tracking."""
        item = self.itemAt(event.position().toPoint())
        if item:
            index = self.row(item)
            if index != self._last_hovered_index:
                # Emit hover leave for previous item
                if self._last_hovered_index >= 0:
                    self.track_hovered.emit(self._last_hovered_index, False)
                # Emit hover enter for new item
                self.track_hovered.emit(index, True)
                self._last_hovered_index = index
        else:
            # Mouse not over any item
            if self._last_hovered_index >= 0:
                self.track_hovered.emit(self._last_hovered_index, False)
                self._last_hovered_index = -1
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave for hover tracking."""
        if self._last_hovered_index >= 0:
            self.track_hovered.emit(self._last_hovered_index, False)
            self._last_hovered_index = -1
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events for track deletion."""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            current_row = self.currentRow()
            if current_row >= 0:
                self.track_deleted.emit(current_row)
        else:
            super().keyPressEvent(event)


class TrackItemWidget(QWidget):
    """Custom widget for track items with expand/collapse button."""

    expand_clicked = Signal(int)  # track_index

    def __init__(self, track_index: int, name: str, is_parent: bool = False):
        super().__init__()
        self.track_index = track_index
        self.is_parent = is_parent
        self.is_expanded = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)

        if is_parent:
            # Add expand/collapse button for parent tracks
            self.expand_btn = QPushButton("▼")
            self.expand_btn.setFixedSize(16, 16)
            # Clean button styling
            self.expand_btn.setStyleSheet("""
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
            """)
            self.expand_btn.clicked.connect(self._on_expand_clicked)
            layout.addWidget(self.expand_btn)
        else:
            # Add spacer for subtracks to align text
            layout.addSpacing(20)

        # Track name label
        self.name_label = QLabel(name)
        layout.addWidget(self.name_label)
        layout.addStretch()

    def _on_expand_clicked(self):
        self.is_expanded = not self.is_expanded
        self.expand_btn.setText("▼" if self.is_expanded else "▶")
        self.expand_clicked.emit(self.track_index)

    def set_expanded(self, expanded: bool):
        self.is_expanded = expanded
        if hasattr(self, 'expand_btn'):
            self.expand_btn.setText("▼" if expanded else "▶")


class ModelSelectionPanel(QWidget):
    """Panel for selecting models and adding tracks to the timeline."""

    track_requested = Signal(object) # Models

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("Model Animation")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # Model selection using ChimeraX ModelMenuButton
        model_frame = QFrame()
        model_layout = QVBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Select Model:"))

        # Use ChimeraX's ModelMenuButton for proper model selection
        from chimerax.ui.widgets import ModelMenuButton
        self.model_menu = ModelMenuButton(
            self.session,
            no_value_button_text="Choose model..."
        )
        self.model_menu.value_changed.connect(self.on_model_changed)
        model_layout.addWidget(self.model_menu)

        layout.addWidget(model_frame)

        # Add Track button
        self.add_track_button = QPushButton("Add Track")
        self.add_track_button.clicked.connect(self.add_track)
        self.add_track_button.setEnabled(False)
        layout.addWidget(self.add_track_button)

        # Stretch to push everything to top
        layout.addStretch()

    def on_model_changed(self):
        """Handle model selection change."""
        model = self.model_menu.value
        self.add_track_button.setEnabled(model is not None)

    def add_track(self):
        """Add an animation track for the selected model."""
        model = self.model_menu.value
        if model:
            self.track_requested.emit(model)


class PlaceEditorWidget(QWidget):
    """Widget for editing Place objects (position, rotation, scale)."""

    place_changed = Signal(object)  # Emits the new Place object
    keyframe_requested = Signal(str, object)  # property_name, value

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self._place = None
        self._updating = False
        self._handlers = []
        self.setup_ui()
        self._setup_handlers()

        # Set up a timer to periodically check for position changes
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._check_position_update)
        self._update_timer.start(100)  # Check every 100ms

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Position group
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

        # Rotation group
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

        # Keyframe buttons
        button_group = QGroupBox("Keyframes")
        button_layout = QVBoxLayout(button_group)

        self.keyframe_btn = QPushButton("Create Keyframe")
        self.keyframe_btn.clicked.connect(self._create_keyframe)
        button_layout.addWidget(self.keyframe_btn)

        layout.addWidget(button_group)
        layout.addStretch()

        # Initially disabled
        self.setEnabled(False)

    def _setup_handlers(self):
        """Set up triggers to listen for position changes."""
        # Listen for graphics updates (camera changes)
        handler = self.session.triggers.add_handler(
            'graphics update', self._on_graphics_update)
        self._handlers.append(handler)

        # Listen for model position changes - try multiple trigger names
        # Different versions of ChimeraX may use different trigger names
        for trigger_name in ['model position changed', 'models changed', 'frame']:
            try:
                handler = self.session.triggers.add_handler(
                    trigger_name, self._on_model_position_changed)
                self._handlers.append(handler)
            except:
                pass  # Trigger doesn't exist, skip it

    def _on_graphics_update(self, trigger_name, view):
        """Handle graphics updates (camera changes)."""
        if not self._updating and self._place is not None:
            # Check if this is a camera track
            if hasattr(self, '_is_camera') and self._is_camera and hasattr(self, '_current_camera'):
                current_camera_pos = self._current_camera.position
                if not self._places_equal(current_camera_pos, self._place):
                    self.set_place(current_camera_pos)

    def _on_model_position_changed(self, trigger_name, model):
        """Handle model position changes."""
        if not self._updating and self._place is not None:
            # Check if this is our current model
            if hasattr(self, '_current_model') and self._current_model == model:
                if not self._places_equal(model.position, self._place):
                    self.set_place(model.position)

    def _places_equal(self, place1, place2, tolerance=1e-6):
        """Check if two Place objects are approximately equal."""
        if place1 is None or place2 is None:
            return place1 is place2
        import numpy as np
        return np.allclose(place1.matrix, place2.matrix, atol=tolerance)

    def _check_position_update(self):
        """Periodically check if the current model/camera position has changed."""
        if self._updating or self._place is None:
            return

        current_position = None
        if hasattr(self, '_current_model') and self._current_model is not None:
            current_position = self._current_model.position
        elif hasattr(self, '_current_camera') and self._current_camera is not None:
            current_position = self._current_camera.position

        if current_position is not None and not self._places_equal(current_position, self._place):
            self.set_place(current_position)

    def cleanup(self):
        """Clean up event handlers."""
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()

        # Stop the update timer
        if hasattr(self, '_update_timer'):
            self._update_timer.stop()

    def set_place(self, place):
        """Set the Place object to edit."""
        self._place = place
        self._updating = True

        if place is not None:
            # Update position - use translation() method
            position = place.translation()
            self.x_spin.setValue(position[0])
            self.y_spin.setValue(position[1])
            self.z_spin.setValue(position[2])

            # Update rotation - use axis and angle for now (simpler and more reliable)
            axis, angle = place.rotation_axis_and_angle()

            # For now, just show the total rotation around the primary axis
            # This is a simplified approach - proper Euler angle conversion is complex
            if abs(axis[0]) > 0.7:  # Primarily X rotation
                self.rx_spin.setValue(angle if axis[0] > 0 else -angle)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(0)
            elif abs(axis[1]) > 0.7:  # Primarily Y rotation
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(angle if axis[1] > 0 else -angle)
                self.rz_spin.setValue(0)
            elif abs(axis[2]) > 0.7:  # Primarily Z rotation
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(angle if axis[2] > 0 else -angle)
            else:
                # Complex rotation - show as zero for now
                self.rx_spin.setValue(0)
                self.ry_spin.setValue(0)
                self.rz_spin.setValue(0)

            self.setEnabled(True)
        else:
            self.setEnabled(False)

        self._updating = False

    def set_object(self, obj):
        """Set the object (model or camera) being edited."""
        if hasattr(obj, 'position') and not hasattr(obj, 'view'):
            # Regular model
            self._current_model = obj
            self._current_camera = None
            self._is_camera = False
            self.set_place(obj.position)
        elif hasattr(obj, 'position'):
            # Camera object
            self._current_model = None
            self._current_camera = obj
            self._is_camera = True
            self.set_place(obj.position)
        else:
            # No valid object
            self._current_model = None
            self._current_camera = None
            self._is_camera = False
            self.set_place(None)

    def _on_position_changed(self):
        """Handle position changes."""
        if self._updating or self._place is None:
            return

        from chimerax.geometry import Place

        # Create new Place with same rotation but new position
        new_position = [self.x_spin.value(), self.y_spin.value(), self.z_spin.value()]
        new_place = Place(axes=self._place.axes(), origin=new_position)

        self._place = new_place
        # Apply the change to the object
        if hasattr(self, '_current_model') and self._current_model is not None:
            self._current_model.position = new_place
        elif hasattr(self, '_current_camera') and self._current_camera is not None:
            self._current_camera.position = new_place

        self.place_changed.emit(new_place)

    def _on_rotation_changed(self):
        """Handle rotation changes."""
        if self._updating or self._place is None:
            return

        from chimerax.geometry import rotation, Place

        # Get current position
        current_pos = self._place.translation()

        # Create rotation transformations
        rx = rotation([1, 0, 0], self.rx_spin.value())
        ry = rotation([0, 1, 0], self.ry_spin.value())
        rz = rotation([0, 0, 1], self.rz_spin.value())

        # Combine rotations (order matters - ZYX)
        combined_rotation = rz * ry * rx

        # Create new Place with new rotation and same position
        new_place = Place(axes=combined_rotation.axes(), origin=current_pos)

        self._place = new_place
        # Apply the change to the object
        if hasattr(self, '_current_model') and self._current_model is not None:
            self._current_model.position = new_place
        elif hasattr(self, '_current_camera') and self._current_camera is not None:
            self._current_camera.position = new_place

        self.place_changed.emit(new_place)

    def _create_keyframe(self):
        """Create a keyframe for the current transformation."""
        if self._place is not None:
            self.keyframe_requested.emit("transform", self._place)


class TrackDetailView(QWidget):
    """Detail view showing track info and property editors."""

    keyframe_requested = Signal(str, object)  # property_name, value

    def __init__(self, session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Track info label
        self.label = QLabel("no track selected", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Place editor
        self.place_editor = PlaceEditorWidget(self.session, self)
        self.place_editor.place_changed.connect(self._on_place_changed)
        self.place_editor.keyframe_requested.connect(self.keyframe_requested)
        layout.addWidget(self.place_editor)

        layout.addStretch()

    @Slot(str)
    def set_track(self, name: str):
        self.label.setText(name)

    def set_model(self, model):
        """Set the model to edit."""
        self.current_model = model
        self.place_editor.set_object(model)

    @Slot(object)
    def _on_place_changed(self, new_place):
        """Handle place changes from the editor."""
        if self.current_model and hasattr(self.current_model, 'position'):
            self.current_model.position = new_place

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.place_editor, 'cleanup'):
            self.place_editor.cleanup()


class KeyframeEditorWidget(QWidget):
    """High‑level composite widget combining track header and timeline."""

    keyframeMoved = Signal(int, int, int)  # track, old_frame, new_frame
    clipResized = Signal(int, int, int)  # track, old_end, new_end

    def __init__(self, session=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.track_models = {}  # track_index -> model
        self.track_subtracks = {}  # track_index -> {property_name: subtrack_index}
        self.track_parents = {}  # subtrack_index -> parent_track_index
        self.collapsed_tracks = set()  # Set of collapsed parent track indices
        self.is_playing = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)
        self.fps = 60  # Frames per second - increased for smoother playback

        # Get or create scene animation manager
        self.scene_animation = self._get_scene_animation_manager()

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Mode switching controls
        mode_controls_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup()

        self.keyframe_mode_btn = QPushButton("Keyframe Mode")
        self.keyframe_mode_btn.setCheckable(True)
        self.keyframe_mode_btn.setChecked(True)  # Default to keyframe mode
        self.keyframe_mode_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)

        self.scene_mode_btn = QPushButton("Scene Mode")
        self.scene_mode_btn.setCheckable(True)
        self.scene_mode_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
        """)

        self.mode_button_group.addButton(self.keyframe_mode_btn, 0)
        self.mode_button_group.addButton(self.scene_mode_btn, 1)
        self.mode_button_group.buttonClicked.connect(self.switch_mode)

        mode_controls_layout.addWidget(self.keyframe_mode_btn)
        mode_controls_layout.addWidget(self.scene_mode_btn)
        mode_controls_layout.addStretch()

        main_layout.addLayout(mode_controls_layout)

        # Stacked widget for different modes - custom class to handle sizing properly
        self.stacked_widget = CompactStackedWidget()
        # Set size policy to only use space of current widget, not the largest
        self.stacked_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Keyframe editor mode (original)
        self.keyframe_widget = self._create_keyframe_widget()
        self.stacked_widget.addWidget(self.keyframe_widget)

        # Scene timeline mode (new)
        self.scene_timeline_widget = SceneTimelineWidget(self.session)
        self.stacked_widget.addWidget(self.scene_timeline_widget)

        # Connect scene timeline signals
        self.scene_timeline_widget.scene_added.connect(self.on_scene_added)
        self.scene_timeline_widget.scene_removed.connect(self.on_scene_removed)
        self.scene_timeline_widget.scene_moved.connect(self.on_scene_moved)
        self.scene_timeline_widget.scene_selected.connect(self.on_scene_selected)
        self.scene_timeline_widget.time_changed.connect(self.on_scene_time_changed)
        self.scene_timeline_widget.play_requested.connect(self.on_scene_play_requested)
        self.scene_timeline_widget.pause_requested.connect(self.on_scene_pause_requested)
        self.scene_timeline_widget.record_requested.connect(self.on_scene_record_requested)
        self.scene_timeline_widget.duration_changed.connect(self.on_scene_duration_changed)
        self.scene_timeline_widget.reset_requested.connect(self.on_scene_reset_requested)

        # Connect scene animation signals to timeline controls
        self.scene_animation.signals.time_changed.connect(self.on_scene_animation_time_changed)
        self.scene_animation.signals.playback_started.connect(self.on_scene_animation_started)
        self.scene_animation.signals.playback_stopped.connect(self.on_scene_animation_stopped)
        self.scene_animation.signals.recording_started.connect(self.on_scene_recording_started)
        self.scene_animation.signals.recording_stopped.connect(self.on_scene_recording_stopped)

        main_layout.addWidget(self.stacked_widget)


    def _get_scene_animation_manager(self):
        """Get or create the scene animation manager"""
        if not hasattr(self.session, '_scene_animation_manager'):
            # Pass the keyframe system's FPS to ensure consistent timing
            self.session._scene_animation_manager = SceneAnimation(self.session, fps=self.fps)
        return self.session._scene_animation_manager


    def _create_keyframe_widget(self):
        """Create the original keyframe editor widget"""
        keyframe_widget = QWidget()
        keyframe_layout = QVBoxLayout(keyframe_widget)
        keyframe_layout.setContentsMargins(0, 0, 0, 0)

        # Create model selection panel
        self.model_selection_panel = ModelSelectionPanel(self.session)
        self.model_selection_panel.track_requested.connect(self.add_model_track)
        self.model_selection_panel.setFixedWidth(200)

        self.track_header = TrackHeaderView()
        self.timeline_scene = TimelineScene()
        self.timeline_view = TimelineView(self.timeline_scene)
        self.track_detail_view = TrackDetailView(self.session)
        self.track_detail_view.setFixedWidth(160)

        # Create transport controls
        self.transport_controls = self._create_transport_controls()

        # Transport controls at top
        keyframe_layout.addWidget(self.transport_controls)

        # Timeline area
        timeline_layout = QHBoxLayout()
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.addWidget(self.model_selection_panel)
        timeline_layout.addWidget(self.track_header)
        timeline_layout.addWidget(self.timeline_view)
        timeline_layout.addWidget(self.track_detail_view)

        # Add timeline to keyframe layout
        keyframe_layout.addLayout(timeline_layout)

        # Connect widget interactions
        # keep vertical scroll positions tied together
        self.track_header.verticalScrollBar().valueChanged.connect(
            self.timeline_view.verticalScrollBar().setValue
        )
        self.timeline_view.verticalScrollBar().valueChanged.connect(
            self.track_header.verticalScrollBar().setValue
        )
        self.track_header.track_selected.connect(self._on_label_selected)
        self.timeline_view.row_clicked.connect(self.track_header.setCurrentRow)
        self.track_header.track_selected.connect(self._on_track_selected)

        # Connect hover events
        self.track_header.track_hovered.connect(self._on_track_hovered)
        self.timeline_view.track_hovered.connect(self._on_track_hovered)

        # Connect track collapse/expand
        self.track_header.track_collapsed.connect(self._on_track_collapsed)

        # Connect keyframe creation
        self.track_detail_view.keyframe_requested.connect(self._on_keyframe_requested)

        # Connect frame changes
        self.timeline_view.frame_changed.connect(self._on_frame_changed)
        # Also connect scene frame changes to update UI
        self.timeline_view.frame_changed.connect(lambda f: self.frame_label.setText(f"Frame: {f}"))

        # Connect keyframe deletion
        self.timeline_view.keyframes_deleted.connect(self._on_keyframes_deleted)

        # Connect scene drops
        self.timeline_view.scene_dropped.connect(self._on_scene_dropped)

        # Connect track deletion
        self.track_header.track_deleted.connect(self._on_track_deleted)
        self.timeline_view.track_deleted.connect(self._on_track_deleted)

        # Add default camera track
        self.add_camera_track()

        return keyframe_widget

    def switch_mode(self, button):
        """Switch between keyframe and scene animation modes"""
        if button == self.keyframe_mode_btn:
            self.stacked_widget.setCurrentIndex(0)
            if self.session:
                self.session.logger.info("Switched to Keyframe Mode")
        elif button == self.scene_mode_btn:
            self.stacked_widget.setCurrentIndex(1)
            if self.session:
                self.session.logger.info("Switched to Scene Mode")

    def on_scene_added(self, scene_name, time=None):
        """Handle scene addition in scene mode"""
        if time is None:
            time = self.scene_timeline_widget.timeline_controls.current_time

        success = self.scene_animation.add_scene_at_time(scene_name, time)

        if success:
            self.scene_timeline_widget.add_scene_marker(time, scene_name)
            if self.session:
                self.session.logger.info(f"Added scene '{scene_name}' to animation at {time:.2f}s")

    def on_scene_removed(self, scene_name):
        """Handle scene removal in scene mode"""
        success = self.scene_animation.remove_scene(scene_name)

        if success:
            self.scene_timeline_widget.timeline_scene.remove_scene_marker(scene_name)
            if self.session:
                self.session.logger.info(f"Removed scene '{scene_name}' from animation")

    def on_scene_selected(self, scene_name):
        """Handle scene selection in scene mode"""
        if self.session and self.session.scenes.get_scene(scene_name):
            self.session.scenes.restore_scene(scene_name)
            if self.session:
                self.session.logger.info(f"Restored scene '{scene_name}'")

    def on_scene_time_changed(self, time):
        """Handle time changes in scene mode for preview"""
        self.scene_animation.preview_at_time(time)

    def on_scene_play_requested(self):
        """Handle play request from scene timeline"""
        current_time = self.scene_timeline_widget.timeline_controls.current_time
        self.scene_animation.play(start_time=current_time)

    def on_scene_pause_requested(self):
        """Handle pause request from scene timeline"""
        self.scene_animation.stop_playing()

    def on_scene_record_requested(self):
        """Handle record request from scene timeline"""
        # Check if already recording
        if self.scene_animation.is_recording:
            # Stop recording
            self.scene_animation.stop_playing()
            return

        # Check if there are scenes to record
        if not self.scene_animation.scenes:
            self.session.logger.warning("No scenes to record. Add some scenes to the timeline first.")
            return

        # Get save path and recording options using file dialog
        save_path, resolution = self._get_movie_save_path_and_options()
        if save_path:
            self.scene_animation.record(save_path, resolution=resolution)
            res_str = f" at {resolution}" if resolution else " at display resolution"
            self.session.logger.info(f"Starting recording to {save_path}{res_str}")

    def _get_movie_save_path_and_options(self):
        """Get save path and recording options using enhanced dialog"""
        dialog = MovieRecordingDialog(self.session, parent=self)
        if dialog.exec():
            return dialog.get_save_path(), dialog.get_resolution()
        return None, None

    def _get_movie_save_path(self):
        """Get save path for movie recording using file dialog (legacy method)"""
        from chimerax.ui.open_save import SaveDialog
        save_dialog = SaveDialog(self.session, parent=self)
        save_dialog.setNameFilter("Video Files (*.mp4 *.mov *.avi *.wmv)")
        if save_dialog.exec():
            file_path = save_dialog.selectedFiles()[0]
            return file_path
        return None

    def on_scene_animation_time_changed(self, time):
        """Handle time updates from scene animation during playback"""
        # Update the timeline controls to show current time
        self.scene_timeline_widget.timeline_controls.set_current_time(time)
        self.scene_timeline_widget.timeline_scene.set_current_time(time)

    def on_scene_animation_started(self):
        """Handle scene animation playback started"""
        # Ensure timeline controls show playing state
        self.scene_timeline_widget.timeline_controls.is_playing = True
        self.scene_timeline_widget.timeline_controls.play_btn.setText("⏸")

    def on_scene_animation_stopped(self):
        """Handle scene animation playback stopped"""
        # Ensure timeline controls show paused state
        self.scene_timeline_widget.timeline_controls.is_playing = False
        self.scene_timeline_widget.timeline_controls.play_btn.setText("▶")

    def on_scene_duration_changed(self, new_duration):
        """Handle duration change from zoom buttons"""
        # Update the scene animation duration
        self.scene_animation.set_duration(new_duration)
        # Also update the timeline controls duration
        self.scene_timeline_widget.timeline_controls.set_duration(new_duration)
        print(f"Scene animation duration set to {new_duration:.2f}s")

    def on_scene_reset_requested(self):
        """Handle reset request from return button"""
        # Stop any current playback
        self.scene_animation.stop_playing()
        # Reset time to 0 and preview at that time
        self.scene_animation.preview_at_time(0.0)
        print("Scene timeline reset to 0.0s")

    def on_scene_recording_started(self):
        """Handle scene recording started"""
        # Update record button appearance to show recording state
        record_btn = self.scene_timeline_widget.timeline_controls.record_btn
        record_btn.setText("⏹")  # Square stop symbol
        record_btn.setStyleSheet("background-color: #ff9800; color: white; border-radius: 15px; font-weight: bold;")

    def on_scene_recording_stopped(self):
        """Handle scene recording stopped"""
        # Reset record button appearance
        record_btn = self.scene_timeline_widget.timeline_controls.record_btn
        record_btn.setText("●")  # Circle record symbol
        record_btn.setStyleSheet("background-color: #f44336; color: white; border-radius: 15px; font-weight: bold;")

    def _create_transport_controls(self):
        """Create the transport control panel with play/pause buttons."""
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_frame.setFixedHeight(40)

        layout = QHBoxLayout(controls_frame)
        layout.setContentsMargins(5, 5, 5, 5)

        # Play/Pause button
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self.play_pause_btn.setFixedSize(60, 30)
        layout.addWidget(self.play_pause_btn)

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setFixedSize(60, 30)
        layout.addWidget(self.stop_btn)

        # Frame display
        self.frame_label = QLabel("Frame: 0")
        layout.addWidget(self.frame_label)

        # FPS control
        layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        self.fps_spin.setFixedWidth(60)
        layout.addWidget(self.fps_spin)

        # Zoom controls
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
        return controls_frame

    @Slot(object)
    def add_model_track(self, model):
        """Add an animation track for the specified model."""
        track_name = f"{model.name} (#{model.id_string})"
        track_index = len(self.timeline_scene.track_rows)
        self.track_models[track_index] = model
        self.add_track(track_name, is_parent=True)  # Make it a parent track

    def add_camera_track(self):
        """Add a default camera animation track."""
        track_name = "Camera"
        track_index = len(self.timeline_scene.track_rows)
        self.track_models[track_index] = self.session.view.camera
        self.add_track(track_name, is_parent=True)  # Make it a parent track

    # label -> row
    @Slot(int)
    def _on_label_selected(self, index: int):
        y = RULER_HEIGHT + index * TRACK_HEIGHT
        self.timeline_view.ensureVisible(0, y, 1, TRACK_HEIGHT)

    @Slot(int)
    def _on_track_selected(self, index: int):
        """Handle track selection - update detail view with model info."""
        # Clear previous highlights
        for row in self.timeline_scene.track_rows:
            row.set_highlighted(False)

        # Clear all manual header backgrounds so CSS takes over
        for i in range(self.track_header.count()):
            item = self.track_header.item(i)
            if item:
                item.setBackground(QBrush())  # Clear manual background

        # Highlight selected track
        if 0 <= index < len(self.timeline_scene.track_rows):
            self.timeline_scene.track_rows[index].set_highlighted(True)

        if index in self.track_models:
            model = self.track_models[index]
            if hasattr(model, 'id_string'):  # Regular model
                display_text = f"Model: {model.name}\nID: #{model.id_string}"
            else:  # Camera or other special track
                display_text = f"Camera"
            self.track_detail_view.set_track(display_text)
            self.track_detail_view.set_model(model)
        else:
            self.track_detail_view.set_track("No model selected")
            self.track_detail_view.set_model(None)

    @Slot(int, bool)
    def _on_track_hovered(self, index: int, is_hovered: bool):
        """Handle track hover - sync hover state between timeline and header."""
        if 0 <= index < len(self.timeline_scene.track_rows):
            self.timeline_scene.track_rows[index].set_hovered(is_hovered)

        # Also update header hover state
        if 0 <= index < self.track_header.count():
            item = self.track_header.item(index)
            if item:
                if is_hovered:
                    item.setBackground(QBrush(QColor("#2A3F5F")))
                else:
                    # Reset to default - check if it's selected
                    if self.track_header.currentRow() == index:
                        item.setBackground(QBrush(QColor("#3D5A80")))  # Keep selection color
                    else:
                        item.setBackground(QBrush())  # Clear background

    @Slot(str, object)
    def _on_keyframe_requested(self, property_name: str, value: object):
        """Handle keyframe creation request."""
        # Get the currently selected track
        current_track = self.track_header.currentRow()
        if current_track < 0:
            return

        # Find the root parent track (in case a subtrack is selected)
        root_track = self._get_root_parent_track(current_track)

        # Create or find subtrack for this property under the root track
        subtrack_index = self._get_or_create_subtrack(root_track, property_name)

        # Add keyframe at current frame
        current_frame = self.timeline_scene.get_current_frame()

        # Check for existing keyframe at this frame and remove it
        self._remove_keyframe_at_frame(subtrack_index, current_frame)

        self.insert_keyframe(subtrack_index, current_frame, value)

    def _get_root_parent_track(self, track_index: int):
        """Get the root parent track (handle case where subtrack is selected)."""
        # If this track is a subtrack, find its parent
        if track_index in self.track_parents:
            return self.track_parents[track_index]
        # Otherwise, it's already a root track
        return track_index

    def _get_or_create_subtrack(self, parent_track_index: int, property_name: str):
        """Get or create a subtrack for a specific property."""
        # Initialize subtracks dict for this track if needed
        if parent_track_index not in self.track_subtracks:
            self.track_subtracks[parent_track_index] = {}

        # Check if subtrack already exists
        if property_name in self.track_subtracks[parent_track_index]:
            existing_subtrack = self.track_subtracks[parent_track_index][property_name]
            return existing_subtrack

        # Create new subtrack
        parent_model = self.track_models.get(parent_track_index)
        if parent_model:
            if hasattr(parent_model, 'name'):
                subtrack_name = f"  └─ {property_name}"
            else:  # Camera
                subtrack_name = f"  └─ {property_name}"
        else:
            subtrack_name = f"  └─ {property_name}"

        # Find the correct position to insert the subtrack (after parent and any existing subtracks)
        insertion_position = self._find_subtrack_insertion_position(parent_track_index)

        # Shift all track indices that come after the insertion point
        self._shift_track_indices_after(insertion_position)

        # Create the subtrack at the insertion position
        subtrack_index = insertion_position
        self.track_subtracks[parent_track_index][property_name] = subtrack_index
        self.track_models[subtrack_index] = parent_model  # Reference same model
        self.track_parents[subtrack_index] = parent_track_index  # Record parent relationship

        # Insert to UI at specific position
        self.insert_track(insertion_position, subtrack_name)

        return subtrack_index

    def _find_subtrack_insertion_position(self, parent_track_index: int):
        """Find the position where a new subtrack should be inserted."""
        # Start right after the parent track
        insertion_pos = parent_track_index + 1

        # If parent already has subtracks, find the position after the last one
        if parent_track_index in self.track_subtracks:
            for property_name, existing_subtrack_index in self.track_subtracks[parent_track_index].items():
                if existing_subtrack_index >= insertion_pos:
                    insertion_pos = existing_subtrack_index + 1

        return insertion_pos

    def _shift_track_indices_after(self, insertion_position: int):
        """Shift all track indices that are >= insertion_position by +1."""
        # Update track_models mapping
        old_models = dict(self.track_models)
        self.track_models.clear()
        for track_index, model in old_models.items():
            if track_index >= insertion_position:
                self.track_models[track_index + 1] = model
            else:
                self.track_models[track_index] = model

        # Update track_subtracks mapping
        old_subtracks = dict(self.track_subtracks)
        self.track_subtracks.clear()
        for parent_index, subtracks in old_subtracks.items():
            # Update parent index if needed
            new_parent_index = parent_index + 1 if parent_index >= insertion_position else parent_index
            self.track_subtracks[new_parent_index] = {}

            # Update subtrack indices
            for property_name, subtrack_index in subtracks.items():
                new_subtrack_index = subtrack_index + 1 if subtrack_index >= insertion_position else subtrack_index
                self.track_subtracks[new_parent_index][property_name] = new_subtrack_index

        # Update track_parents mapping
        old_parents = dict(self.track_parents)
        self.track_parents.clear()
        for subtrack_index, parent_index in old_parents.items():
            new_subtrack_index = subtrack_index + 1 if subtrack_index >= insertion_position else subtrack_index
            new_parent_index = parent_index + 1 if parent_index >= insertion_position else parent_index
            self.track_parents[new_subtrack_index] = new_parent_index

    @Slot(int)
    def _on_track_collapsed(self, track_index: int):
        """Handle track collapse/expand."""
        if track_index in self.collapsed_tracks:
            # Track is currently collapsed, expand it
            self.collapsed_tracks.remove(track_index)
            self._show_subtracks(track_index)
            self.track_header.set_track_expanded(track_index, True)
        else:
            # Track is currently expanded, collapse it
            self.collapsed_tracks.add(track_index)
            self._hide_subtracks(track_index)
            self.track_header.set_track_expanded(track_index, False)

    def _hide_subtracks(self, parent_track_index: int):
        """Hide all subtracks of a parent track."""
        if parent_track_index in self.track_subtracks:
            for property_name, subtrack_index in self.track_subtracks[parent_track_index].items():
                # Hide in both header and timeline
                self.track_header.set_track_visible(subtrack_index, False)
                self.timeline_scene.set_track_visible(subtrack_index, False)
                # Hide all keyframes for this subtrack
                self._hide_subtrack_keyframes(subtrack_index)

    def _show_subtracks(self, parent_track_index: int):
        """Show all subtracks of a parent track."""
        if parent_track_index in self.track_subtracks:
            for property_name, subtrack_index in self.track_subtracks[parent_track_index].items():
                # Show in both header and timeline
                self.track_header.set_track_visible(subtrack_index, True)
                self.timeline_scene.set_track_visible(subtrack_index, True)
                # Show all keyframes for this subtrack
                self._show_subtrack_keyframes(subtrack_index)

    def _hide_subtrack_keyframes(self, track_index: int):
        """Hide all keyframes for a specific subtrack."""
        if track_index >= len(self.timeline_scene.track_rows):
            return

        # Find and hide all keyframes for this track
        for item in self.timeline_scene.items():
            if (isinstance(item, KeyframeItem) and
                hasattr(item, '_track_row') and
                item._track_row.index == track_index):
                item.setVisible(False)

    def _show_subtrack_keyframes(self, track_index: int):
        """Show all keyframes for a specific subtrack."""
        if track_index >= len(self.timeline_scene.track_rows):
            return

        # Find and show all keyframes for this track
        for item in self.timeline_scene.items():
            if (isinstance(item, KeyframeItem) and
                hasattr(item, '_track_row') and
                item._track_row.index == track_index):
                item.setVisible(True)

    @Slot(int)
    def _on_frame_changed(self, frame: int):
        """Handle frame changes from the timeline."""
        self.frame_label.setText(f"Frame: {frame}")
        if self.session:
            self.session.logger.info(f"Frame changed to {frame}, evaluating animation...")
        # Apply animation at this frame
        self._evaluate_animation_at_frame(frame)

    @Slot()
    def _toggle_playback(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start timeline playback."""
        self.is_playing = True
        self.play_pause_btn.setText("Pause")
        # Calculate timer interval from FPS
        interval = int(1000 / self.fps)  # milliseconds
        self.playback_timer.start(interval)
        if self.session:
            self.session.logger.info(f"Started keyframe playback at {self.fps} FPS (interval: {interval}ms)")

    def _pause_playback(self):
        """Pause timeline playback."""
        self.is_playing = False
        self.play_pause_btn.setText("Play")
        self.playback_timer.stop()

    @Slot()
    def _stop_playback(self):
        """Stop playback and return to frame 0."""
        self._pause_playback()
        self.timeline_scene.set_current_frame(0)
        self.timeline_view.frame_changed.emit(0)

    def _advance_frame(self):
        """Advance to the next frame during playback."""
        current_frame = self.timeline_scene.get_current_frame()
        next_frame = current_frame + 1

        # Loop back to start if we reach the end
        if next_frame > self.timeline_scene.num_frames:
            next_frame = 0

        self.timeline_scene.set_current_frame(next_frame)
        self.timeline_view.frame_changed.emit(next_frame)

    @Slot(int)
    def _on_fps_changed(self, fps: int):
        """Handle FPS changes."""
        self.fps = fps
        if self.is_playing:
            # Restart timer with new interval
            interval = int(1000 / self.fps)
            self.playback_timer.start(interval)

        # Sync FPS with scene animation manager
        if hasattr(self, 'scene_animation'):
            self.scene_animation.set_fps(fps)

    def _evaluate_animation_at_frame(self, frame: int):
        """Evaluate and apply animation at the given frame."""
        for track_index, model in self.track_models.items():
            if model is None:
                continue

            # Check if this track has subtracks (animated properties)
            if track_index in self.track_subtracks:
                for property_name, subtrack_index in self.track_subtracks[track_index].items():
                    if property_name == "transform":
                        # Get interpolated value for this frame
                        interpolated_value = self._interpolate_keyframes(subtrack_index, frame)

                        if interpolated_value is not None:
                            # Apply the transformation using positions (plural) for complex models
                            if hasattr(model, 'positions'):
                                try:
                                    # The interpolated value could be a single Place or a Places array
                                    from chimerax.geometry import Place, Places

                                    if isinstance(interpolated_value, Places):
                                        # Multiple positions for complex models
                                        model.positions = interpolated_value
                                    elif isinstance(interpolated_value, Place):
                                        # Single position - convert to Places format
                                        model.positions = Places([interpolated_value])
                                    elif hasattr(interpolated_value, '__iter__') and all(isinstance(p, Place) for p in interpolated_value):
                                        # List of Place objects
                                        model.positions = Places(interpolated_value)
                                    else:
                                        if self.session:
                                            self.session.logger.warning(f"Invalid interpolated value type: {type(interpolated_value)}")
                                except Exception as e:
                                    if self.session:
                                        self.session.logger.error(f"Error setting model positions: {e}")
                                    # Fallback: try to preserve current positions
                                    pass

    def _interpolate_keyframes(self, track_index: int, frame: int):
        """Interpolate between keyframes for the given track at the given frame."""
        if track_index >= len(self.timeline_scene.track_rows):
            return None

        # Collect all keyframes for this track from the scene
        keyframes = []
        for item in self.timeline_scene.items():
            if (isinstance(item, KeyframeItem) and
                hasattr(item, '_track_row') and
                item._track_row.index == track_index):
                keyframes.append((item.data.frame, item.data.value))

        if not keyframes:
            return None

        # Sort keyframes by frame
        keyframes.sort(key=lambda x: x[0])

        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf_frame, kf_value in keyframes:
            if kf_frame <= frame:
                prev_kf = (kf_frame, kf_value)
            if kf_frame >= frame and next_kf is None:
                next_kf = (kf_frame, kf_value)
                break

        # If we're exactly on a keyframe, return that value
        if prev_kf and prev_kf[0] == frame:
            return prev_kf[1]
        if next_kf and next_kf[0] == frame:
            return next_kf[1]

        # If we only have one keyframe or we're before/after all keyframes
        if not prev_kf and next_kf:
            return next_kf[1]  # Before first keyframe
        if prev_kf and not next_kf:
            return prev_kf[1]  # After last keyframe
        if not prev_kf and not next_kf:
            return None  # No keyframes

        # Interpolate between keyframes
        if prev_kf and next_kf and prev_kf[0] != next_kf[0]:
            # Calculate interpolation factor
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            t = max(0, min(1, t))  # Clamp to [0, 1]

            # Interpolate Place/Places objects
            from chimerax.geometry import Place, Places

            # Handle both single Place and multiple Places
            if ((isinstance(prev_kf[1], (Place, Places)) and isinstance(next_kf[1], (Place, Places))) or
                (hasattr(prev_kf[1], '__iter__') and hasattr(next_kf[1], '__iter__') and
                 all(isinstance(p, Place) for p in prev_kf[1]) and all(isinstance(p, Place) for p in next_kf[1]))):
                try:
                    # Normalize to consistent format for interpolation
                    prev_positions = prev_kf[1] if isinstance(prev_kf[1], Places) else Places([prev_kf[1]]) if isinstance(prev_kf[1], Place) else Places(list(prev_kf[1]))
                    next_positions = next_kf[1] if isinstance(next_kf[1], Places) else Places([next_kf[1]]) if isinstance(next_kf[1], Place) else Places(list(next_kf[1]))

                    # Ensure both have same number of positions
                    if len(prev_positions) != len(next_positions):
                        if self.session:
                            self.session.logger.warning(f"Mismatched position counts: {len(prev_positions)} vs {len(next_positions)}")
                        return prev_kf[1]  # Fallback

                    # Get interpolation center
                    if track_index in self.track_models:
                        model = self.track_models[track_index]
                        if hasattr(model, 'view') or model == self.session.view.camera:
                            # Camera interpolation
                            center = self._get_scene_center()
                        else:
                            # Model interpolation
                            center = self._get_model_center(model)
                    else:
                        center = [0, 0, 0]

                    # Ensure center is a numpy array as expected by Place.interpolate()
                    import numpy as np
                    if not isinstance(center, np.ndarray):
                        center = np.array(center, dtype=np.float32)

                    # Interpolate each position
                    interpolated_positions = []
                    for i in range(len(prev_positions)):
                        result = prev_positions[i].interpolate(next_positions[i], center, t)
                        if isinstance(result, Place):
                            interpolated_positions.append(result)
                        else:
                            if self.session:
                                self.session.logger.warning(f"Position {i} interpolation returned {type(result)}, expected Place")
                            interpolated_positions.append(prev_positions[i])  # Fallback

                    # Return in the same format as input
                    if len(interpolated_positions) == 1 and isinstance(prev_kf[1], Place):
                        return interpolated_positions[0]  # Single Place
                    else:
                        return Places(interpolated_positions)  # Multiple Places
                except Exception as e:
                    if self.session:
                        self.session.logger.error(f"Error interpolating Place objects: {e}")
                    # Fallback to the previous keyframe value
                    return prev_kf[1]

        return prev_kf[1] if prev_kf else next_kf[1]

    def _get_scene_center(self):
        """Calculate the center of rotation for camera interpolation."""
        if not self.session:
            return [0, 0, 0]
        
        try:
            # Try to get ChimeraX's current center of rotation first
            if hasattr(self.session.view, 'center_of_rotation'):
                cor = self.session.view.center_of_rotation
                if cor is not None:
                    return [cor[0], cor[1], cor[2]]
        except:
            pass
        
        try:
            # Get bounds of all visible models
            from chimerax.atomic import AtomicStructure
            bounds_list = []
            
            for model in self.session.models:
                if isinstance(model, AtomicStructure) and model.visible:
                    try:
                        bounds = model.bounds()
                        if bounds is not None:
                            bounds_list.append(bounds)
                    except:
                        continue
                elif hasattr(model, 'bounds') and model.visible:
                    try:
                        bounds = model.bounds()
                        if bounds is not None:
                            bounds_list.append(bounds)
                    except:
                        continue
            
            if bounds_list:
                # Calculate overall bounds
                from chimerax.geometry import union_bounds
                overall_bounds = union_bounds(bounds_list)
                if overall_bounds is not None:
                    # Return center of bounds
                    center = overall_bounds.center()
                    return [center[0], center[1], center[2]]
        except:
            pass
        
        # Fallback to origin if we can't calculate scene center
        return [0, 0, 0]

    def _get_model_center(self, model):
        """Get the center of a model for in-place rotation."""
        if not model:
            return [0, 0, 0]
        
        try:
            # Try to get the model's bounding box center
            if hasattr(model, 'bounds'):
                bounds = model.bounds()
                if bounds is not None:
                    center = bounds.center()
                    return [center[0], center[1], center[2]]
        except:
            pass
        
        try:
            # Fallback: use the model's current position as center
            if hasattr(model, 'position'):
                pos = model.position.translation()
                return [pos[0], pos[1], pos[2]]
        except:
            pass
        
        # Final fallback
        return [0, 0, 0]

    def _interpolate_camera_direct(self, place1, place2, t):
        """Direct linear interpolation for camera positions."""
        from chimerax.geometry import Place
        import numpy as np
        
        # Interpolate position (translation) linearly
        pos1 = place1.translation()
        pos2 = place2.translation()
        interp_pos = pos1 + t * (pos2 - pos1)
        
        # Interpolate rotation using quaternion slerp for smooth rotation
        # Get rotation matrices
        rot1 = place1.axes()
        rot2 = place2.axes()
        
        # Convert to quaternions for better interpolation
        q1 = self._matrix_to_quaternion(rot1)
        q2 = self._matrix_to_quaternion(rot2)
        
        # Spherical linear interpolation (slerp) for quaternions
        interp_q = self._slerp_quaternion(q1, q2, t)
        
        # Convert back to rotation matrix
        interp_rot = self._quaternion_to_matrix(interp_q)
        
        # Create new Place with interpolated position and rotation
        return Place(axes=interp_rot, origin=interp_pos)

    def _matrix_to_quaternion(self, matrix):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        import numpy as np
        
        # Simplified conversion - assumes matrix is orthogonal
        trace = np.trace(matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
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
        """Spherical linear interpolation between two quaternions."""
        import numpy as np
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, use -q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very similar, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Calculate slerp
        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2

    def _quaternion_to_matrix(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        import numpy as np
        
        w, x, y, z = q
        
        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to matrix
        matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        return matrix

    @Slot(list)
    def _on_keyframes_deleted(self, deleted_keyframes):
        """Handle keyframe deletion."""
        # The keyframes have already been removed from the scene
        # We could add additional cleanup here if needed
        pass

    def _remove_keyframe_at_frame(self, track_index: int, frame: int):
        """Remove any existing keyframe at the specified frame."""
        if track_index >= len(self.timeline_scene.track_rows):
            return

        # Find keyframes in the scene that belong to this track and frame
        items_to_remove = []
        for item in self.timeline_scene.items():
            if (isinstance(item, KeyframeItem) and
                hasattr(item, '_track_row') and
                item._track_row.index == track_index and
                item.data.frame == frame):
                items_to_remove.append(item)

        for item in items_to_remove:
            self.timeline_scene.removeItem(item)

    def add_track(self, name: str, is_parent: bool = False):
        row = self.timeline_scene.add_track(name)
        self.track_header.add_track(name, is_parent)  # keep in sync
        return row

    def insert_track(self, position: int, name: str, is_parent: bool = False):
        """Insert a track at a specific position."""
        row = self.timeline_scene.insert_track(position, name)
        self.track_header.insert_track(position, name, is_parent)  # keep in sync
        return row

    def insert_keyframe(self, track_index: int, frame: int, value: any):  # noqa: ANN401 – allow arbitrary
        kf_data = KeyframeData(frame, value)
        row = self.timeline_scene.track_rows[track_index]
        kf_item = row.add_keyframe(kf_data)
        return kf_item

    def insert_clip(self, track_index: int, start: int, end: int, payload: any):  # noqa: ANN401
        clip_data = ClipData(start, end, payload)
        row = self.timeline_scene.track_rows[track_index]
        clip_item = row.add_clip(clip_data)
        return clip_item

    def _zoom_in(self):
        """Zoom in the timeline."""
        self.timeline_scene.zoom_in()
        self._update_zoom_label()

    def _zoom_out(self):
        """Zoom out the timeline."""
        self.timeline_scene.zoom_out()
        self._update_zoom_label()

    def _zoom_fit(self):
        """Zoom to fit all content in view."""
        # Calculate zoom to fit timeline width in view
        view_width = self.timeline_view.viewport().width()
        scene_width = self.timeline_scene.num_frames * self.timeline_scene.base_frame_width
        if scene_width > 0:
            fit_zoom = view_width / scene_width * 0.9  # 90% to leave some margin
            self.timeline_scene.set_zoom(fit_zoom)
            self._update_zoom_label()

    def _update_zoom_label(self):
        """Update the zoom percentage label."""
        zoom_percent = int(self.timeline_scene.zoom_factor * 100)
        self.zoom_label.setText(f"{zoom_percent}%")

    @Slot(str, int)
    def _on_scene_dropped(self, scene_name: str, frame: int):
        """Handle scene drop onto timeline."""
        if not self.session:
            return

        # Get the scene from the session
        scene = self.session.scenes.get_scene(scene_name)
        if not scene:
            self.session.logger.warning(f"Scene '{scene_name}' not found")
            return

        # Get models from the scene's named_view positions (these are the models that were saved)
        scene_models = list(scene.named_view.positions.keys())
        self.session.logger.info(f"Dropping scene '{scene_name}' with {len(scene_models)} models at frame {frame}")

        # Create tracks for each model in the scene
        for model in scene_models:
            # Check if this model still exists in the session
            if model not in self.session.models.list():
                continue

            # Find existing track for this model
            existing_track = None
            for track_index, track_model in self.track_models.items():
                if track_model == model:
                    existing_track = track_index
                    break

            if existing_track is None:
                # Create new track for this model
                if hasattr(model, 'name') and hasattr(model, 'id_string'):
                    track_name = f"{model.name} (#{model.id_string})"
                else:
                    track_name = f"Model {id(model)}"

                track_index = len(self.timeline_scene.track_rows)
                self.track_models[track_index] = model
                self.add_track(track_name, is_parent=True)
                existing_track = track_index

            # Create or find the transform subtrack
            subtrack_index = self._get_or_create_subtrack(existing_track, "transform")

            # Remove any existing keyframe at this frame
            self._remove_keyframe_at_frame(subtrack_index, frame)

            # Add keyframe with the model positions from the scene (note: positions plural!)
            scene_positions = scene.named_view.positions[model]
            self.session.logger.info(f"  Adding transform keyframe for {getattr(model, 'name', 'unnamed model')} with {len(scene_positions) if hasattr(scene_positions, '__len__') else 1} positions")
            self.insert_keyframe(subtrack_index, frame, scene_positions)

            # Also handle model-specific scene data if it exists
            if model in scene.scene_models:
                restore_implemented, model_scene_data = scene.scene_models[model]
                # For now, we'll focus on position data, but this is where we could
                # expand to handle other model properties like display settings,
                # coloring, etc. in separate subtracks

        # Handle camera data from the scene
        # Get camera track (create if needed)
        camera_track = self._get_or_create_camera_track()

        if camera_track is not None:
            # Create or find the camera transform subtrack
            camera_subtrack = self._get_or_create_subtrack(camera_track, "camera")

            # Remove any existing keyframe at this frame
            self._remove_keyframe_at_frame(camera_subtrack, frame)

            # Add keyframe with camera data from the scene's main view
            # The scene stores camera data in main_view_data
            if scene.main_view_data and 'camera' in scene.main_view_data:
                camera_data = scene.main_view_data['camera']
                self.insert_keyframe(camera_subtrack, frame, camera_data)
            else:
                # Fallback to current camera if no scene camera data
                camera = self.session.view.camera
                camera_data = {
                    'position': camera.position,
                    'field_of_view': getattr(camera, 'field_of_view', None),
                    'field_width': getattr(camera, 'field_width', None)
                }
                self.insert_keyframe(camera_subtrack, frame, camera_data)

    def _get_or_create_camera_track(self):
        """Get existing camera track or create one if it doesn't exist"""
        # Look for existing camera track
        for track_index, model in self.track_models.items():
            if hasattr(model, 'view'):  # This indicates it's a camera track
                return track_index

        # Camera track doesn't exist, check if we have a camera track by name
        camera_track_index = None
        for i, row in enumerate(self.timeline_scene.track_rows):
            if row.track_name.lower() == "camera":
                camera_track_index = i
                break

        if camera_track_index is not None:
            # Mark this as camera track in our tracking
            self.track_models[camera_track_index] = self.session.view
            return camera_track_index

        # No camera track exists, so we return None and let the calling code decide
        # whether to create one. For scene drops, we'll add camera keyframes to
        # the existing camera track if it exists.
        return None

    @Slot(int)
    def _on_track_deleted(self, track_index: int):
        """Handle track deletion."""
        if track_index < 0 or track_index >= len(self.timeline_scene.track_rows):
            return
        
        # Check if this is the camera track - prevent deletion
        if track_index in self.track_models:
            model = self.track_models[track_index]
            if hasattr(model, 'view') or model == self.session.view.camera:
                # This is the camera track, don't allow deletion
                return
        
        # Check if this is a parent track with subtracks
        if track_index in self.track_subtracks:
            # Delete all subtracks first
            subtracks_to_delete = list(self.track_subtracks[track_index].values())
            # Sort in reverse order to delete from bottom up
            subtracks_to_delete.sort(reverse=True)
            
            for subtrack_index in subtracks_to_delete:
                self._delete_track_at_index(subtrack_index)
        
        # Delete the main track
        self._delete_track_at_index(track_index)

    def _delete_track_at_index(self, track_index: int):
        """Delete a track at the specified index and update all data structures."""
        if track_index < 0 or track_index >= len(self.timeline_scene.track_rows):
            return
        
        # Remove all keyframes for this track from the scene
        items_to_remove = []
        for item in self.timeline_scene.items():
            if (isinstance(item, KeyframeItem) and 
                hasattr(item, '_track_row') and 
                item._track_row.index == track_index):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.timeline_scene.removeItem(item)
        
        # Remove the track row from the scene
        track_row = self.timeline_scene.track_rows[track_index]
        self.timeline_scene.removeItem(track_row)
        self.timeline_scene.track_rows.pop(track_index)
        
        # Remove from header
        self.track_header.takeItem(track_index)
        
        # Update all data structures by shifting indices down
        self._shift_track_indices_after_deletion(track_index)
        
        # Update the scene to reflect changes
        self.timeline_scene._update_scene_rect()
        self.timeline_scene._draw_playhead()

    def _shift_track_indices_after_deletion(self, deleted_index: int):
        """Shift all track indices that are > deleted_index by -1."""
        # Update track_models mapping
        old_models = dict(self.track_models)
        self.track_models.clear()
        for track_index, model in old_models.items():
            if track_index < deleted_index:
                self.track_models[track_index] = model
            elif track_index > deleted_index:
                self.track_models[track_index - 1] = model
            # Skip the deleted index
        
        # Update track_subtracks mapping
        old_subtracks = dict(self.track_subtracks)
        self.track_subtracks.clear()
        for parent_index, subtracks in old_subtracks.items():
            if parent_index == deleted_index:
                continue  # Skip deleted parent
            
            # Update parent index if needed
            new_parent_index = parent_index - 1 if parent_index > deleted_index else parent_index
            self.track_subtracks[new_parent_index] = {}
            
            # Update subtrack indices
            for property_name, subtrack_index in subtracks.items():
                if subtrack_index == deleted_index:
                    continue  # Skip deleted subtrack
                new_subtrack_index = subtrack_index - 1 if subtrack_index > deleted_index else subtrack_index
                self.track_subtracks[new_parent_index][property_name] = new_subtrack_index
        
        # Update track_parents mapping
        old_parents = dict(self.track_parents)
        self.track_parents.clear()
        for subtrack_index, parent_index in old_parents.items():
            if subtrack_index == deleted_index or parent_index == deleted_index:
                continue  # Skip deleted relationships
            
            new_subtrack_index = subtrack_index - 1 if subtrack_index > deleted_index else subtrack_index
            new_parent_index = parent_index - 1 if parent_index > deleted_index else parent_index
            self.track_parents[new_subtrack_index] = new_parent_index
        
        # Update collapsed_tracks set
        old_collapsed = set(self.collapsed_tracks)
        self.collapsed_tracks.clear()
        for collapsed_index in old_collapsed:
            if collapsed_index == deleted_index:
                continue  # Skip deleted track
            new_collapsed_index = collapsed_index - 1 if collapsed_index > deleted_index else collapsed_index
            self.collapsed_tracks.add(new_collapsed_index)
        
        # Update track row indices in the remaining tracks
        for i, track_row in enumerate(self.timeline_scene.track_rows):
            track_row.index = i
            # Update position
            new_y = i * TRACK_HEIGHT + RULER_HEIGHT
            track_row.setPos(0, new_y)
            
            # Update all keyframes for this track to match the new position
            for item in self.timeline_scene.items():
                if (isinstance(item, KeyframeItem) and 
                    hasattr(item, '_track_row') and 
                    item._track_row == track_row):
                    # Update keyframe Y position to match new track position
                    kf_y = new_y + TRACK_HEIGHT / 2
                    item.setPos(item.pos().x(), kf_y)
        
        # Update track widget indices in header
        old_track_widgets = dict(self.track_header.track_widgets)
        self.track_header.track_widgets.clear()
        for old_index, widget in old_track_widgets.items():
            if old_index == deleted_index:
                continue  # Skip deleted widget
            new_index = old_index - 1 if old_index > deleted_index else old_index
            widget.track_index = new_index
            self.track_header.track_widgets[new_index] = widget

    # Scene Mode Methods
    def on_scene_added(self, scene_name):
        """Handle scene added to timeline"""
        # Scene is already added to timeline via drag/drop
        # Find the time where this scene was added and add it to scene animation
        for marker_data in self.scene_timeline_widget.timeline_scene.scene_markers:
            time, name = marker_data[:2]  # Only take first 2 elements
            if name == scene_name:
                # Add the scene to the scene animation manager at this time
                self.scene_animation.add_scene_at_time(scene_name, time)
                self.session.logger.info(f"Added scene '{scene_name}' to animation at {time:.2f}s")
                break

    def on_scene_removed(self, scene_name):
        """Handle scene removed from timeline"""
        # Remove scene from scene animation manager
        self.scene_animation.remove_scene(scene_name)

    def on_scene_moved(self, scene_name, old_time, new_time):
        """Handle scene moved on timeline"""
        # Remove scene from old time and add at new time in scene animation manager
        self.scene_animation.remove_scene_at_time(old_time)
        self.scene_animation.add_scene_at_time(scene_name, new_time)

    def on_scene_selected(self, scene_name):
        """Handle scene selected on timeline"""
        # This is called when user clicks on a scene marker
        # Scene restoration is handled directly in the timeline widget
        pass

    def on_scene_time_changed(self, time):
        """Handle time scrubber changed in scene mode"""
        # Preview the animation at this time
        self.scene_animation.preview_at_time(time)

    def on_scene_play_requested(self):
        """Handle play button pressed in scene mode"""
        # Get current time from timeline controls
        current_time = self.scene_timeline_widget.timeline_controls.current_time

        # Sync scene data from timeline to animation manager
        self._sync_timeline_to_scene_animation()

        # Start playback from current time
        self.scene_animation.play(start_time=current_time)

    def on_scene_pause_requested(self):
        """Handle pause button pressed in scene mode"""
        self.scene_animation.stop_playing()

    def _sync_timeline_to_scene_animation(self):
        """Sync scene markers from timeline to scene animation manager"""
        # Clear existing scenes from animation manager
        self.scene_animation.clear_all_scenes()

        # Set duration to match timeline
        duration = self.scene_timeline_widget.timeline_scene.duration
        self.scene_animation.set_duration(duration)

        # Add all scenes from timeline to animation manager
        for marker_data in self.scene_timeline_widget.timeline_scene.scene_markers:
            if len(marker_data) >= 4:
                # Full format with transition data
                time, scene_name, pixmap, transition_data = marker_data
                self.scene_animation.add_scene_at_time(
                    scene_name, time,
                    transition_data.get('type', 'linear'),
                    transition_data.get('fade_models', False)
                )
            elif len(marker_data) >= 2:
                # Backward compatibility - no transition data
                time, scene_name = marker_data[:2]
                self.scene_animation.add_scene_at_time(scene_name, time)

    # Scene Animation Signal Handlers
    def on_scene_animation_time_changed(self, time):
        """Handle scene animation time change"""
        # Update timeline controls to reflect current time
        self.scene_timeline_widget.timeline_controls.set_current_time(time)
        self.scene_timeline_widget.timeline_scene.set_current_time(time)

    def on_scene_animation_started(self):
        """Handle scene animation playback started"""
        # Update timeline controls to show playing state
        self.scene_timeline_widget.timeline_controls.is_playing = True
        self.scene_timeline_widget.timeline_controls.play_btn.setText("⏸")

    def on_scene_animation_stopped(self):
        """Handle scene animation playback stopped"""
        # Update timeline controls to show stopped state
        self.scene_timeline_widget.timeline_controls.is_playing = False
        self.scene_timeline_widget.timeline_controls.play_btn.setText("▶")


class MovieRecordingDialog(QDialog):
    """Dialog for movie recording options including resolution"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Record Animation")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # File selection
        file_group = QGroupBox("Output File")
        file_layout = QHBoxLayout(file_group)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Choose output file...")
        file_layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)

        layout.addWidget(file_group)

        # Resolution selection
        res_group = QGroupBox("Recording Resolution")
        res_layout = QVBoxLayout(res_group)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Display Resolution (Current)",
            "4K UHD (3840×2160)",
            "1080p Full HD (1920×1080)",
            "720p HD (1280×720)",
            "480p SD (640×480)",
            "Custom..."
        ])

        # Set default based on settings
        self._set_default_resolution()

        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)
        res_layout.addWidget(self.resolution_combo)

        # Custom resolution inputs (initially hidden)
        self.custom_widget = QWidget()
        custom_layout = QHBoxLayout(self.custom_widget)
        custom_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 7680)
        self.width_spin.setValue(1920)
        custom_layout.addWidget(self.width_spin)
        custom_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 4320)
        self.height_spin.setValue(1080)
        custom_layout.addWidget(self.height_spin)
        self.custom_widget.hide()
        res_layout.addWidget(self.custom_widget)

        layout.addWidget(res_group)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.accept)
        self.record_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        button_layout.addWidget(self.record_btn)

        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(button_layout)

    def _set_default_resolution(self):
        """Set the default resolution based on settings"""
        try:
            from .settings import get_settings
            settings = get_settings(self.session)
            default_res = settings.recording_resolution

            # Map settings values to combo box items
            if default_res == '4k':
                self.resolution_combo.setCurrentText("4K UHD (3840×2160)")
            elif default_res == '1080p':
                self.resolution_combo.setCurrentText("1080p Full HD (1920×1080)")
            elif default_res == 'custom':
                self.resolution_combo.setCurrentText("Custom...")
            else:
                # Default to 1080p if unknown setting
                self.resolution_combo.setCurrentText("1080p Full HD (1920×1080)")
        except Exception:
            # Fallback to 1080p if settings not available
            self.resolution_combo.setCurrentText("1080p Full HD (1920×1080)")

    def browse_file(self):
        from chimerax.ui.open_save import SaveDialog
        save_dialog = SaveDialog(self.session, parent=self)
        save_dialog.setNameFilter("Video Files (*.mp4 *.mov *.avi *.wmv)")
        if save_dialog.exec():
            file_path = save_dialog.selectedFiles()[0]
            self.file_path_edit.setText(file_path)

    def on_resolution_changed(self, text):
        self.custom_widget.setVisible("Custom" in text)

    def get_save_path(self):
        return self.file_path_edit.text()

    def get_resolution(self):
        text = self.resolution_combo.currentText()
        if "Display Resolution" in text:
            return None
        elif "4K UHD" in text:
            return "4k"
        elif "1080p" in text:
            return "1080p"
        elif "720p" in text:
            return "720p"
        elif "480p" in text:
            return "480p"
        elif "Custom" in text:
            return (self.width_spin.value(), self.height_spin.value())
        return None
