"""
Scene Timeline Widget for ChimeraX Animations

This module provides a simplified timeline interface similar to Chimera, focused on
scene-based animation rather than complex keyframe interpolation.

Components:
- SceneTimelineWidget: Main widget with scenes, actions, and timeline panels
- SceneThumbnailWidget: Displays scene thumbnails
- TimelineControlWidget: Timeline scrubber and playback controls
"""

from Qt.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QSlider,
    QSizePolicy,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QMenu,
    QAction,
    QStyle,
)
from Qt.QtCore import Qt, QSize, QTimer, Signal, QPointF, QEvent
from Qt.QtGui import QPixmap, QIcon, QPainter, QColor, QPalette

import io
from PIL import Image

from chimerax.scenes.tool import SCENE_EVENT_MIME_FORMAT

class ActionThumbnailWidget(QWidget):
    """Widget displaying an action thumbnail (Rock, Roll, etc.)"""

    action_selected = Signal(str)  # action_name

    def __init__(self, action_name, icon_name=None, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        self.setFixedSize(80, 80)
        self.setStyleSheet("""
            ActionThumbnailWidget {
                border: 2px solid #555;
                border-radius: 40px;
                background-color: #444;
            }
            ActionThumbnailWidget:hover {
                border-color: #777;
                background-color: #555;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Icon label (circular)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(60, 60)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("""
            border: none;
            background-color: #666;
            border-radius: 30px;
            color: white;
            font-weight: bold;
        """)

        # Set icon or text
        if icon_name:
            # Try to load icon, fallback to text
            self.icon_label.setText(action_name[0].upper())
        else:
            self.icon_label.setText(action_name[0].upper())

        layout.addWidget(self.icon_label, alignment=Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.action_selected.emit(self.action_name)
        super().mousePressEvent(event)


class TimelineControlWidget(QWidget):
    """Timeline control widget with scrubber and playback controls"""

    time_changed = Signal(float)  # time in seconds
    play_requested = Signal()
    pause_requested = Signal()
    record_requested = Signal()
    add_scene_requested = Signal(float)  # time to add scene at
    duration_changed = Signal(float)  # new duration in seconds
    reset_requested = Signal()  # reset timeline to zero
    preferences_requested = Signal()  # preferences button clicked

    def __init__(self, duration=5.0, session=None, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.current_time = 0.0
        self.is_playing = False
        self.session = session

        self.setup_ui()

        # Timer for playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.update_playback)

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Playback controls
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 30)
        self.play_btn.clicked.connect(self.toggle_playback)

        self.add_keyframe_btn = QPushButton("+")
        self.add_keyframe_btn.setFixedSize(30, 30)
        self.add_keyframe_btn.setStyleSheet(
            "background-color: #2196F3; color: white; border-radius: 15px; font-weight: bold;"
        )
        self.add_keyframe_btn.clicked.connect(self.add_scene_at_current_time)

        self.reverse_btn = QPushButton()
        self.reverse_btn.setFixedSize(30, 30)
        self.reverse_btn.setIconSize(
            QSize(18, 18)
        )  # Set icon size explicitly for proper centering
        self.reverse_btn.clicked.connect(self.reset_timeline)
        self._update_reverse_btn_icon()  # Set initial icon

        layout.addWidget(self.play_btn)
        layout.addWidget(self.add_keyframe_btn)
        layout.addWidget(self.reverse_btn)
        layout.addStretch()

        # Time adjustment controls
        time_label = QLabel("Adjust Time:")
        layout.addWidget(time_label)

        self.time_minus_30_btn = QPushButton("-30")
        self.time_minus_30_btn.setFixedHeight(30)
        self.time_minus_30_btn.clicked.connect(lambda: self.adjust_duration(-30))

        self.time_minus_5_btn = QPushButton("-5")
        self.time_minus_5_btn.setFixedHeight(30)
        self.time_minus_5_btn.clicked.connect(lambda: self.adjust_duration(-5))

        self.time_minus_1_btn = QPushButton("-1")
        self.time_minus_1_btn.setFixedHeight(30)
        self.time_minus_1_btn.clicked.connect(lambda: self.adjust_duration(-1))

        self.time_plus_1_btn = QPushButton("+1")
        self.time_plus_1_btn.setFixedHeight(30)
        self.time_plus_1_btn.clicked.connect(lambda: self.adjust_duration(1))

        self.time_plus_5_btn = QPushButton("+5")
        self.time_plus_5_btn.setFixedHeight(30)
        self.time_plus_5_btn.clicked.connect(lambda: self.adjust_duration(5))

        self.time_plus_30_btn = QPushButton("+30")
        self.time_plus_30_btn.setFixedHeight(30)
        self.time_plus_30_btn.clicked.connect(lambda: self.adjust_duration(30))

        layout.addWidget(self.time_minus_30_btn)
        layout.addWidget(self.time_minus_5_btn)
        layout.addWidget(self.time_minus_1_btn)
        layout.addWidget(self.time_plus_1_btn)
        layout.addWidget(self.time_plus_5_btn)
        layout.addWidget(self.time_plus_30_btn)
        layout.addStretch()

        # Record button
        self.record_btn = QPushButton("●")
        self.record_btn.setFixedSize(30, 30)
        self.record_btn.setStyleSheet(
            "background-color: #f44336; color: white; border-radius: 15px; font-weight: bold;"
        )
        self.record_btn.clicked.connect(self.record_requested.emit)
        layout.addWidget(self.record_btn)

        # Preferences button with gear icon
        from chimerax.ui.icons import get_qt_icon
        from Qt.QtWidgets import QToolButton
        self.preferences_btn = QToolButton()
        self.preferences_btn.setIcon(get_qt_icon("gear"))
        self.preferences_btn.setToolTip("Preferences")
        self.preferences_btn.setFixedSize(30, 30)
        self.preferences_btn.clicked.connect(self.preferences_requested.emit)
        layout.addWidget(self.preferences_btn)

    def toggle_playback(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        self.is_playing = True
        self.play_btn.setText("⏸")
        # Don't start internal timer - let scene animation manage timing
        self.play_requested.emit()

    def pause_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_timer.stop()  # Stop any internal timer
        self.pause_requested.emit()

    def update_playback(self):
        # This method is no longer used for timeline control
        # Scene animation will manage the timing
        pass

    def set_current_time(self, time):
        """Set the current time position"""
        self.current_time = max(0, min(time, self.duration))
        # If playing, emit time change to update preview
        if self.is_playing:
            self.time_changed.emit(self.current_time)

    def add_scene_at_current_time(self):
        """Handle add scene button click"""
        self.add_scene_requested.emit(self.current_time)

    def reset_timeline(self):
        """Reset timeline to zero seconds"""
        self.current_time = 0.0
        self.time_changed.emit(0.0)
        self.reset_requested.emit()

    def adjust_duration(self, amount):
        """Adjust animation duration by the specified amount in seconds"""
        new_duration = self.duration + amount
        # Ensure duration stays at minimum 1 second
        if new_duration >= 1.0:
            self.duration = new_duration
            self.duration_changed.emit(self.duration)

    def set_duration(self, duration):
        """Set the timeline duration"""
        self.duration = duration

    def _update_reverse_btn_icon(self):
        """Update reverse button icon based on current color scheme"""
        standard_icon = self.style().standardIcon(QStyle.SP_MediaSkipBackward)
        icon_size = 18  # Match the iconSize we set

        # Get the pixmap at the correct size
        pixmap = standard_icon.pixmap(icon_size, icon_size)

        if self.session and hasattr(self.session, "ui") and self.session.ui.dark_mode():
            # Recolor icon to white for dark mode
            colored_pixmap = QPixmap(icon_size, icon_size)
            colored_pixmap.fill(Qt.transparent)
            painter = QPainter(colored_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
            painter.fillRect(colored_pixmap.rect(), QColor(255, 255, 255))
            painter.end()
            # Create icon and explicitly set the pixmap at the correct size
            icon = QIcon()
            icon.addPixmap(colored_pixmap, QIcon.Normal, QIcon.Off)
            self.reverse_btn.setIcon(icon)
        else:
            # Use the pixmap at the correct size for light mode too
            icon = QIcon()
            icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
            self.reverse_btn.setIcon(icon)

    def changeEvent(self, event):
        """Handle widget change events, including palette changes"""
        if event.type() == QEvent.PaletteChange:
            # Update icon when color scheme changes
            self._update_reverse_btn_icon()
        super().changeEvent(event)


class TimelineSceneWidget(QWidget):
    """Main timeline widget showing scene markers and timeline"""

    scene_dropped = Signal(str, float)  # scene_name, time
    scene_moved = Signal(str, float, float)  # scene_name, old_time, new_time
    scene_deleted = Signal(str)  # scene_name
    time_clicked = Signal(float)  # time position clicked

    def __init__(self, duration=5.0, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.scene_markers = []  # List of (time, scene_name, thumbnail_pixmap, transition_data) tuples
        self.current_time = 0.0
        self.drag_time = None  # Time position for drag preview
        self.setFixedHeight(120)  # Increased height for thumbnails
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Enable focus to receive key events
        self.setFocusPolicy(Qt.StrongFocus)

        # Selected scene tracking
        self.selected_scene = None

        # Store reference to session for key handling
        self.session = None

        # Dragging state for scene markers
        self.dragging_scene = None  # Scene name being dragged
        self.potential_drag_scene = None  # Scene that might be dragged if mouse moves far enough
        self.drag_start_pos = None  # Mouse position when drag started
        self.original_scene_time = None  # Original time of scene being dragged

        # Dragging state for playhead scrubbing
        self.dragging_playhead = False  # Whether we're dragging the playhead

    def add_scene_marker(self, time, scene_name, transition_data=None):
        """Add a scene marker at the specified time"""
        if transition_data is None:
            transition_data = {"type": "linear", "fade_models": False}

        # Get scene thumbnail from session
        thumbnail_pixmap = self._get_scene_thumbnail(scene_name)
        self.scene_markers.append((time, scene_name, thumbnail_pixmap, transition_data))
        self.scene_markers.sort()  # Keep sorted by time
        self.update()

    def remove_scene_marker(self, scene_name):
        """Remove scene marker by name"""
        self.scene_markers = [
            (t, s, p, td) for t, s, p, td in self.scene_markers if s != scene_name
        ]
        self.update()

    def _get_scene_thumbnail(self, scene_name):
        """Get thumbnail pixmap for a scene"""
        session = self._get_session()
        if session and session.scenes:
            scene = session.scenes.get_scene(scene_name)
            if scene:
                try:
                    # Get thumbnail data and convert to pixmap
                    thumbnail_data = scene.get_thumbnail()
                    if thumbnail_data:
                        pixmap = QPixmap()
                        import base64

                        image_data = base64.b64decode(thumbnail_data)
                        pixmap.loadFromData(image_data)
                        # Scale to reasonable size for timeline
                        return pixmap.scaled(
                            40, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                except Exception as e:
                    pass
                    # print(f"Error loading thumbnail for {scene_name}: {e}")

        # Return default thumbnail if scene not found or error
        default_pixmap = QPixmap(40, 30)
        default_pixmap.fill(QColor(100, 100, 100))
        return default_pixmap

    def _get_scene_at_position(self, x):
        """Get scene name if click position is on a scene marker"""
        width = self.width()

        # Check each scene marker to see if click is within its bounds
        for marker_data in self.scene_markers:
            if len(marker_data) >= 2:
                time, scene_name = marker_data[:2]
                marker_x = int((time / self.duration) * width)

                # Check if click is within the thumbnail bounds (40px wide, centered on marker_x)
                thumb_left = marker_x - 20
                thumb_right = marker_x + 20

                if thumb_left <= x <= thumb_right:
                    return scene_name

        return None

    def _get_scene_time(self, scene_name):
        """Get the time of a scene marker"""
        for marker_data in self.scene_markers:
            if len(marker_data) >= 2:
                time, name = marker_data[:2]
                if name == scene_name:
                    return time
        return None

    def set_current_time(self, time):
        """Set current playback time"""
        self.current_time = time
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(42, 42, 42))

            # Draw timeline ruler
            width = self.width()
            height = self.height()

            # Timeline background
            painter.fillRect(0, 20, width, height - 40, QColor(60, 60, 60))

            # Time marks every 100ms (0.1 seconds)
            painter.setPen(QColor(200, 200, 200))
            tick_interval = 0.1  # 100ms

            # Major ticks every second with labels (better font size)
            font = painter.font()
            font.setPointSize(10)  # Slightly larger than before but still compact
            painter.setFont(font)
            for i in range(int(self.duration) + 1):
                x = int((i / self.duration) * width)
                painter.drawLine(x, 0, x, 20)  # Draw tick marks in ruler area only
                painter.drawText(x + 2, 15, f"{i}s")

            # Minor ticks every 100ms
            painter.setPen(QColor(150, 150, 150))
            num_ticks = int(self.duration / tick_interval) + 1
            for i in range(num_ticks):
                time_pos = i * tick_interval
                # Skip major tick positions (every second)
                if time_pos % 1.0 != 0:
                    x = int((time_pos / self.duration) * width)
                    # Draw smaller tick marks in the ruler area only
                    painter.drawLine(x, 10, x, 20)

            # Draw scene markers with thumbnails
            for marker_data in self.scene_markers:
                if len(marker_data) >= 3:
                    time, scene_name, thumbnail_pixmap = marker_data[
                        :3
                    ]  # Only take first 3 elements
                else:
                    # Handle old format without thumbnails
                    time, scene_name = marker_data[:2]
                    thumbnail_pixmap = self._get_scene_thumbnail(scene_name)

                x = int((time / self.duration) * width)

                # Draw thumbnail background
                thumb_rect_x = x - 20
                thumb_rect_y = 25
                thumb_rect_w = 40
                thumb_rect_h = 30

                # Background for thumbnail
                painter.setBrush(QColor(60, 60, 60))
                painter.setPen(QColor(255, 255, 255))
                painter.drawRect(thumb_rect_x, thumb_rect_y, thumb_rect_w, thumb_rect_h)

                # Draw thumbnail
                if thumbnail_pixmap and not thumbnail_pixmap.isNull():
                    painter.drawPixmap(
                        thumb_rect_x,
                        thumb_rect_y,
                        thumb_rect_w,
                        thumb_rect_h,
                        thumbnail_pixmap,
                    )

                # Draw border around thumbnail (highlight if selected)
                painter.setBrush(Qt.NoBrush)
                if scene_name == self.selected_scene:
                    # Highlight selected scene with thicker, brighter border
                    painter.setPen(QColor(255, 255, 0, 255))  # Yellow selection
                    painter.drawRect(thumb_rect_x - 1, thumb_rect_y - 1, thumb_rect_w + 2, thumb_rect_h + 2)
                    painter.setPen(QColor(255, 255, 0, 200))
                else:
                    painter.setPen(QColor(100, 150, 255, 200))
                painter.drawRect(thumb_rect_x, thumb_rect_y, thumb_rect_w, thumb_rect_h)

                # Scene name below thumbnail
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(x - 25, 70, scene_name[:8])  # Truncate long names

                # Draw vertical line from scene marker up into timeline ruler area for clarity
                painter.setPen(
                    QColor(100, 150, 255, 200)
                )  # Blue line for scene position clarity
                painter.drawLine(
                    x, 0, x, 20
                )  # Line in the dark timeline ruler area (y=0 to y=20)

                # Small diamond indicator at the exact time position
                painter.setBrush(QColor(255, 100, 100))
                painter.setPen(QColor(255, 255, 255))
                diamond_points = [
                    QPointF(x, 20),  # top
                    QPointF(x + 4, 25),  # right
                    QPointF(x, 30),  # bottom
                    QPointF(x - 4, 25),  # left
                ]
                painter.drawPolygon(diamond_points)

            # Draw transition curves between scenes
            self._draw_transition_curves(painter, width, height)

            # Draw current time indicator
            x = int((self.current_time / self.duration) * width)
            painter.setPen(QColor(255, 100, 100))
            painter.drawLine(x, 0, x, height)

            # Draw drag preview indicator
            if self.drag_time is not None:
                x = int((self.drag_time / self.duration) * width)
                painter.setPen(QColor(100, 255, 100))  # Green for drag preview
                painter.drawLine(x, 0, x, height)

                # Draw "drop zone" indicator
                painter.setPen(QColor(100, 255, 100, 128))
                painter.setBrush(QColor(100, 255, 100, 50))
                painter.drawRect(x - 5, 20, 10, height - 40)

        finally:
            painter.end()  # Ensure painter is properly ended

    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        # Check for ChimeraX scene data first (preferred)
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            event.acceptProposedAction()
            return

        # Fallback to text format
        if event.mimeData().hasText():
            # Check if this is a scene name from the scenes tool
            scene_name = event.mimeData().text()
            session = self._get_session()

            if session and session.scenes.get_scene(scene_name):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def _get_session(self):
        """Helper to get session from parent widget hierarchy"""
        widget = self
        while widget:
            if hasattr(widget, "session"):
                return widget.session
            widget = widget.parent()
        return None

    def dragMoveEvent(self, event):
        """Handle drag move events"""
        if (
            event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT)
            or event.mimeData().hasText()
        ):
            # Update drag preview position
            x = event.position().x()
            self.drag_time = (x / self.width()) * self.duration
            self.drag_time = max(0, min(self.drag_time, self.duration))

            self.update()  # Trigger repaint to show drag preview
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        self.drag_time = None
        self.update()  # Clear drag preview
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        """Handle drop events"""
        scene_name = None

        # Try to get scene name from ChimeraX scene data first
        if event.mimeData().hasFormat(SCENE_EVENT_MIME_FORMAT):
            try:
                import json

                scene_data_bytes = event.mimeData().data(SCENE_EVENT_MIME_FORMAT)
                scene_data_str = scene_data_bytes.data().decode("utf-8")

                # Try to parse as JSON first
                try:
                    scene_data = json.loads(scene_data_str)
                    scene_name = scene_data.get("name")
                except json.JSONDecodeError:
                    # Fallback: treat as plain scene name
                    scene_name = scene_data_str

            except Exception as e:
                # print(f"Error parsing scene data: {e}")
                scene_name = None

        # Fallback to text format
        if not scene_name and event.mimeData().hasText():
            scene_name = event.mimeData().text()

        if scene_name:
            # Use the drag time if available, otherwise calculate from drop position
            if self.drag_time is not None:
                time = self.drag_time
            else:
                x = event.position().x()
                time = (x / self.width()) * self.duration
                time = max(0, min(time, self.duration))  # Clamp to valid range

            # Clear drag preview
            self.drag_time = None
            self.update()

            self.scene_dropped.emit(scene_name, time)
            event.acceptProposedAction()
        else:
            self.drag_time = None
            self.update()
            event.ignore()

    def mousePressEvent(self, event):
        """Handle mouse clicks on timeline for scene selection"""
        if event.button() == Qt.LeftButton:
            # Calculate time from click position
            x = event.position().x()
            time = (x / self.width()) * self.duration
            time = max(0, min(time, self.duration))  # Clamp to valid range

            # Check if we clicked on a scene marker
            clicked_scene = self._get_scene_at_position(x)
            if clicked_scene:
                # Select the scene for potential deletion
                self.selected_scene = clicked_scene
                self.setFocus()  # Ensure we can receive key events

                # Prepare for potential scene dragging, but don't actually drag until significant movement
                self.dragging_scene = None  # Don't set this until we actually start dragging
                self.potential_drag_scene = clicked_scene  # Track which scene might be dragged
                self.drag_start_pos = event.position()
                self.original_scene_time = self._get_scene_time(clicked_scene)
                self.dragging_playhead = False

                # Single click now only selects - don't restore scene
                # Don't move playhead when clicking on scenes
                self.update()  # Redraw to show selection
            else:
                # Clear scene selection and dragging state
                self.selected_scene = None
                self.dragging_scene = None
                self.potential_drag_scene = None
                self.drag_start_pos = event.position()
                self.original_scene_time = None
                self.dragging_playhead = True

                # Only update current time and emit signal when NOT clicking on a scene
                # Also check if we're clicking within timeline bounds
                if x <= self.width():  # Only if click is within timeline
                    self.set_current_time(time)
                    self.time_clicked.emit(time)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for dragging scene markers or playhead"""
        # Check if we have a potential drag that should become actual dragging
        if self.potential_drag_scene and self.drag_start_pos and not self.dragging_scene:
            current_pos = event.position()
            from Qt.QtWidgets import QApplication

            # Only start dragging if mouse has moved significantly (default drag distance)
            if (current_pos - self.drag_start_pos).manhattanLength() > QApplication.startDragDistance():
                # Now start actually dragging
                self.dragging_scene = self.potential_drag_scene
                self.potential_drag_scene = None

        if self.dragging_scene and self.drag_start_pos:
            # Calculate how far we've moved
            current_pos = event.position()

            # Calculate new time position
            x = current_pos.x()
            new_time = (x / self.width()) * self.duration
            new_time = max(0, min(new_time, self.duration))  # Clamp to valid range

            # Update the scene marker position
            self._move_scene_marker(self.dragging_scene, new_time)
            self.update()

        elif self.dragging_playhead and self.drag_start_pos:
            # Handle playhead dragging for timeline scrubbing
            current_pos = event.position()

            # Calculate new time position
            x = current_pos.x()
            new_time = (x / self.width()) * self.duration
            new_time = max(0, min(new_time, self.duration))  # Clamp to valid range

            # Update current time and preview animation at this time
            self.set_current_time(new_time)
            self.time_clicked.emit(new_time)
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish dragging"""
        if event.button() == Qt.LeftButton:
            if self.dragging_scene:
                # Calculate final time position for scene marker
                x = event.position().x()
                new_time = (x / self.width()) * self.duration
                new_time = max(0, min(new_time, self.duration))  # Clamp to valid range

                # Move the scene marker to final position
                old_time = self.original_scene_time
                self._move_scene_marker(self.dragging_scene, new_time)

                # Emit signal that scene was moved
                if old_time != new_time:
                    self.scene_moved.emit(self.dragging_scene, old_time, new_time)

                # Clear scene dragging state
                self.dragging_scene = None
                self.potential_drag_scene = None
                self.drag_start_pos = None
                self.original_scene_time = None

            elif self.potential_drag_scene:
                # Clear potential drag state (was a simple click, not a drag)
                self.potential_drag_scene = None
                self.drag_start_pos = None
                self.original_scene_time = None

            elif self.dragging_playhead:
                # Clear playhead dragging state
                self.dragging_playhead = False
                self.drag_start_pos = None

            self.update()

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-clicks for scene restoration"""
        if event.button() == Qt.LeftButton:
            # Calculate click position
            x = event.position().x()

            # Check if we double-clicked on a scene marker
            clicked_scene = self._get_scene_at_position(x)
            if clicked_scene:
                # Restore the double-clicked scene
                session = self._get_session()
                if session:
                    from chimerax.core.commands import run

                    run(session, f'scene restore "{clicked_scene}"')
                    # print(f"Restored scene: {clicked_scene}")

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Handle key presses for scene deletion"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.selected_scene:
                # Delete the selected scene
                self.scene_deleted.emit(self.selected_scene)
                self.selected_scene = None
                self.update()
            # Accept the event regardless to prevent forwarding to command line
            event.accept()
            return
        else:
            super().keyPressEvent(event)

    def _grab_focus_and_handle_delete(self):
        """Alternative approach: just handle delete in a more direct way"""
        if self.selected_scene:
            self.scene_deleted.emit(self.selected_scene)
            self.selected_scene = None
            self.update()

    def contextMenuEvent(self, event):
        """Show context menu on right-click for transition settings"""
        # print(f"DEBUG: Context menu event at position {event.pos().x()}")
        # Check if we right-clicked on a scene marker
        clicked_scene = self._get_scene_at_position(event.pos().x())
        # print(f"DEBUG: Clicked scene: {clicked_scene}")
        if clicked_scene:
            self._show_transition_menu(clicked_scene, event.globalPos())
        else:
            super().contextMenuEvent(event)

    def _show_transition_menu(self, scene_name, global_pos):
        """Show transition selection menu for a scene"""
        # print(f"DEBUG: Showing transition menu for scene '{scene_name}'")
        from .scene_animation import TRANSITION_TYPES

        # Get current transition data
        current_transition = self._get_scene_transition_data(scene_name)
        # print(f"DEBUG: Current transition data: {current_transition}")
        current_type = (
            current_transition.get("type", "linear") if current_transition else "linear"
        )
        current_fade = (
            current_transition.get("fade_models", False)
            if current_transition
            else False
        )

        menu = QMenu(self)

        # Add delete option at the top
        delete_action = menu.addAction(f"Delete '{scene_name}' from Timeline")
        delete_action.triggered.connect(lambda: self.scene_deleted.emit(scene_name))
        menu.addSeparator()

        # Transition type submenu
        transition_menu = menu.addMenu("Transition Type")

        transition_names = {
            "linear": "Linear",
            "ease_in_sine": "Ease In (Sine)",
            "ease_out_sine": "Ease Out (Sine)",
            "ease_in_out_sine": "Ease In-Out (Sine)",
            "ease_in_quad": "Ease In (Quad)",
            "ease_out_quad": "Ease Out (Quad)",
            "ease_in_out_quad": "Ease In-Out (Quad)",
            "ease_in_cubic": "Ease In (Cubic)",
            "ease_out_cubic": "Ease Out (Cubic)",
            "ease_in_out_cubic": "Ease In-Out (Cubic)",
        }

        for transition_key, transition_name in transition_names.items():
            action = QAction(transition_name, menu)
            action.setCheckable(True)
            action.setChecked(current_type == transition_key)
            # Fix lambda closure issue by storing the transition key in the action
            action.setProperty("transition_key", transition_key)
            action.setProperty("scene_name", scene_name)
            action.triggered.connect(self._on_transition_action_triggered)
            transition_menu.addAction(action)

        menu.addSeparator()

        # Fade models option
        fade_action = QAction("Fade Models", menu)
        fade_action.setCheckable(True)
        fade_action.setChecked(current_fade)
        fade_action.triggered.connect(
            lambda checked: self._set_scene_fade_models(scene_name, checked)
        )
        menu.addAction(fade_action)

        menu.exec(global_pos)

    def _on_transition_action_triggered(self):
        """Handle transition type action triggered"""
        action = self.sender()
        if action:
            transition_key = action.property("transition_key")
            scene_name = action.property("scene_name")
            # print(f"DEBUG: Action triggered for scene '{scene_name}', transition '{transition_key}'")
            self._set_scene_transition_type(scene_name, transition_key)

    def _get_scene_transition_data(self, scene_name):
        """Get transition data for a scene"""
        for time, name, pixmap, transition_data in self.scene_markers:
            if name == scene_name:
                return transition_data
        return None

    def _set_scene_transition_type(self, scene_name, transition_type):
        """Set transition type for a scene"""
        # print(f"DEBUG: Setting scene '{scene_name}' transition to '{transition_type}'")
        # Update the scene marker data
        for i, (time, name, pixmap, transition_data) in enumerate(self.scene_markers):
            if name == scene_name:
                # print(f"DEBUG: Found scene '{name}', old transition: {transition_data}")
                transition_data["type"] = transition_type
                self.scene_markers[i] = (time, name, pixmap, transition_data)
                # print(f"DEBUG: Updated transition data: {transition_data}")
                break

        # Update the scene animation manager
        self._sync_to_scene_animation()
        self.update()

    def _set_scene_fade_models(self, scene_name, fade_models):
        """Set fade models option for a scene"""
        # Update the scene marker data
        for i, (time, name, pixmap, transition_data) in enumerate(self.scene_markers):
            if name == scene_name:
                transition_data["fade_models"] = fade_models
                self.scene_markers[i] = (time, name, pixmap, transition_data)
                break

        # Update the scene animation manager
        self._sync_to_scene_animation()
        self.update()

    def _sync_to_scene_animation(self):
        """Sync timeline data to scene animation manager"""
        # print(f"DEBUG: _sync_to_scene_animation called")
        # Get the scene animation manager and update it with our current data
        scene_timeline_widget = self._get_scene_timeline_widget()
        # print(f"DEBUG: Found widget: {scene_timeline_widget}")
        if scene_timeline_widget and hasattr(scene_timeline_widget, "session"):
            session = scene_timeline_widget.session
            # print(f"DEBUG: Found session: {session}")
            # Get the scene animation manager from the session (stored as a custom attribute)
            scene_animation = getattr(session, "_scene_animation_manager", None)
            # print(f"DEBUG: Found scene_animation from session: {scene_animation}")
            if scene_animation:
                # Instead of clearing all scenes, just update this specific scene
                # Remove existing scene at this time/name first
                for time, name, pixmap, transition_data in self.scene_markers:
                    # Remove old scene if it exists
                    scene_animation.remove_scene(name)
                    # Add with new transition data
                    # print(f"DEBUG: Syncing scene '{name}' at {time}s with transition: {transition_data}")
                    scene_animation.add_scene_at_time(
                        name,
                        time,
                        transition_data.get("type", "linear"),
                        transition_data.get("fade_models", False),
                    )
            else:
                pass
            # print(f"DEBUG: Could not get scene_animation from session")
        else:
            pass
        # print(f"DEBUG: Could not find scene_timeline_widget or session")

    def _get_scene_timeline_widget(self):
        """Find the parent SceneTimelineWidget"""
        # print(f"DEBUG: Looking for SceneTimelineWidget parent...")
        parent = self.parent()
        level = 0
        while parent:
            # print(f"DEBUG: Parent level {level}: {parent.__class__.__name__}")
            if (
                hasattr(parent, "__class__")
                and "SceneTimelineWidget" in parent.__class__.__name__
            ):
                # print(f"DEBUG: Found SceneTimelineWidget at level {level}")
                return parent
            # Also check if this parent has scene_animation directly
            if hasattr(parent, "scene_animation"):
                # print(f"DEBUG: Found parent with scene_animation at level {level}: {parent.__class__.__name__}")
                return parent
            parent = parent.parent()
            level += 1
        # print(f"DEBUG: No suitable parent found")
        return None

    def _draw_transition_curves(self, painter, width, height):
        """Draw visual curves showing transition types between scenes"""
        if len(self.scene_markers) < 2:
            return

        # Sort markers by time
        sorted_markers = sorted(self.scene_markers, key=lambda x: x[0])

        # Draw curves between consecutive scenes
        for i in range(len(sorted_markers) - 1):
            scene1_data = sorted_markers[i]
            scene2_data = sorted_markers[i + 1]

            t1, name1 = scene1_data[:2]
            t2, name2 = scene2_data[:2]

            # Get transition data for the target scene
            transition_data = (
                scene2_data[3] if len(scene2_data) > 3 else {"type": "linear"}
            )
            transition_type = transition_data.get("type", "linear")

            # Calculate X positions
            x1 = int((t1 / self.duration) * width)
            x2 = int((t2 / self.duration) * width)

            # Only draw curve if scenes are far enough apart
            if x2 - x1 < 40:
                continue

            # Set curve color based on transition type
            curve_colors = {
                "linear": QColor(100, 100, 100, 150),
                "ease_in_sine": QColor(100, 150, 255, 150),
                "ease_out_sine": QColor(255, 150, 100, 150),
                "ease_in_out_sine": QColor(150, 255, 150, 150),
                "ease_in_quad": QColor(255, 100, 150, 150),
                "ease_out_quad": QColor(150, 255, 255, 150),
                "ease_in_out_quad": QColor(255, 255, 100, 150),
                "ease_in_cubic": QColor(200, 100, 255, 150),
                "ease_out_cubic": QColor(255, 200, 100, 150),
                "ease_in_out_cubic": QColor(100, 255, 200, 150),
            }

            curve_color = curve_colors.get(transition_type, QColor(100, 100, 100, 150))
            painter.setPen(curve_color)

            # Draw curve based on transition type
            curve_y = 85  # Below the scene markers
            curve_height = 15

            if transition_type == "linear":
                # Straight line
                painter.drawLine(x1, curve_y, x2, curve_y)
            else:
                # Draw approximated easing curve using line segments
                self._draw_easing_curve(
                    painter, x1, x2, curve_y, curve_height, transition_type
                )

            # Add small text label for transition type
            mid_x = (x1 + x2) // 2
            painter.setPen(QColor(200, 200, 200, 180))
            painter.drawText(
                mid_x - 20, curve_y + 12, transition_type.replace("_", " ").title()
            )

    def _draw_easing_curve(self, painter, x1, x2, y_center, height, transition_type):
        """Draw an approximated easing curve using line segments"""
        from .scene_animation import TRANSITION_TYPES

        easing_func = TRANSITION_TYPES.get(transition_type, TRANSITION_TYPES["linear"])
        num_segments = 20  # Number of line segments to approximate curve

        points = []
        for i in range(num_segments + 1):
            t = i / num_segments  # Linear time from 0 to 1
            eased_t = easing_func(t)  # Apply easing function

            # Map to screen coordinates
            x = x1 + (x2 - x1) * t
            y = y_center - height * (eased_t - 0.5)  # Center curve around y_center

            points.append(QPointF(x, y))

        # Draw the curve as connected line segments
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _move_scene_marker(self, scene_name, new_time):
        """Move a scene marker to a new time position"""
        # Find and update the scene marker
        for i, marker_data in enumerate(self.scene_markers):
            if len(marker_data) >= 2 and marker_data[1] == scene_name:
                # Preserve thumbnail and transition data if they exist
                if len(marker_data) >= 4:
                    # Full format: (time, name, thumbnail, transition_data)
                    self.scene_markers[i] = (
                        new_time,
                        scene_name,
                        marker_data[2],
                        marker_data[3],
                    )
                elif len(marker_data) >= 3:
                    # Backward compatibility: add default transition data
                    self.scene_markers[i] = (
                        new_time,
                        scene_name,
                        marker_data[2],
                        {"type": "linear", "fade_models": False},
                    )
                else:
                    # Very old format
                    self.scene_markers[i] = (
                        new_time,
                        scene_name,
                        None,
                        {"type": "linear", "fade_models": False},
                    )
                break

        # Re-sort markers by time
        self.scene_markers.sort(key=lambda x: x[0])


class SceneTimelineWidget(QWidget):
    """Main scene timeline widget combining all components"""

    scene_added = Signal(str)  # scene_name
    scene_removed = Signal(str)  # scene_name
    scene_moved = Signal(str, float, float)  # scene_name, old_time, new_time
    scene_selected = Signal(str)  # scene_name
    time_changed = Signal(float)  # time
    play_requested = Signal()
    pause_requested = Signal()
    record_requested = Signal()  # record animation
    duration_changed = Signal(float)  # new duration in seconds
    reset_requested = Signal()  # reset timeline to zero
    preferences_requested = Signal()  # preferences button clicked

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        # self.actions = ["Rock", "Roll"]  # Default actions

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Actions panel (top)
        # actions_frame = QFrame()
        # actions_frame.setFrameStyle(QFrame.StyledPanel)
        # actions_layout = QVBoxLayout(actions_frame)

        # actions_header = QLabel("▼Actions")
        # actions_header.setStyleSheet("font-weight: bold; color: white; padding: 5px;")
        # actions_layout.addWidget(actions_header)

        ## Actions container
        # actions_container = QWidget()
        # actions_grid = QHBoxLayout(actions_container)  # Horizontal layout for actions

        # Add default actions
        # for action in self.actions:
        #    action_widget = ActionThumbnailWidget(action)
        #    action_widget.action_selected.connect(self.apply_action)
        #    actions_grid.addWidget(action_widget)

        # actions_grid.addStretch()  # Push actions to the left
        # actions_layout.addWidget(actions_container)

        # Hide actions frame since we're not using it currently
        # main_layout.addWidget(actions_frame)

        # Timeline panel
        timeline_frame = QFrame()
        timeline_frame.setFrameStyle(QFrame.StyledPanel)
        timeline_layout = QVBoxLayout(timeline_frame)

        # Timeline controls
        self.timeline_controls = TimelineControlWidget(session=self.session)
        self.timeline_controls.time_changed.connect(self.time_changed.emit)
        # Connect timeline control signals to scene animation
        self.timeline_controls.play_requested.connect(self.on_play_requested)
        self.timeline_controls.pause_requested.connect(self.on_pause_requested)
        self.timeline_controls.record_requested.connect(self.on_record_requested)
        self.timeline_controls.add_scene_requested.connect(self.on_add_scene_requested)
        self.timeline_controls.duration_changed.connect(self.on_duration_changed)
        self.timeline_controls.reset_requested.connect(self.on_reset_requested)
        self.timeline_controls.preferences_requested.connect(self.preferences_requested.emit)
        timeline_layout.addWidget(self.timeline_controls)

        # Timeline scene view
        self.timeline_scene = TimelineSceneWidget()
        self.timeline_scene.session = self.session  # Pass session to timeline
        self.timeline_scene.scene_dropped.connect(self.on_scene_dropped)
        self.timeline_scene.scene_moved.connect(self.on_scene_moved)
        self.timeline_scene.scene_deleted.connect(self.on_scene_deleted)
        self.timeline_scene.time_clicked.connect(self.on_time_clicked)
        timeline_layout.addWidget(self.timeline_scene)

        main_layout.addWidget(timeline_frame)

        # Set panel proportions
        main_layout.setStretch(0, 1)  # Actions panel (smaller)
        main_layout.setStretch(1, 2)  # Timeline panel (larger)

    def on_scene_dropped(self, scene_name, time):
        """Handle scene dropped onto timeline"""
        # Add the scene marker directly to the timeline
        self.timeline_scene.add_scene_marker(time, scene_name)
        self.scene_added.emit(scene_name)
        # The parent tool will handle adding the scene to the animation

    def on_scene_moved(self, scene_name, old_time, new_time):
        """Handle scene marker moved on timeline"""
        # Scene marker is already moved in the timeline widget
        # Emit signal so parent tool can update the animation
        self.scene_moved.emit(scene_name, old_time, new_time)
        print(f"Scene '{scene_name}' moved from {old_time:.2f}s to {new_time:.2f}s")

    def on_scene_deleted(self, scene_name):
        """Handle scene deleted from timeline"""
        # Remove scene marker from timeline widget
        self.timeline_scene.remove_scene_marker(scene_name)

        # Also remove from the scene animation manager to prevent interpolation errors
        scene_animation = getattr(self.session, "_scene_animation_manager", None)
        if scene_animation:
            scene_animation.remove_scene(scene_name)
            #print(f"Scene '{scene_name}' removed from timeline and animation manager")
        else:
            #print(
            #    f"Scene '{scene_name}' removed from timeline (no animation manager found)"
            #)
            pass

    def on_time_clicked(self, time):
        """Handle timeline click to preview at that time"""
        # Update the timeline controls to reflect the new time
        self.timeline_controls.set_current_time(time)
        # Emit time changed signal so parent tool can preview at this time
        self.time_changed.emit(time)

    def on_play_requested(self):
        """Handle play button pressed"""
        self.play_requested.emit()

    def on_pause_requested(self):
        """Handle pause button pressed"""
        self.pause_requested.emit()

    def on_record_requested(self):
        """Handle record button pressed"""
        self.record_requested.emit()

    def on_add_scene_requested(self, time):
        """Handle add scene button pressed"""
        # Create a new scene with auto-generated name and save current state
        scene_name = self._generate_scene_name()
        if scene_name:
            # Save current state as a scene
            from chimerax.core.commands import run

            run(self.session, f'scene save "{scene_name}"')

            # Add the scene to the timeline
            self.timeline_scene.add_scene_marker(time, scene_name)
            self.scene_added.emit(scene_name)

            # print(f"Created scene '{scene_name}' at {time:.2f}s")

    def on_duration_changed(self, new_duration):
        """Handle duration change from zoom buttons"""
        # Update the timeline scene widget duration
        self.timeline_scene.duration = new_duration
        self.timeline_scene.update()  # Trigger repaint with new duration
        # Emit signal for parent to handle
        self.duration_changed.emit(new_duration)

    def on_reset_requested(self):
        """Handle reset button (return arrow) pressed"""
        # Reset timeline to time 0
        self.timeline_controls.set_current_time(0.0)
        self.timeline_scene.set_current_time(0.0)
        # Emit time change to trigger preview at time 0
        self.time_changed.emit(0.0)
        # Emit reset signal for parent to handle
        self.reset_requested.emit()
        # print("Timeline reset to 0.0s")

    def _generate_scene_name(self):
        """Generate a unique scene name"""
        base_name = "Scene"
        counter = 1

        # Get existing scene names
        existing_names = set()
        if self.session and self.session.scenes:
            existing_names = set(self.session.scenes.get_scene_names())

        # Find an unused name
        while True:
            scene_name = f"{base_name} {counter}"
            if scene_name not in existing_names:
                return scene_name
            counter += 1
            # Safety limit to avoid infinite loop
            if counter > 1000:
                return None

    def apply_action(self, action_name):
        """Apply an action (Rock, Roll, etc.)"""
        # This would implement specific actions
        # For now, just log the action
        self.session.logger.info(f"Applied action: {action_name}")

    def add_scene_marker(self, time, scene_name):
        """Add a scene marker to the timeline"""
        self.timeline_scene.add_scene_marker(time, scene_name)

    def remove_scene_marker(self, scene_name):
        """Remove a scene marker from the timeline"""
        self.timeline_scene.remove_scene_marker(scene_name)

    def set_current_time(self, time):
        """Set the current time on the timeline"""
        self.timeline_scene.set_current_time(time)
