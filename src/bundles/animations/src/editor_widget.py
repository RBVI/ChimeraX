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

"""Dual-mode animations editor wrapper."""

from __future__ import annotations

from typing import Optional

from Qt.QtCore import Signal
from Qt.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .keyframe_timeline_widget import KeyframeTimelineWidget
from .scene_timeline import SceneTimelineWidget
from .scene_timeline_controller import SceneTimelineController

__all__ = ["KeyframeEditorWidget", "MovieRecordingDialog"]


class CompactStackedWidget(QStackedWidget):
    """QStackedWidget that sizes itself to the current page."""

    def sizeHint(self):
        current_widget = self.currentWidget()
        if current_widget:
            return current_widget.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        current_widget = self.currentWidget()
        if current_widget:
            return current_widget.minimumSizeHint()
        return super().minimumSizeHint()


class KeyframeEditorWidget(QWidget):
    """Composite widget exposing keyframe and scene animation modes."""

    keyframeMoved = Signal(int, int, int)
    clipResized = Signal(int, int, int)

    def __init__(self, session=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        mode_controls_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup()

        self.keyframe_mode_btn = QPushButton("Keyframe Mode")
        self.keyframe_mode_btn.setCheckable(True)
        self.keyframe_mode_btn.setChecked(True)
        self.keyframe_mode_btn.setStyleSheet(
            """
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            """
        )

        self.scene_mode_btn = QPushButton("Scene Mode")
        self.scene_mode_btn.setCheckable(True)
        self.scene_mode_btn.setStyleSheet(
            """
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            """
        )

        self.mode_button_group.addButton(self.keyframe_mode_btn, 0)
        self.mode_button_group.addButton(self.scene_mode_btn, 1)
        self.mode_button_group.buttonClicked.connect(self.switch_mode)

        mode_controls_layout.addWidget(self.keyframe_mode_btn)
        mode_controls_layout.addWidget(self.scene_mode_btn)
        mode_controls_layout.addStretch()
        main_layout.addLayout(mode_controls_layout)

        self.stacked_widget = CompactStackedWidget()
        self.stacked_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.keyframe_widget = KeyframeTimelineWidget(self.session)
        self.keyframe_widget.keyframeMoved.connect(self.keyframeMoved.emit)
        self.keyframe_widget.clipResized.connect(self.clipResized.emit)
        self.keyframe_widget.preferences_requested.connect(self.show_preferences)
        self.stacked_widget.addWidget(self.keyframe_widget)

        self.scene_timeline_widget = SceneTimelineWidget(self.session)
        self.scene_timeline_controller = SceneTimelineController(
            self.session,
            self.scene_timeline_widget,
            fps=self.keyframe_widget.fps,
        )
        self.scene_timeline_widget.preferences_requested.connect(self.show_preferences)
        self.scene_timeline_widget.controller = self.scene_timeline_controller
        self.scene_animation = self.scene_timeline_controller.scene_animation
        self.stacked_widget.addWidget(self.scene_timeline_widget)

        self.fps_combo = self.keyframe_widget.fps_combo
        main_layout.addWidget(self.stacked_widget)

    def __getattr__(self, name):
        keyframe_widget = self.__dict__.get("keyframe_widget")
        if keyframe_widget is not None and hasattr(keyframe_widget, name):
            return getattr(keyframe_widget, name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def switch_mode(self, button):
        if button == self.keyframe_mode_btn:
            self.stacked_widget.setCurrentIndex(0)
        elif button == self.scene_mode_btn:
            self.stacked_widget.setCurrentIndex(1)

    def show_preferences(self):
        from .settings import AnimationsPreferencesDialog

        dialog = AnimationsPreferencesDialog(self.session, parent=self)
        dialog.show()

    def _on_fps_changed(self, fps: int):
        self.keyframe_widget._on_fps_changed(fps)
        self.scene_animation.set_fps(fps)

    def cleanup(self):
        self.keyframe_widget.cleanup()
        self.scene_timeline_controller.cleanup()


class MovieRecordingDialog:
    """Dialog for movie recording options including resolution."""

    def __init__(self, session, parent=None):
        from chimerax.ui.open_save import SaveDialog

        self.session = session
        self._dialog = SaveDialog(session, parent, "Record Animation")
        self._dialog.setNameFilter("Video Files (*.mp4 *.mov *.avi *.wmv)")
        self._dialog.setDefaultSuffix("mp4")
        self._dialog.selectFile("animation.mp4")
        self._setup_custom_area()

    def _setup_custom_area(self):
        from Qt.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

        options_area = self._dialog.custom_area
        layout = QVBoxLayout(options_area)
        layout.addWidget(QLabel("Recording Resolution:"))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(
            [
                "Display Resolution (Current)",
                "4K UHD (3840×2160)",
                "1080p Full HD (1920×1080)",
                "720p HD (1280×720)",
                "480p SD (640×480)",
                "Custom...",
            ]
        )
        self._set_default_resolution()
        self.resolution_combo.currentTextChanged.connect(self._on_resolution_changed)
        layout.addWidget(self.resolution_combo)

        self.custom_widget = QWidget()
        custom_layout = QHBoxLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
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
        layout.addWidget(self.custom_widget)

    def _set_default_resolution(self):
        try:
            from .settings import get_settings

            settings = get_settings(self.session)
            default_res = settings.recording_resolution

            if default_res == "4k":
                self.resolution_combo.setCurrentText("4K UHD (3840×2160)")
            elif default_res == "1080p":
                self.resolution_combo.setCurrentText("1080p Full HD (1920×1080)")
            elif default_res == "custom":
                self.resolution_combo.setCurrentText("Custom...")
            else:
                self.resolution_combo.setCurrentText("Display Resolution (Current)")
        except Exception:
            self.resolution_combo.setCurrentText("Display Resolution (Current)")

    def _on_resolution_changed(self, text):
        self.custom_widget.setVisible(text == "Custom...")

    def exec(self):
        return self._dialog.exec()

    def get_save_path(self):
        selected_files = self._dialog.selectedFiles()
        return selected_files[0] if selected_files else None

    def get_resolution(self):
        text = self.resolution_combo.currentText()
        if text == "Display Resolution (Current)":
            return None
        if text == "4K UHD (3840×2160)":
            return (3840, 2160)
        if text == "1080p Full HD (1920×1080)":
            return (1920, 1080)
        if text == "720p HD (1280×720)":
            return (1280, 720)
        if text == "480p SD (640×480)":
            return (640, 480)
        if text == "Custom...":
            return (self.width_spin.value(), self.height_spin.value())
        return None
