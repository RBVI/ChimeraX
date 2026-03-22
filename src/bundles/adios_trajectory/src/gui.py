"""
BP5 trajectory slider tool.

Provides a lightweight panel with a step slider, play/pause button, and
ring-buffer prefetch indicator.  Opens automatically when a BP5Trajectory
model is added to the session.
"""

from __future__ import annotations


def show_trajectory_slider(session, trajectory) -> None:
    """Open (or raise) the trajectory slider for *trajectory*."""
    from chimerax.core.tools import ToolInstance

    existing = session.tools.find_by_class(BP5SliderTool)
    for tool in existing:
        if tool.trajectory is trajectory:
            tool.display(True)
            return

    BP5SliderTool(session, trajectory)


class BP5SliderTool:
    """
    Minimal Qt panel for scrubbing a BP5Trajectory.

    Not a full ToolInstance subclass so it can be shown without a tool
    registration step (useful during development).
    """

    def __init__(self, session, trajectory):
        self.session = session
        self.trajectory = trajectory
        self._playing = False
        self._timer = None
        self._build_ui()

    def _build_ui(self):
        from Qt.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QLabel
        )
        from Qt.QtCore import Qt

        dlg = QDialog()
        dlg.setWindowTitle(f"BP5 Trajectory: {self.trajectory.name}")
        dlg.setMinimumWidth(400)

        layout = QVBoxLayout(dlg)

        # Step slider
        slider = QSlider(Qt.Orientation.Horizontal)
        n = max(1, self.trajectory.n_steps - 1)
        slider.setRange(0, n)
        slider.setValue(max(0, self.trajectory.current_step))
        slider.setTickInterval(max(1, n // 10))
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.valueChanged.connect(self._on_slider)
        layout.addWidget(slider)

        # Step label
        self._step_label = QLabel(self._step_text())
        layout.addWidget(self._step_label)

        # Controls
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        play_btn = QPushButton("Play")
        play_btn.setCheckable(True)
        play_btn.toggled.connect(self._on_play)
        ctrl.addWidget(play_btn)

        prefetch_btn = QPushButton("Prefetch ±5")
        prefetch_btn.clicked.connect(lambda: self.trajectory.prefetch(
            self.trajectory.current_step, radius=5))
        ctrl.addWidget(prefetch_btn)

        self._slider = slider
        self._dlg = dlg
        dlg.show()

    def _step_text(self) -> str:
        traj = self.trajectory
        return f"Step {traj.current_step + 1} / {traj.n_steps}"

    def _on_slider(self, step: int):
        self.trajectory.goto_step(step)
        self._step_label.setText(self._step_text())
        self.session.update_loop.update_graphics_now()

    def _on_play(self, checked: bool):
        from Qt.QtCore import QTimer
        self._playing = checked
        if checked:
            self._timer = QTimer()
            self._timer.setInterval(33)  # ~30 fps
            self._timer.timeout.connect(self._advance_frame)
            self._timer.start()
        else:
            if self._timer:
                self._timer.stop()
                self._timer = None

    def _advance_frame(self):
        traj = self.trajectory
        next_step = (traj.current_step + 1) % traj.n_steps
        self._slider.setValue(next_step)
