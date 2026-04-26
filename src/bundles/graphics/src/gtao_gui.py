"""AO parameter tuning GUI (GTAO + analytical sphere-sphere AO).

Usage from ChimeraX command line:
    from chimerax.graphics.gtao_gui import show
    show(session)
"""

from Qt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QGroupBox, QFormLayout, QPushButton)
from Qt.QtCore import Qt


class AOTuner(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Ambient Occlusion")
        self.setMinimumWidth(400)

        lp = self._lighting()
        layout = QVBoxLayout(self)

        # =================================================================
        # Sphere-sphere analytical AO
        # =================================================================
        sao_group = QGroupBox("Analytical Sphere-Sphere AO")
        sg = QVBoxLayout(sao_group)

        # Enable button
        btn_row = QHBoxLayout()
        self._sao_enable_btn = QPushButton("Enable")
        self._sao_enable_btn.clicked.connect(self._on_sao_enable)
        self._sao_disable_btn = QPushButton("Disable")
        self._sao_disable_btn.clicked.connect(self._on_sao_disable)
        self._sao_rebuild_btn = QPushButton("Rebuild")
        self._sao_rebuild_btn.clicked.connect(self._on_sao_enable)
        btn_row.addWidget(self._sao_enable_btn)
        btn_row.addWidget(self._sao_disable_btn)
        btn_row.addWidget(self._sao_rebuild_btn)
        sg.addLayout(btn_row)

        # Cutoff
        self._sao_cutoff_slider, self._sao_cutoff_spin = self._add_float_row(
            sg, "Cutoff (A):", 2.0, 20.0,
            getattr(lp, 'sphere_ao_cutoff', 5.0),
            decimals=1, step=0.5)

        # Max neighbors
        form = QFormLayout()
        self._sao_max_nb = QSpinBox()
        self._sao_max_nb.setRange(1, 64)
        self._sao_max_nb.setValue(int(getattr(lp, 'sphere_ao_max_neighbors', 16)))
        form.addRow("Max neighbors:", self._sao_max_nb)
        sg.addLayout(form)

        # Strength
        self._sao_strength_slider, self._sao_strength_spin = self._add_float_row(
            sg, "Strength:", 0.0, 10.0,
            getattr(lp, 'sphere_ao_strength', 2.0),
            decimals=2, step=0.1)
        self._sao_strength_spin.valueChanged.connect(self._on_sao_strength)

        layout.addWidget(sao_group)

        # =================================================================
        # GTAO (screen-space)
        # =================================================================
        gtao_group = QGroupBox("GTAO (screen-space)")
        gg = QVBoxLayout(gtao_group)

        form2 = QFormLayout()
        self._gtao_enable = QCheckBox()
        self._gtao_enable.setChecked(getattr(lp, 'gtao', 0) > 0)
        self._gtao_enable.toggled.connect(self._on_gtao_enable)
        form2.addRow("Enabled:", self._gtao_enable)

        self._gtao_directions = QSpinBox()
        self._gtao_directions.setRange(1, 16)
        self._gtao_directions.setValue(max(int(getattr(lp, 'gtao', 4)), 1))
        self._gtao_directions.valueChanged.connect(self._on_gtao_param)
        form2.addRow("Directions:", self._gtao_directions)

        self._gtao_steps = QSpinBox()
        self._gtao_steps.setRange(1, 16)
        self._gtao_steps.setValue(int(getattr(lp, 'gtao_steps', 4)))
        self._gtao_steps.valueChanged.connect(self._on_gtao_param)
        form2.addRow("Steps:", self._gtao_steps)
        gg.addLayout(form2)

        self._gtao_radius_slider, self._gtao_radius_spin = self._add_float_row(
            gg, "Radius (px):", 1.0, 100.0,
            getattr(lp, 'gtao_radius', 16.0),
            decimals=1, step=0.5)
        self._gtao_radius_spin.valueChanged.connect(self._on_gtao_param)

        self._gtao_intensity_slider, self._gtao_intensity_spin = self._add_float_row(
            gg, "Intensity:", 0.0, 5.0,
            getattr(lp, 'gtao_intensity', 1.5),
            decimals=2, step=0.05)
        self._gtao_intensity_spin.valueChanged.connect(self._on_gtao_param)

        self._gtao_bias_slider, self._gtao_bias_spin = self._add_float_row(
            gg, "Bias (x R):", 0.0, 0.2, 0.02,
            decimals=3, step=0.002)
        self._gtao_bias_spin.valueChanged.connect(self._on_gtao_param)

        layout.addWidget(gtao_group)
        layout.addStretch()

    # ------------------------------------------------------------------
    def _lighting(self):
        return self.session.main_view.render.lighting

    def _add_float_row(self, parent_layout, label, vmin, vmax, value,
                       decimals=2, step=0.1):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(self._val_to_slider(value, vmin, vmax))

        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)

        slider.valueChanged.connect(
            lambda v, s=spin, mn=vmin, mx=vmax:
                s.setValue(mn + (mx - mn) * v / 1000.0))
        spin.valueChanged.connect(
            lambda v, sl=slider, mn=vmin, mx=vmax:
                sl.setValue(self._val_to_slider(v, mn, mx)))

        row.addWidget(slider)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider, spin

    @staticmethod
    def _val_to_slider(value, vmin, vmax):
        if vmax <= vmin:
            return 0
        return int((value - vmin) / (vmax - vmin) * 1000.0)

    # ------------------------------------------------------------------
    # Sphere-sphere AO callbacks
    # ------------------------------------------------------------------
    def _on_sao_enable(self):
        lp = self._lighting()
        cutoff = self._sao_cutoff_spin.value()
        max_nb = self._sao_max_nb.value()
        lp.sphere_ao_cutoff = cutoff
        lp.sphere_ao_max_neighbors = max_nb
        lp.sphere_ao_strength = self._sao_strength_spin.value()

        r = self.session.main_view.render
        from .analytical_ao import enable_sphere_ao
        for d in self._impostor_drawings():
            enable_sphere_ao(d, r, cutoff=cutoff, max_neighbors=max_nb)
        self._redraw()

    def _on_sao_disable(self):
        from .analytical_ao import disable_sphere_ao
        for d in self._impostor_drawings():
            disable_sphere_ao(d)
        self._redraw()

    def _on_sao_strength(self, _=None):
        lp = self._lighting()
        lp.sphere_ao_strength = self._sao_strength_spin.value()
        self._redraw()

    def _impostor_drawings(self):
        """Find all impostor sphere drawings in the scene."""
        drawings = []
        view = self.session.main_view
        for d in view.drawing.all_drawings(displayed_only=True):
            sas = getattr(d, 'positions', None)
            if sas is not None and hasattr(sas, 'shift_and_scale_array'):
                sa = sas.shift_and_scale_array()
                if sa is not None and len(sa) > 0:
                    drawings.append(d)
        return drawings

    # ------------------------------------------------------------------
    # GTAO callbacks
    # ------------------------------------------------------------------
    def _on_gtao_enable(self, checked):
        lp = self._lighting()
        if checked:
            lp.gtao = self._gtao_directions.value()
        else:
            lp.gtao = 0
        self._redraw()

    def _on_gtao_param(self, _=None):
        lp = self._lighting()
        if self._gtao_enable.isChecked():
            lp.gtao = self._gtao_directions.value()
        lp.gtao_radius = self._gtao_radius_spin.value()
        lp.gtao_intensity = self._gtao_intensity_spin.value()
        lp.gtao_steps = self._gtao_steps.value()
        lp.gtao_bias_frac = self._gtao_bias_spin.value()
        self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self):
        self.session.main_view.redraw_needed = True


_dialog = None

def show(session):
    global _dialog
    if _dialog is not None:
        _dialog.close()
    mw = session.ui.main_window if hasattr(session, 'ui') else None
    _dialog = AOTuner(session, parent=mw)
    _dialog.show()
    return _dialog
