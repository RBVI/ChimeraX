"""Exercise native Qt controls against real lighting and material commands."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    from Qt.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool(app, monkeypatch, test_production_session):
    from Qt.QtWidgets import QWidget
    from chimerax.graphics import View, Drawing
    from chimerax.core.commands import run

    module = load_source("lighting_controls", Path(__file__).parents[1] / "src/tool.py")
    window = SimpleNamespace(ui_area=QWidget(), shown=True, manage=lambda: None)
    monkeypatch.setattr(module, "MainToolWindow", lambda tool: window)
    run(test_production_session, "lighting default; material default; graphics silhouettes false")
    preview_view = View(Drawing("preview"))
    monkeypatch.setattr(module, "LightingPreviewWidget", lambda parent, session, gui:
                        SimpleNamespace(widget=QWidget(parent), view=preview_view,
                                        update_arrow_directions=lambda: None,
                                        render=lambda: None, close=preview_view.delete))
    gui = module.LightingGUI(test_production_session)
    yield gui
    gui.delete()
    window.ui_area.deleteLater()


def run(tool, command):
    from chimerax.core.commands import run
    return run(tool.session, command)


def test_layout_and_default_button(tool):
    from Qt.QtWidgets import QTabWidget, QPushButton
    assert not tool.tool_window.ui_area.findChildren(QTabWidget)
    assert {tool.material_preset.itemText(i) for i in range(4)} == {
        "default", "shiny", "dull", "chimera"}
    assert not hasattr(tool, "depth_cue_start")
    run(tool, "lighting color red; material dull; graphics silhouettes true")
    button = next(b for b in tool.tool_window.ui_area.findChildren(QPushButton)
                  if b.text() == "Default lighting")
    button.click()
    v = tool.session.main_view
    assert v.lighting.key_light_color == pytest.approx((1, 1, 1))
    assert v.material.specular_reflectivity == pytest.approx(0.3)
    assert v.silhouette.enabled
    assert tool.preset_combo.currentText() == "simple"
    assert tool.material_preset.currentText() == "default"


@pytest.mark.parametrize("name,attr", [
    ("key", "key_light_color"), ("fill", "fill_light_color"),
    ("ambient", "ambient_light_color"),
])
def test_color_wells_preserve_byte_colors_and_sync_preview(tool, name, attr):
    color = np.array([128, 64, 255, 255], dtype=np.uint8)
    well = getattr(tool, name + "_color")
    well.color_changed.emit(color)
    expected = color[:3] / 255
    assert getattr(tool.session.main_view.lighting, attr) == pytest.approx(expected)
    assert getattr(tool.preview_widget.view.lighting, attr) == pytest.approx(expected)
    assert np.array_equal(well.color, color)
    assert tool.preset_combo.currentIndex() == -1
    tool.key_intensity.slider.setValue(80)
    assert np.array_equal(well.color, color)


def test_command_and_toolbar_presets_update_numbers(tool):
    run(tool, "lighting full")
    assert tool.key_intensity.value_entry.value() == pytest.approx(0.7)
    assert tool.ambient_intensity.value_entry.value() == pytest.approx(0.8)
    assert tool.preset_combo.currentText() == "full"
    run(tool, "lighting color 100,0,0")
    assert tuple(tool.key_color.color) == (255, 0, 0, 255)
    assert tool.preset_combo.currentIndex() == -1
    run(tool, "lighting default")
    assert tool.preset_combo.currentText() == "simple"


@pytest.mark.parametrize("preset,specular,exponent", [
    ("default", 0.3, 30), ("shiny", 1, 30),
    ("dull", 0, 30), ("chimera", 1, 6),
])
def test_material_presets_update_controls_and_preview(tool, preset, specular, exponent):
    run(tool, "material default")
    index = tool.material_preset.findText(preset)
    tool.material_preset.activated.emit(index)
    assert tool.reflectivity.value() == pytest.approx(specular)
    assert tool.sharpness.value() == pytest.approx(exponent)
    assert tool.preview_widget.view.material.specular_reflectivity == pytest.approx(specular)
    assert tool.material_preset.currentText() == preset
    tool.sharpness.value_entry.setValue(45)
    assert tool.material_preset.currentIndex() == -1


def test_specular_edit_leaves_diffuse_reflectivity_unchanged(tool):
    run(tool, "material reflectivity 0.63")
    tool.reflectivity.value_entry.setValue(0.42)
    tool.reflectivity.value_entry.editingFinished.emit()
    mat = tool.session.main_view.material
    assert mat.specular_reflectivity == pytest.approx(0.42)
    assert mat.diffuse_reflectivity == pytest.approx(0.63)
    assert tool.preview_widget.view.material.specular_reflectivity == pytest.approx(0.42)


def test_presets_survive_changes_to_the_other_group(tool):
    run(tool, "lighting full")
    tool.material_preset.activated.emit(tool.material_preset.findText("shiny"))
    assert tool.preset_combo.currentText() == "full"
    run(tool, "lighting color red")
    assert tool.material_preset.currentText() == "shiny"
    assert tool.preset_combo.currentIndex() == -1
    run(tool, "material specularReflectivity 0.42")
    assert tool.material_preset.currentIndex() == -1


def test_additional_material_controls_and_external_values(tool):
    run(tool, "material reflectivity 3.5 ambientReflectivity 0.62 exponent 256 meshesCastShadows true")
    assert tool._material_sliders["diffuse_reflectivity"].value_entry.value() == 3.5
    assert tool.sharpness.value_entry.value() == 256
    assert tool._material_checkboxes["meshes_cast_shadows"].isChecked()
    slider = tool._material_sliders["ambient_reflectivity"]
    slider.value_entry.setValue(0.27)
    slider.value_entry.editingFinished.emit()
    assert tool.session.main_view.material.ambient_reflectivity == pytest.approx(0.27)
    assert tool.preview_widget.view.material.ambient_reflectivity == pytest.approx(0.27)
    tool._material_checkboxes["transparent_cast_shadows"].click()
    assert tool.session.main_view.material.transparent_cast_shadows


def test_depth_cue_controls_follow_commands_and_send_colors(app, test_production_session):
    from chimerax.core.commands import run
    path = Path(__file__).parents[2] / "sideview/src/tool.py"
    module = load_source("sideview_controls", path)
    controls = module.DepthCueControls(test_production_session)
    try:
        run(test_production_session, "lighting depthCue false depthCueStart 0.25 depthCueEnd 0.85 depthCueColor red")
        assert not controls.enabled.isChecked()
        assert controls.start.value() == 0.25
        assert controls.end.value() == 0.85
        assert tuple(controls.color.color) == (255, 0, 0, 255)
        controls.enabled.click()
        controls.start.setValue(0.35)
        controls.end.setValue(0.95)
        controls.color.color_changed.emit(np.array([128, 64, 255, 255], dtype=np.uint8))
        lp = test_production_session.main_view.lighting
        assert lp.depth_cue
        assert lp.depth_cue_start == 0.35
        assert lp.depth_cue_end == 0.95
        assert lp.depth_cue_color == pytest.approx((128 / 255, 64 / 255, 1))
    finally:
        controls.remove_handler()
        controls.deleteLater()
