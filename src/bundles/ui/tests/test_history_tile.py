"""Painted history tiles retain logical sizing, captions, and selection clicks."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def tile():
    from Qt.QtGui import QPixmap
    from Qt.QtWidgets import QApplication
    from chimerax.ui.file_history import Tile

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    pixmap = QPixmap(256, 128)
    pixmap.fill()
    pixmap.setDevicePixelRatio(2)
    clicks = []
    history = SimpleNamespace(thumbnail=lambda *args: pixmap,
        display_name=lambda fs: fs.path, is_selected=lambda fs: False,
        tile_clicked=lambda tile, modifiers: clicks.append((tile.file_spec, modifiers)))
    spec = SimpleNamespace(path='a long filename with <angle brackets> & \N{GREEK SMALL LETTER ALPHA}',
                           database=None)
    widget = Tile(history, spec, {})
    yield widget, clicks
    widget.close()
    widget.deleteLater()


def test_caption_is_accessible_and_fits_with_retina_thumbnail(tile):
    from Qt.QtCore import QSize

    widget, clicks = tile
    hint = widget.sizeHint()
    fm = widget.fontMetrics()
    assert widget.accessibleName() == widget.file_spec.path
    assert widget.toolTip() == widget.file_spec.path
    assert hint.width() >= fm.horizontalAdvance(widget.file_spec.path) + 6
    assert hint.height() == 64 + fm.height() + 8
    assert widget.minimumSizeHint() == hint
    widget.resize(hint)
    # Exercise painting of a caption containing characters that QLabel could
    # otherwise interpret as HTML, together with a Retina thumbnail.
    assert widget.grab().size() != QSize()


def test_click_keeps_selection_modifiers_and_ignores_release_outside(tile):
    from Qt.QtCore import QPointF, Qt

    widget, clicks = tile
    widget.resize(widget.sizeHint())
    modifiers = Qt.KeyboardModifier.ControlModifier
    event = SimpleNamespace(button=lambda: Qt.MouseButton.LeftButton,
        position=lambda: QPointF(10, 10), modifiers=lambda: modifiers)
    widget.mousePressEvent(event)
    widget.mouseReleaseEvent(event)
    assert clicks == [(widget.file_spec, modifiers)]
    widget.mousePressEvent(event)
    event.position = lambda: QPointF(-10, -10)
    widget.mouseReleaseEvent(event)
    assert len(clicks) == 1
