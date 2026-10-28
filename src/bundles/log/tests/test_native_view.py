"""Specialized Qt helpers must still work when loaded by their first message."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def view():
    from Qt.QtWidgets import QApplication
    from chimerax.log.view import LogView

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    widget = LogView(None, link_cb=lambda url: None, command_color_cb=lambda: 'lightgray')
    yield widget
    widget.close()
    widget.deleteLater()


def test_first_disclosure_and_color_swatch_render(view):
    html = ('<details open><summary>Colors</summary>'
        '<div style="display:inline-block;width:12px;height:12px;'
        'background-color:red;border:1px solid black"></div> red</details>')
    view.set_content(html, [0], 0)
    assert 'Colors' in view.document().toPlainText()
    assert 'red' in view.document().toPlainText()
    assert view._swatches
    assert '<img ' in next(iter(view._swatches.values()))


def test_first_remote_image_reply_updates_document_and_releases_reply(view):
    from Qt.QtCore import QBuffer, QByteArray, QUrl
    from Qt.QtGui import QImage, QTextDocument
    from Qt.QtNetwork import QNetworkReply

    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(0xff0000)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, 'PNG')
    released = []
    reply = SimpleNamespace(error=lambda: QNetworkReply.NetworkError.NoError,
        readAll=lambda: data, deleteLater=lambda: released.append(True))
    url = QUrl('https://example.invalid/swatch.png')
    view._image_fetched(reply, url)
    loaded = view.document().resource(QTextDocument.ResourceType.ImageResource.value, url)
    assert not loaded.isNull()
    assert loaded.size() == image.size()
    assert released == [True]


def test_user_agent_function_is_called_on_first_image_fetch(monkeypatch):
    from Qt.QtWidgets import QApplication
    import Qt.QtNetwork as QtNetwork
    from Qt.QtCore import QUrl
    from chimerax.log.view import LogView

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    calls = []
    headers = []
    class Request:
        KnownHeaders = SimpleNamespace(UserAgentHeader='agent')
        def __init__(self, url):
            pass
        def setHeader(self, header, value):
            headers.append(value)
    class Reply:
        finished = SimpleNamespace(connect=lambda cb: None)
    monkeypatch.setattr(QtNetwork, 'QNetworkRequest', Request)
    widget = LogView(None, link_cb=lambda url: None, command_color_cb=lambda: 'lightgray',
                     user_agent=lambda: calls.append(True) or 'ChimeraX/test')
    widget._network = SimpleNamespace(get=lambda request: Reply())
    try:
        assert calls == []
        widget._fetch_image(QUrl('https://example.invalid/a.png'))
        widget._fetch_image(QUrl('https://example.invalid/b.png'))
        assert calls == [True]
        assert headers == ['ChimeraX/test', 'ChimeraX/test']
    finally:
        widget.close()
        widget.deleteLater()
