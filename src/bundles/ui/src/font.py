# vim: set expandtab shiftwidth=4 softtabstop=4:

def _start_font_cache(ui):
    """Warm macOS's font database while the main thread initializes bundles.

    QFontMetrics is reentrant, but PySide's height() wrapper retains the GIL.
    Call the same public Qt method through CDLL so Python startup can proceed.
    Resolve through the loaded binding's own Qt dependency on macOS. If the
    native symbol is unavailable, ordinary layout initializes fonts instead.
    """
    import sys
    if sys.platform != 'darwin':
        return None
    from Qt import using_pyside6
    if not using_pyside6:
        return None

    import ctypes
    from threading import Thread
    from PySide6 import QtGui as native_gui
    from shiboken6 import getCppPointer
    from Qt.QtGui import QFontMetrics

    try:
        library = ctypes.CDLL(native_gui.__file__)
        height = library._ZNK12QFontMetrics6heightEv
        height.argtypes = [ctypes.c_void_p]
        height.restype = ctypes.c_int
        metrics = QFontMetrics(ui.font())
        pointer = getCppPointer(metrics)[0]
    except (OSError, AttributeError, TypeError):
        return None

    def warm(metrics=metrics):
        # Retain the value object while native code runs without the GIL.
        try:
            height(pointer)
        except Exception:
            # Normal layout initializes fonts if the warmup cannot finish.
            pass

    worker = Thread(target=warm, name='ChimeraX font cache')
    try:
        worker.start()
    except RuntimeError:
        return None
    return worker


def shrink_font(widget, fraction=0.75, slant=None):
    font = widget.font()
    from Qt.QtGui import QFont
    new_font = QFont(font)
    if slant:
        new_font.setItalic(True)
    new_font.setPointSize(int(fraction * font.pointSize()))
    widget.setFont(new_font)

def set_line_edit_width(line_edit, num_chars):
    le = line_edit
    fm = le.fontMetrics()
    tm = le.textMargins()
    cm = le.contentsMargins()
    w = num_chars * fm.averageCharWidth() + tm.left() + tm.right() + cm.left() + cm.right() + 8
    le.setMaximumWidth(w)
