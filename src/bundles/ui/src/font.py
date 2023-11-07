# vim: set expandtab shiftwidth=4 softtabstop=4:

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
