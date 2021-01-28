# vim: set expandtab shiftwidth=4 softtabstop=4:

def shrink_font(widget, fraction=0.75, slant=None):
    font = widget.font()
    from Qt.QtGui import QFont
    new_font = QFont(font)
    if slant:
        new_font.setItalic(True)
    new_font.setPointSize(int(fraction * font.pointSize()))
    widget.setFont(new_font)
