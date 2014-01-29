from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtCore import Qt

# TODO: Monkey business for Qt to find cocoa platform plugin.
#       Can a relative path be built into Qt during compile?
from os.path import join, dirname
import PyQt5
plugins_path = join(dirname(PyQt5.__file__), 'plugins')
QtCore.QCoreApplication.addLibraryPath(plugins_path)


def draw_image_text(qi, text, color = (255,255,255), bgcolor = None,
                    font_name = 'Helvetica', font_size = 40):
  p = QtGui.QPainter(qi)
  w,h = qi.width(), qi.height()

  while True and font_size > 6:
    f = QtGui.QFont(font_name, font_size)
    p.setFont(f)
    fm = p.fontMetrics()
    wt = fm.width(text)
    if wt <= w:
      break
    font_size = int(font_size * (w/wt))

  fh = fm.height()
  r = QtCore.QRect(0,h-fh,w,fh)
  if not bgcolor is None:
    p.fillRect(r, QtGui.QColor(*bgcolor))
  p.setPen(QtGui.QColor(*color))
  p.drawText(r, QtCore.Qt.AlignCenter, text)
