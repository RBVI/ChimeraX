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

keep_alive = []
def register_html_image_identifier(qdoc, uri, image):
  qi = _qt_image(image)
  qdoc.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(uri), qi)
  # TODO: QImage will be deleted unless reference kept.
  if not hasattr(qdoc, 'keep_images_alive'):
    qdoc.keep_images_alive = []
  qdoc.keep_images_alive.append(qi)
  global keep_alive
  keep_alive.append(qi)

def _qt_image(pil_image):
  from numpy import asarray, empty, uint32
  rgb = asarray(pil_image)
  h,w = rgb.shape[:2]
  rgba = empty((h,w), uint32)
  rgba[:,:] = rgb[:,:,0]
  rgba <<= 8
  rgba[:,:] += rgb[:,:,1]
  rgba <<= 8
  rgba[:,:] += rgb[:,:,2]
  qi = QtGui.QImage(rgba, w, h, QtGui.QImage.Format_RGB32)
# The following gives skewed image as if line padding is done.
#  qi = QtGui.QImage(rgb.reshape((w*h*3,)), w, h, QtGui.QImage.Format_RGB888)
  return qi
