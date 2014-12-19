from .qt import QtWidgets, QtCore, QtGui

class Scene_Thumbnails:

    def __init__(self, session):

        self.session = session

        self.dock_widget = dw = QtWidgets.QDockWidget('Scenes', session.main_window)
        dw.setTitleBarWidget(QtWidgets.QWidget(dw))   # No title bar
        dw.setFeatures(dw.NoDockWidgetFeatures)       # No close button

        class Thumbnail_Viewer(QtWidgets.QTextBrowser):
            height = 140
            close_button = None
            def sizeHint(self):
                return QtCore.QSize(600,self.height)
            def resizeEvent(self, e):
                QtWidgets.QTextBrowser.resizeEvent(self, e)
                c = self.close_button
                if c:
                    c.move(e.size().width()-c.width()-5,5)

        self.text = e = Thumbnail_Viewer(dw)
        e.setOpenLinks(False)

        e.close_button = ct = QtWidgets.QPushButton('X', e)
        ct.setStyleSheet("padding: 1px; min-width: 1em")
        ct.adjustSize()
        ct.clicked.connect(lambda e: self.hide())

        dw.setWidget(e)
        dw.setVisible(False)

        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)          # Handle clicks on anchors

    def show(self, scenes):
        self.set_height(scenes = scenes)
        self.html = html = scene_thumbnails_html(scenes, self.text.document())
        self.text.setHtml(html)

        dw = self.dock_widget
        mw = self.session.main_window
        mw.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def shown(self):
        return self.dock_widget.isVisible()

    def hide(self):
        mw = self.session.main_window
        mw.removeDockWidget(self.dock_widget)

    def anchor_callback(self, qurl):
        url = qurl.toString()
        id = int(url)
        self.session.scenes.show_scene(id)

    def set_height(self, h = None, scenes = []):
        if h is None:
            h = 220 if [s for s in scenes if s.description] else 140
        self.text.height = h
#        self.text.adjustSize()
#        self.dock_widget.adjustSize()

def scene_thumbnails_html(scenes, qdoc):

  from os.path import basename, splitext
  lines = ['<html>', '<head>', '<style>',
           'body { background-color: black; }',
           'a { text-decoration: none; }',      # No underlining of links
           'a:link { color: #FFFFFF; }',        # Link text color white.
           'table { float:left; }',             # Multiple image/caption tables per row.
           'td { font-size:large; }',
#           'td { text-align:center; }',        # Does not work in Qt 5.0.2
           '</style>', '</head>', '<body>',
           '<table style="float:left;">', '<tr>',
           ]
  for s in scenes:
      i = s.image
      w,h = i.size
      lines.append('<td width=%d valign=bottom><a href="%d"><img src="%s" width=%d height=%d></a>'
                   % (w+10, s.id, image_uri(s, qdoc), w, h))

  lines.append('<tr>')
  for s in scenes:
      lines.append('<td><a href="%d"><center>%d</center></a>' % (s.id, s.id))

  if [s for s in scenes if s.description]:
      lines.append('<tr>')
      import cgi
      for s in scenes:
          line = ('<td><a href="%d">%s</a>' % (s.id, cgi.escape(s.description))) if s.description else '<td>'
          lines.append(line)
  lines.extend(['</table>', '</body>', '</html>'])
  html = '\n'.join(lines)
  return html

def image_uri(scene, qdoc):

    if hasattr(scene, 'uri'):
        return scene.uri

    if scene.image is None:
        return None

    scene.uri = uri = "file://image%d" % scene.id
    from . import qt
    qt.register_html_image_identifier(qdoc, uri, scene.image)

    return uri
