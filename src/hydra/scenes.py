#
# Record and restore scenes, ie where the camera is positioned. where the models are positioned, which models
# are shown, what display styles, colors and transparencies.  This much like session saving but scenes don't
# create or delete objects, they just change the view of an existing set of objects.  Also scenes are remembered
# in memory and are included in session files.
#

# -----------------------------------------------------------------------------
# Add, show, and remove scenes.
#
def scene_command(cmdname, args, session):

    from .ui.commands import string_arg, int_arg, perform_operation
    ops = {
        'add': (session.scenes.add_scene,
                (),
                (('id', int_arg),),
                (('description', string_arg),)),
        'show': (session.scenes.show_scene,
                 (('id', int_arg),),
                 (),
                 ()),
        'delete': (session.scenes.delete_scene,
                 (('id', string_arg),),
                 (),
                 ()),
    }
    perform_operation(cmdname, args, ops, session)

class Scenes:

    def __init__(self, session):
        self.session = session
        self.scenes = []
        self.scene_thumbs = None

    def add_scene(self, id = None, description = None):
        sl = self.scenes
        if id is None:
            id = max(s.id for s in sl)+1 if sl else 1
        else:
            self.delete_scene(id)
        sl.append(Scene(id, description, self.session))
        self.show_thumbnails()

    def show_scene(self, id):
        for s in self.scenes:
            if s.id == id:
                s.show()
                return
        self.session.show_status('No scene with id %d' % id)

    def delete_scene(self, id):
        if id == 'all':
            self.scenes = []
        elif isinstance(id, str):
            try:
                ids = set(int(i) for i in id.split(','))
            except:
                from .ui.commands import CommandError
                raise CommandError('Scene ids must be integers, got "%s"' % id)
            self.scenes = [s for s in self.scenes if not s.id in ids]
        else:
            self.scenes = [s for s in self.scenes if s.id != id]
        self.show_thumbnails()

    def delete_all_scenes(self):
        if self.scenes:
            self.scenes = []
            self.hide_thumbnails()

    def show_thumbnails(self, toggle = False):
        st = self.scene_thumbs
        if st is None:
            self.scene_thumbs = st = Scene_Thumbnails(self.session)
        if toggle and st.shown():
            st.hide()
        else:
            st.show(self.scenes)

    def hide_thumbnails(self):
        if self.scene_thumbs:
            self.scene_thumbs.hide()

class Scene:

    def __init__(self, id, description, session = None):
        self.id = id
        self.description = description
        self.session = session
        self.cross_fade_frames = 30
        self.thumbnail_size = (128,128)
        self.uri = None         # HTML reference to image.

        if session is None:
            self.image = None
            self.state = None
        else:
            w, h = self.thumbnail_size
            self.image = i = session.view.image(w,h)         # QImage

            from .file_io import session_file
            self.state = session_file.scene_state(session)

    def show(self):
        s = self.session
        if self.cross_fade_frames:
            from .ui.crossfade import Cross_Fade
            Cross_Fade(s.view, self.cross_fade_frames)

        # Hide all models so models that did not exist in scene are hidden.
        for m in s.model_list():
            m.displayed = False

        from .file_io import session_file
        session_file.restore_scene(self.state, s)
        
        msg = 'Showing scene "%s"' % self.description if self.description else 'Showing scene %d' % self.id
        s.show_status(msg)

    def image_uri(self, qdoc):

        if self.uri is None and not self.image is None:
            self.uri = uri = "file://image%d" % self.id
            from .ui.qt import QtGui, QtCore
            qdoc.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(uri), self.image)
        return self.uri

    def scene_state(self):

        s = {
            'id': self.id,
            'description': self.description,
            'image': image_as_string(self.image),
            'state': self.state,
         }
        return s

    def set_state(self, scene_state, session):

        self.session = session
        s = scene_state
        self.id = s['id']
        self.description = s['description']
        self.image = string_to_image(s['image'])
        self.state = s['state']

def image_as_string(qimage, iformat = 'JPG'):

    i = image_as_bytes(qimage, iformat)
    import base64
    s = base64.b64encode(i)
    return s

def image_as_bytes(qimage, iformat = 'JPG'):

    from .ui.qt import QtCore
    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.WriteOnly)
    qimage.save(buf, iformat)
    i = ba.data()
    return i

def string_to_image(s, iformat = 'JPG'):

    import base64
    i = base64.b64decode(s)
    from .ui.qt import QtCore, QtGui
    ba = QtCore.QByteArray(i)
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.ReadOnly)
    qi = QtGui.QImage()
    qi.load(buf, iformat)
    return qi

def scene_from_state(scene_state, session):
    st = scene_state
    s = Scene(st['id'], st['description'])
    s.set_state(st, session)
    return s

class Scene_Thumbnails:

    def __init__(self, session):

        self.session = session

        from .ui.qt import QtWidgets, QtCore
        self.dock_widget = dw = QtWidgets.QDockWidget('Scenes', session.main_window)
        dw.setTitleBarWidget(QtWidgets.QWidget(dw))   # No title bar
        dw.setFeatures(dw.NoDockWidgetFeatures)       # No close button

        class Thumbnail_Viewer(QtWidgets.QTextBrowser):
            height = 140
            def sizeHint(self):
                return QtCore.QSize(600,self.height)
        self.text = e = Thumbnail_Viewer(dw)
        e.setOpenLinks(False)
#        self.text = e = QtWidgets.QTextBrowser(dw)
        dw.setWidget(e)
        dw.setVisible(False)

        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)          # Handle clicks on anchors

    def show(self, scenes):
        self.set_height(scenes = scenes)
        self.html = html = scene_thumbnails_html(scenes, self.text.document())
        self.text.setHtml(html)

        from .ui.qt import QtCore
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
      w,h = i.width(), i.height()
      lines.append('<td width=%d valign=bottom><a href="%d"><img src="%s" width=%d height=%d></a>'
                   % (w+10, s.id, s.image_uri(qdoc), w, h))

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

def scene_state(session):

    slist = session.scenes.scenes
    if len(slist) == 0:
        return None

    s = tuple(s.scene_state() for s in slist)
    return s

def restore_scenes(scene_states, session):

    scenes = session.scenes
    scenes.scenes = sl = [scene_from_state(s, session) for s in scene_states]
    if len(sl) == 0:
        scenes.hide_thumbnails()
    else:
        scenes.show_thumbnails()
