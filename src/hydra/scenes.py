#
# Record and restore scenes, ie where the camera is positioned. where the models are positioned, which models
# are shown, what display styles, colors and transparencies.  This much like session saving but scenes don't
# create or delete objects, they just change the view of an existing set of objects.  Also scenes are remembered
# in memory and are included in session files.
#

# -----------------------------------------------------------------------------
# Add, show, and remove scenes.
#
def scene_command(cmdname, args):

    from .ui.commands import string_arg, int_arg, perform_operation
    ops = {
        'add': (add_scene,
                (),
                (),
                (('id', int_arg),
                 ('description', string_arg),)),
        'show': (show_scene,
                 (('id', int_arg),),
                 (),
                 ()),
        'delete': (delete_scene,
                 (('id', int_arg),),
                 (),
                 ()),
    }
    perform_operation(cmdname, args, ops)

scenes = []
def add_scene(id = None, description = None):
    global scenes
    if id is None:
        id = len(scenes) + 1
    else:
        delete_scene(id)
    scenes.append(Scene(id, description))
    show_thumbnails()
def show_scene(id):
    global scenes
    for s in scenes:
        if s.id == id:
            s.show()
            return
    from .ui import gui
    gui.show_status('No scene with id %d' % id)
def delete_scene(id):
    global scenes
    scenes = [s for s in scenes if s.id != id]
    show_thumbnails()

class Scene:

    def __init__(self, id, description):
        self.id = id
        self.description = description
        from .ui.gui import main_window
        v = main_window.view
        self.camera_view = v.camera_view
        self.field_of_view = v.field_of_view
        self.near_far_clip = v.near_far_clip
        self.center_of_rotation = v.center_of_rotation

        w = h = 128
        self.image = i = v.image((w,h))         # QImage

        from .molecule.mol_session import molecule_state
        self.molecule_states = tuple(molecule_state(m) for m in v.molecules())

        from .VolumeViewer import session
        self.map_states = session.map_states()

    def __delete__(self):
        if not hasattr(self, '_image_path'):
            import os
            try:
                os.remove(self._image_path)
            except:
                pass

    def show(self):
        from .ui.gui import main_window
        v = main_window.view
        v.set_camera_view(self.camera_view)
        v.field_of_view = self.field_of_view
        v.near_far_clip = self.near_far_clip
        v.center_of_rotation = self.center_of_rotation
        
        from .molecule.mol_session import set_molecule_state
        mids = dict((m.id, m) for m in v.molecules())
        for ms in self.molecule_states:
            m = mids.get(ms['id'])
            if m:
                set_molecule_state(m, ms)

        from .VolumeViewer import session
        session.restore_map_states(self.map_states)

        msg = 'Showing scene "%s"' % self.description if self.description else 'Showing scene %d' % self.id
        from .ui import gui        
        gui.show_status(msg)

    def image_path(self, iformat = 'JPG'):
        if not hasattr(self, '_image_path'):
            import tempfile, os
            f, ipath = tempfile.mkstemp(suffix = '.' + iformat.lower())
            os.close(f)
            self._image_path = ipath
            self.image.save(self._image_path, iformat)
        return self._image_path

    view_attributes = ('camera_view', 'field_of_view', 'near_far_clip', 'center_of_rotation')

    def scene_state(self):

        vstate = dict((attr, getattr(self,attr)) for attr in self.view_attributes)
        vstate['camera_view'] = self.camera_view.matrix
        from .file_io.SessionUtil import objecttree
        maps = objecttree.instance_tree_to_basic_tree(self.map_states)
        s = {
            'id': self.id,
            'description': self.description,
            'image': image_as_string(self.image),
            'view': vstate,
            'molecules': self.molecule_states,
            'maps': maps,
         }
        return s

    def set_state(self, scene_state):

        s = scene_state
        self.id = s['id']
        self.description = s['description']
        self.image = string_to_image(s['image'])
        va = s['view']
        for attr in self.view_attributes:
            setattr(self, attr, va[attr])
        from .geometry.place import Place
        self.camera_view = Place(va['camera_view'])
        self.molecule_states = s['molecules']
        maps = s['maps']
        from .VolumeViewer import session
        map_states = session.volume_manager_state_from_basic_tree(maps)
        self.map_states = map_states

def image_as_string(qimage, iformat = 'JPG'):

    from .ui.qt import QtCore
    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.WriteOnly)
    qimage.save(buf, iformat)
    i = ba.data()
    import base64
    s = base64.b64encode(i)
    return s

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

def scene_from_state(scene_state):
    st = scene_state
    s = Scene(st['id'], st['description'])
    s.set_state(st)
    return s

scene_thumbs = None
def show_thumbnails(toggle = False):
    global scene_thumbs, scenes
    if scene_thumbs is None:
        scene_thumbs = Scene_Thumbnails()
    if toggle and scene_thumbs.shown():
        scene_thumbs.hide()
    else:
        scene_thumbs.show(scenes)

def hide_thumbnails():
    global scene_thumbs
    scene_thumbs.hide()

class Scene_Thumbnails:

    def __init__(self):

        from .ui.gui import main_window
        from .ui.qt import QtWidgets
        self.dock_widget = dw = QtWidgets.QDockWidget('Scenes', main_window)
        dw.setTitleBarWidget(QtWidgets.QWidget(dw))   # No title bar
        dw.setFeatures(dw.NoDockWidgetFeatures)       # No close button

        self.text = e = QtWidgets.QTextBrowser(dw)          # Handle clicks on anchors
        dw.setWidget(e)
        dw.setVisible(False)

        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)

    def show(self, scenes):
        self.html = html = scene_thumbnails_html(scenes)
        self.text.setHtml(html)

        dw = self.dock_widget

        from .ui.qt import QtCore
        from .ui.gui import main_window
        main_window.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def shown(self):
        return self.dock_widget.isVisible()

    def hide(self):
        from .ui.gui import main_window
        main_window.removeDockWidget(self.dock_widget)

    def anchor_callback(self, qurl):
        url = qurl.toString()
        id = int(url)
        show_scene(id)
        self.text.setHtml(self.html)

def scene_thumbnails_html(scenes):

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
      lines.append('<td valign=bottom><a href="%d"><img src="%s" width=%d height=%d></a>'
                   % (s.id,s.image_path(), w, h))
  lines.append('<tr>')
  for s in scenes:
      lines.append('<td><a href="%d"><center>%d</center></a>' % (s.id, s.id))
  lines.extend(['</table>', '</body>', '</html>'])
  html = '\n'.join(lines)
  return html

def save_scenes(f, viewer):

    global scenes
    if len(scenes) == 0:
        return

    states = tuple(s.scene_state() for s in scenes)
    
    f.write("'scenes':(\n")
    from .file_io.SessionUtil import objecttree
    objecttree.write_basic_tree(states, f, indent = ' ')
    f.write('),\n')

def restore_scenes(d, viewer):

    scene_states = d.get('scenes', [])
    global scenes
    scenes = [scene_from_state(s) for s in scene_states]
    if len(scenes) == 0:
        hide_thumbnails()
    else:
        show_thumbnails()
