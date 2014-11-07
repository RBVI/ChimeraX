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

    from .commands.parse import string_arg, int_arg, perform_operation
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
                from .commands.parse import CommandError
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
            from . import ui
            self.scene_thumbs = st = ui.Scene_Thumbnails(self.session)
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

        if session is None:
            self.image = None
            self.state = None
        else:
            w, h = self.thumbnail_size
            self.image = i = session.view.image(w,h)         # QImage

            from .files import session_file
            self.state = session_file.scene_state(session)

    def show(self):
        s = self.session
        if self.cross_fade_frames:
            from .graphics import Cross_Fade
            Cross_Fade(s.view, self.cross_fade_frames)

        # Hide all models so models that did not exist in scene are hidden.
        for m in s.model_list():
            m.display = False

        from .files import session_file
        session_file.restore_scene(self.state, s)
        
        msg = 'Showing scene "%s"' % self.description if self.description else 'Showing scene %d' % self.id
        s.show_status(msg)

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

def image_as_string(image, iformat = 'JPEG'):

    i = encode_image(image, iformat)
    import base64
    s = base64.b64encode(i)
    return s

def encode_image(image, iformat = 'JPEG'):

    import io
    f = io.BytesIO()
    image.save(f, iformat)
    s = f.getvalue()
    return s

def string_to_image(s, iformat = 'JPEG'):

    import base64
    i = base64.b64decode(s)
    import io
    s = io.BytesIO(i)
    from PIL import Image
    i = Image.open(s)
    return i

def scene_from_state(scene_state, session):
    st = scene_state
    s = Scene(st['id'], st['description'])
    s.set_state(st, session)
    return s

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
