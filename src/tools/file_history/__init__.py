# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    return get_singleton(session, create=True)

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from chimerax.core import tools
    from .gui import FilePanel
    return tools.get_singleton(session, FilePanel, 'file history', create=create)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'FilePanel':
        from . import gui
        return gui.FilePanel
    return None

# TODO: File history should be recorded even when the GUI is not shown.
# TODO: Should not record file history for data opened in scripts.
class FileHistory:
    def __init__(self, session):

        self.session = session

        from chimerax.core.history import ObjectCache
        self._file_cache = c = ObjectCache(session, 'file_history')
        fh = c.load()
        self._files = {} if fh is None else fh	# Maps file path to [image, time]
        self._save_files = False
        self._need_thumbnails = []
        self._thumbnail_size = (128,128)

        t = session.triggers
        t.add_trigger('file history changed')
        t.add_handler('file opened', self.file_opened_cb)

        import atexit
        atexit.register(self.quit_cb)

    def files(self):
        return self._files

    def file_opened_cb(self, name, data):
        file_path, models = data
        f = self._files
        from time import time
        t = time()
        if file_path in f:
            f[file_path][1] = t
        else:
            f[file_path] = [None, t]
        self._save_files = True

        # Delay capturing thumbnails until after models added to session.
        # Smart display style for molecules is only done after model added to session.
        self._need_thumbnails.append((file_path, models))
        t = self.session.triggers
        t.add_handler('frame drawn', self.capture_thumbnails_cb)

    def capture_thumbnails_cb(self, name, data):
        f = self._files
        ses = self.session
        mset = set(ses.models.list())
        pmlist = self._need_thumbnails
        self._need_thumbnails = []
        for path, models in pmlist:
            mopen = [m for m in models if m in mset]
            if mopen:
                f[path][0] = png_image(ses, models, self._thumbnail_size)
        ses.triggers.activate_trigger('file history changed', f)
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def quit_cb(self):
        if self._save_files:
            self._file_cache.save(self._files)

def file_history(session):
    fh = getattr(session, 'file_history', None)
    if fh is None:
        session.file_history = fh = FileHistory(session)
    return fh.files()

def png_image(session, models, size):
    if len(models) == 0:
        return None
    width, height = size
    from chimerax.core.graphics import camera
    c = camera.camera_framing_drawings(models)
    v = session.main_view
    image = v.image(width, height, camera = c, drawings = models)
    import io
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    png_data = img_io.getvalue()
    import codecs
    png_base64 = codecs.encode(png_data, 'base64').decode('utf-8')
    return png_base64
