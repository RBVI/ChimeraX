#
# Remember opened and saved files so they can be easily opened in a later session.
#

# TODO: Should not record file history for data opened in scripts.
class FileHistory:
    def __init__(self, session):

        self.session = session
        self.version = 1		# In case cache file changes format.
        self.max_files = 200

        self._save_files = False
        self._need_thumbnails = []
        self._thumbnail_size = (128,128)

        from .history import ObjectCache
        self._file_cache = ObjectCache(session, 'file_history')
        self._files = self.load_history()	# Map file path to FileSpec
        
        session.triggers.add_trigger('file history changed')

        import atexit
        atexit.register(self.quit_cb)

    @property
    def files(self):
        flist = list(self._files.values())
        flist.sort(key = lambda f: f.access_time)
        return flist

    def remember_file(self, path, format, models, database = None, file_saved = False):
        f = self._files
        from os.path import abspath
        apath = abspath(path) if database is None else path
        fs = f.get(apath, None)
        if fs:
            fs.set_access_time()
        else:
            f[apath] = fs = FileSpec(apath, format, database = database)
        if fs.image is None or file_saved:
            self._need_thumbnails.append((apath, models))
            if file_saved:
                self.capture_thumbnails_cb()
            else:
                # Delay capturing thumbnails until after models added to session.
                # Smart display style for molecules is only done after model added to session.
                t = self.session.triggers
                t.add_handler('frame drawn', lambda *args, s=self: s.capture_thumbnails_cb)
        self._save_files = True

    def capture_thumbnails_cb(self):
        f = self._files
        ses = self.session
        mset = set(ses.models.list())
        pmlist = self._need_thumbnails
        self._need_thumbnails = []
        for path, models in pmlist:
            if path in f:
                if models != 'all models':
                    models = [m for m in models if m in mset]
                if models:
                    f[path].capture_image(models, ses)
        ses.triggers.activate_trigger('file history changed', f)
        from .triggerset import DEREGISTER
        return DEREGISTER

    def remove_missing_files(self):
        f = self._files
        from os.path import isfile
        from glob import glob
        remove = [path for path, fspec in f.items()
                  if fspec.database is None and not isfile(path) and not glob(path)]
        if remove:
            for p in remove:
                del f[p]
            self._save_files = True
            self.session.triggers.activate_trigger('file history changed', f)
            
    def quit_cb(self):
        if self._save_files:
            self.save_history()

    def load_history(self):
        fc = self._file_cache.load()
        fmap = {}
        if fc is not None:
            for f in fc['files']:
                fs = FileSpec.from_state(f)
                fmap[fs.path] = fs
        return fmap

    def save_history(self):
        data = {
            'version': self.version,
            'files': [f.state() for f in self.files[-self.max_files:]]
        }
        self._file_cache.save(data)
        self._save_files = False

class FileSpec:
    def __init__(self, path, format, database = None):
        self.path = path
        self.format = format
        self.database = database
        self.access_time = None
        self.image = None	# JPEG encoded as base64 string
        self.set_access_time()

    def set_access_time(self):
        from time import time
        self.access_time = time()

    def short_name(self):
        from os.path import basename,splitext
        n = basename(self.path)
        if n.endswith('.cif') or n.endswith('.pdb') or n.endswith('.map'):
            n = splitext(n)[0]
        return n

    def capture_image(self, models, session, size = (128, 128)):
        self.image = models_image(session, models, size)

    def open_command(self):
        p = self.path
        if ' ' in p:
            p = '"%s"' % p 	# Quote path
        cmd = 'open %s' % p
        if self.format:
            cmd += ' format %s' % self.format
        if self.database:
            cmd += ' fromDatabase %s' % self.database
        return cmd

    def state(self):
        return {k:getattr(self,k) for k in ('path', 'format', 'database', 'access_time', 'image')}

    @classmethod
    def from_state(self, state):
        f = FileSpec(state['path'], state['format'], database = state['database'])
        for k in ('access_time', 'image'):
            setattr(f, k, state[k])
        return f

def file_history(session):
    fh = getattr(session, 'file_history', None)
    if fh is None:
        session.file_history = fh = FileHistory(session)
    return fh

def remember_file(session, filename, format, models, database = None, file_saved = False):
    if session.in_script:
        return		# Don't remember files opened by scripts
    h = file_history(session)
    h.remember_file(filename, format, models, database = database, file_saved = file_saved)

def models_image(session, models, size, format = 'JPEG'):
    v = session.main_view
    width, height = size
    if models == 'all models':
        image = v.image(width, height)
    else:
        from .graphics import camera
        c = camera.camera_framing_drawings(models)
        image = v.image(width, height, camera = c, drawings = models)
    import io
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    image_bytes = img_io.getvalue()
    import codecs
    image_base64 = codecs.encode(image_bytes, 'base64').decode('utf-8')
    return image_base64
