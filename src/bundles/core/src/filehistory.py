# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Remember opened and saved files so they can be easily opened in a later session.
#

# TODO: Should not record file history for data opened in scripts.
class FileHistory:
    def __init__(self, session):

        self.session = session
        self.version = 1        # In case cache file changes format.
        self.max_files = 200

        self._save_files = False
        self._need_thumbnails = []
        self._thumbnail_size = (128,128)

        from .history import ObjectHistory
        self._file_cache = ObjectHistory('file_history')
        self._files = self.load_history()    # Map (file path, database) to FileSpec
        
        session.triggers.add_trigger('file history changed')

        import atexit
        atexit.register(self.quit_cb)

    @property
    def files(self):
        flist = list(self._files.values())
        flist.sort(key = lambda f: f.access_time)
        return flist

    def remember_file(self, path, format, models, database = None, file_saved = False,
                      open_options = {}):
        if not _supported_option_value_types(open_options):
            return
        f = self._files
        from os.path import abspath
        apath = abspath(path) if database is None else path
        fs = f.get((apath,database))
        if fs:
            fs.set_access_time()
            fchange = (format != fs.format)
            fs.format = format
            optchange = fs.set_open_options(open_options)
            if fchange or optchange:
                fs.image = None
                self.history_changed()
        else:
            f[(apath,database)] = fs = FileSpec(apath, format, database = database,
                                                open_options = open_options)
        has_graphics = self.session.main_view.render is not None
        if has_graphics and (fs.image is None or file_saved):
            self._need_thumbnails.append((fs, models))
            if file_saved:
                self.capture_thumbnails_cb()
            else:
                # Delay capturing thumbnails until after models added to session.
                # Smart display style for molecules is only done after model added to session.
                t = self.session.triggers
                t.add_handler('frame drawn', lambda *args, s=self: s.capture_thumbnails_cb())
        self._save_files = True

    def capture_thumbnails_cb(self):
        ses = self.session
        mset = set(ses.models.list())
        fsmlist = self._need_thumbnails
        self._need_thumbnails = []
        for fs, models in fsmlist:
            if models != 'all models':
                models = [m for m in models if m in mset]
            if models:
                fs.capture_image(models, ses)
        self.history_changed()
        from .triggerset import DEREGISTER
        return DEREGISTER

    def remove_missing_files(self):
        f = self._files
        from os.path import isfile
        from glob import glob
        remove = [fspec for fspec in f.values()
                  if fspec.database is None and not isfile(fspec.path) and not glob(fspec.path)]
        if remove:
            for fspec in remove:
                del f[(fspec.path,fspec.database)]
            self._save_files = True
            self.history_changed()

    def clear_file_history(self):
        f = self._files
        f.clear()
        self._save_files = True
        self.history_changed()

    def history_changed(self):
        self.session.triggers.activate_trigger('file history changed', self._files)
            
    def quit_cb(self):
        if self._save_files and self.session.ui.is_gui:
            self.save_history()

    def load_history(self):
        try:
            fc = self._file_cache.load()
        except Exception as e:
            backup_path = self._file_cache.backup()
            msg = ('The history of data files opened in ChimeraX was unreadable.\n'
                   'The unreadable file has been copied to %s.\n' % backup_path +
                   'Please report this as a bug using menu Help / Report a Bug.\n\n' +
                   'The error was "%s".' % str(e))
            self.session.logger.bug(msg)
            fc = None
        fmap = {}
        if fc is not None:
            for f in fc['files']:
                fs = FileSpec.from_state(self.session, f)
                fmap[(fs.path,fs.database)] = fs
        return fmap

    def save_history(self):
        data = {
            'version': self.version,
            'files': [f.state() for f in self.files[-self.max_files:]]
        }
        self._file_cache.save(data)
        self._save_files = False

_savable_option_value_types = (bool, int, float, str)
def _supported_option_value_types(open_options):
    for k,v in open_options.items():
        if not isinstance(v, _savable_option_value_types):
            return False
    return True
        

class FileSpec:
    def __init__(self, path, format, database = None, open_options = {}):
        self.path = path
        self.format = format    # Can be None
        self.database = database
        self.access_time = None
        self.image = None    # JPEG encoded as base64 string
        self.open_options = {}
        self.set_access_time()
        self.set_open_options(open_options)    # Dictionary of open command keyword to value.

    def set_open_options(self, open_options):
        opt = {k:str(v) for k,v in open_options.items()
               if isinstance(v, _savable_option_value_types)}
        change = (opt != self.open_options)
        self.open_options = opt
        return change

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
        from chimerax.core.commands import quote_path_if_necessary
        cmd = 'open %s' % quote_path_if_necessary(self.path)
        f = self.format
        if f:
            if ' ' in f:
                f = '"%s"' % f
            cmd += ' format %s' % f
        if self.database:
            cmd += ' fromDatabase %s' % self.database
        if self.open_options:
            cmd += ' ' + ' '.join('%s %s' % (kw, quoted_string(value))
                            for kw,value in self.open_options.items())
        return cmd

    def state(self):
        return {k:getattr(self,k) for k in ('path', 'format', 'database', 'access_time', 'image', 'open_options')}

    @classmethod
    def from_state(self, session, state):
        format = state['format']
        if format is not None:
            # map old format names to those used in new open command
            try:
                format = session.data_formats[format].nicknames[0]
            except KeyError:
                pass
        f = FileSpec(state['path'], format, database = state['database'],
                     open_options = state.get('open_options', {}))
        for k in ('access_time', 'image'):
            setattr(f, k, state[k])
        return f

def quoted_string(s):
    if ' ' in s:
        return '"%s"' % s
    return s

def file_history(session):
    fh = getattr(session, 'file_history', None)
    if fh is None:
        session.file_history = fh = FileHistory(session)
    return fh

def remember_file(session, filename, format, models, database = None, file_saved = False,
                  open_options = {}):
    if session.in_script:
        return        # Don't remember files opened by scripts
    h = file_history(session)
    h.remember_file(filename, format, models, database = database, file_saved = file_saved,
                    open_options = open_options)

def models_image(session, models, size, format = 'JPEG'):
    v = session.main_view
    width, height = size
    if models == 'all models':
        image = v.image(width, height)
    else:
        from chimerax.graphics import camera
        c = camera.camera_framing_drawings(models)
        image = v.image(width, height, camera = c, drawings = models)
    import io
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    image_bytes = img_io.getvalue()
    import codecs
    image_base64 = codecs.encode(image_bytes, 'base64').decode('utf-8')
    return image_base64
