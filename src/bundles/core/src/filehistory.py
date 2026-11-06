# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Remember opened and saved files so they can be easily opened in a later session.
#

# TODO: Should not record file history for data opened in scripts.
class FileHistory:
    def __init__(self, session):

        self.session = session
        self.version = 1        # In case cache file changes format.
        self.sections_version = 1
        self.max_files = 200

        self._save_files = False
        self._need_thumbnails = []
        self._thumbnail_size = (128,128)

        # User defined sections that group files.  Files in a section keep their
        # position when reopened.  Files in no section are ordered by access time.
        self._sections = []    # List of HistorySection
        self._current_section = None    # Name of section new files are added to

        from .history import ObjectHistory
        self._file_cache = ObjectHistory('file_history')
        self._sections_cache = ObjectHistory('file_history_sections')
        self._files = self.load_history()    # Map (file path, database) to FileSpec

        if not session.triggers.has_trigger('file history changed'):
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
            s = self._section(self._current_section)
            if s:
                s.keys.append((apath,database))
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
            self._prune_section_keys()
            self._save_files = True
            self.history_changed()

    def clear_file_history(self):
        '''Remove all files from the history.  Sections are kept, now empty.'''
        f = self._files
        f.clear()
        self._prune_section_keys()
        self._save_files = True
        self.history_changed()

    def remove_file(self, fspec):
        self.remove_files([fspec])

    def remove_files(self, fspecs):
        f = self._files
        keys = [(fs.path, fs.database) for fs in fspecs]
        keys = [k for k in keys if k in f]
        if keys:
            for k in keys:
                del f[k]
            self._prune_section_keys()
            self._organization_changed()

    def find_file(self, path, database = None):
        return self._files.get((path, database))

    # ------------------------------------------------------------------------
    # User defined sections.
    #
    @property
    def sections(self):
        '''Names of the user defined sections in display order.'''
        return [s.name for s in self._sections]

    def section_files(self, section_name = None):
        '''
        FileSpecs in a section in user order.  If section_name is None
        return the files not in any section, most recently used first.
        '''
        if section_name is None:
            placed = self._placed_keys()
            return [fs for fs in reversed(self.files)
                    if (fs.path, fs.database) not in placed]
        s = self._section(section_name)
        if s is None:
            raise ValueError('No file history section named "%s"' % section_name)
        return [self._files[k] for k in s.keys if k in self._files]

    def file_section(self, fspec):
        '''Name of section containing the file or None.'''
        key = (fspec.path, fspec.database)
        for s in self._sections:
            if key in s.keys:
                return s.name
        return None

    def add_section(self, name, index = None):
        name = self._check_new_section_name(name)
        s = HistorySection(name)
        if index is None:
            self._sections.append(s)
        else:
            self._sections.insert(index, s)
        self._organization_changed()
        return name

    def rename_section(self, name, new_name):
        s = self._section(name)
        if s is None:
            raise ValueError('No file history section named "%s"' % name)
        if new_name.strip() == name:
            return
        s.name = self._check_new_section_name(new_name)
        if self._current_section == name:
            self._current_section = s.name
        self._organization_changed()

    def remove_section(self, name):
        '''Remove a section.  Its files go back to the unsorted files.'''
        s = self._section(name)
        if s is None:
            raise ValueError('No file history section named "%s"' % name)
        self._sections.remove(s)
        if self._current_section == name:
            self._current_section = None
        self._organization_changed()

    def move_section(self, name, index):
        s = self._section(name)
        if s is None:
            raise ValueError('No file history section named "%s"' % name)
        self._sections.remove(s)
        index = max(0, min(index, len(self._sections)))
        self._sections.insert(index, s)
        self._organization_changed()

    def move_file(self, fspec, section_name, before = None):
        self.move_files([fspec], section_name, before)

    def move_files(self, fspecs, section_name, before = None):
        '''
        Put files in a section, in the given order, placed before FileSpec
        "before", or at the end if before is None.  A section_name of None
        removes the files from their sections.
        '''
        keys = []
        for fs in fspecs:
            key = (fs.path, fs.database)
            if key not in self._files:
                raise ValueError('File %s is not in the file history' % fs.path)
            if key not in keys:
                keys.append(key)
        dest = None
        if section_name is not None:
            dest = self._section(section_name)
            if dest is None:
                raise ValueError('No file history section named "%s"' % section_name)
        # If inserting before one of the moved files use the next file that is not moved.
        bkey = None
        if dest is not None and before is not None:
            bkey = (before.path, before.database)
            if bkey in dest.keys:
                following = dest.keys[dest.keys.index(bkey):]
                bkey = next((k for k in following if k not in keys), None)
        moved = set(keys)
        for s in self._sections:
            s.keys = [k for k in s.keys if k not in moved]
        if dest is not None:
            i = dest.keys.index(bkey) if bkey in dest.keys else len(dest.keys)
            dest.keys[i:i] = keys
        self._organization_changed()

    def _get_current_section(self):
        return self._current_section
    def _set_current_section(self, name):
        if name is not None and self._section(name) is None:
            raise ValueError('No file history section named "%s"' % name)
        if name != self._current_section:
            self._current_section = name
            self._organization_changed()
    current_section = property(_get_current_section, _set_current_section,
                               doc = 'Name of the section that newly opened files are added to, or None.')

    def _section(self, name):
        if name is None:
            return None
        for s in self._sections:
            if s.name == name:
                return s
        return None

    def _check_new_section_name(self, name):
        name = name.strip()
        if not name:
            raise ValueError('File history section name cannot be empty')
        if self._section(name) is not None:
            raise ValueError('File history section "%s" already exists' % name)
        return name

    def _placed_keys(self):
        return set(k for s in self._sections for k in s.keys)

    def _prune_section_keys(self):
        f = self._files
        for s in self._sections:
            s.keys = [k for k in s.keys if k in f]

    def _organization_changed(self):
        # Save right away so organizing work is not lost if ChimeraX does not exit cleanly.
        self.save_history()
        self.history_changed()

    def history_changed(self):
        self.session.triggers.activate_trigger('file history changed', self._files)
            
    def quit_cb(self):
        if self._save_files and self.session.ui.is_gui:
            self.save_history()

    def load_history(self):
        fc = self._load_cache(self._file_cache, 'The history of data files opened in ChimeraX')
        fmap = {}
        if fc is not None:
            for f in fc['files']:
                fs = FileSpec.from_state(self.session, f)
                fmap[(fs.path,fs.database)] = fs

        # Sections are kept in a separate file because older ChimeraX versions rewrite
        # the file history without them.  That file also has the files in sections in
        # case an older ChimeraX dropped them from the file history.
        sc = self._load_cache(self._sections_cache, 'The file history sections')
        if sc is not None:
            for f in sc.get('files', []):
                fs = FileSpec.from_state(self.session, f)
                key = (fs.path,fs.database)
                cur = fmap.get(key)
                if cur is None or cur.access_time < fs.access_time:
                    fmap[key] = fs
            names = set()
            for sstate in sc.get('sections', []):
                s = HistorySection.from_state(sstate)
                if s.name not in names:
                    names.add(s.name)
                    s.keys = [k for k in s.keys if k in fmap]
                    self._sections.append(s)
            cur = sc.get('current_section')
            self._current_section = cur if cur in names else None
        return fmap

    def _load_cache(self, cache, description):
        try:
            return cache.load()
        except Exception as e:
            backup_path = cache.backup()
            msg = ('%s was unreadable.\n' % description +
                   'The unreadable file has been copied to %s.\n' % backup_path +
                   'Please report this as a bug using menu Help / Report a Bug.\n\n' +
                   'The error was "%s".' % str(e))
            self.session.logger.bug(msg)
            return None

    def save_history(self):
        # Keep all files placed in sections and the most recent unplaced files.
        placed = self._placed_keys()
        files = self.files
        unplaced = [f for f in files if (f.path,f.database) not in placed]
        keep = set(id(f) for f in unplaced[-self.max_files:])
        saved = [f for f in files if id(f) in keep or (f.path,f.database) in placed]
        data = {
            'version': self.version,
            'files': [f.state() for f in saved],
        }
        self._file_cache.save(data)
        sections = {
            'version': self.sections_version,
            'sections': [s.state() for s in self._sections],
            'current_section': self._current_section,
            'files': [f.state() for f in files if (f.path,f.database) in placed],
        }
        self._sections_cache.save(sections)
        self._save_files = False

class HistorySection:
    def __init__(self, name, keys = None):
        self.name = name
        self.keys = [] if keys is None else keys    # List of (path, database) in display order

    def state(self):
        return {'name': self.name, 'files': [list(k) for k in self.keys]}

    @classmethod
    def from_state(cls, state):
        return cls(state['name'], [tuple(k) for k in state.get('files', [])])

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
