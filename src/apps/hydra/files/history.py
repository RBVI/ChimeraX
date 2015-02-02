# vim: set expandtab ts=2 sw=2:
# Keep list of recently accessed files and thumbnail images.
class File_History:

  def __init__(self, session, history_file = None):

    self.session = session

    if history_file is None:
      from os.path import join
      history_file = self.recent_files_index()

    self.history_file = history_file
    from os.path import dirname, join
    self.thumbnail_directory = join(dirname(history_file), 'images')
    self.read = False
    self.changed = False
    self.files = {}             # Path to (access_time, image_name)
    self.thumbnail_size = 256
    self.image_format = 'JPEG'

    session.at_quit(self.write_history)

  def read_history(self, remove_missing = True):

    hfile = self.history_file
    from os.path import join, isfile
    if not isfile(hfile):
      if not self.install_example_sessions(hfile):
        return

    f = open(hfile, 'r')
    lines = f.readlines()
    f.close()

    files = {}
    for line in lines:
      fields = line.rstrip().split('|')
      if len(fields) == 3:
        fields.insert(1, '')  # No database specified
      spath,dbname,iname = fields[:3]
      atime = int(fields[3])
      files[spath] = (atime, iname, dbname)

    if remove_missing:
      removed_some = False
      for spath, (atime,iname,dbname) in tuple(files.items()):
        if not dbname and len([p for p in spath.split(',') if not isfile(p)]) > 0:
          files.pop(spath)
          removed_some = True
      if removed_some:
        self.changed = True

    self.read = True

    self.files = files

  def install_example_sessions(self, hfile):

    from os.path import dirname, join, exists
    from os import mkdir, listdir

    sdir = join(dirname(hfile), 'example_sessions')
    if exists(sdir):
      return False

    # Make directory for example sessions
    mkdir(sdir)

    # Make thumbnail images directory
    if not exists(self.thumbnail_directory):
      mkdir(self.thumbnail_directory)

    import hydra
    esdir = join(dirname(hydra.__file__), 'example_sessions')
    f = open(join(esdir, 'sessions'), 'r')
    slines = f.readlines()
    f.close()

    # Copy example sessions and thumbnail images and write history file.
    sfile = open(hfile, 'a')
    from shutil import copyfile
    for line in slines:
      fields = line.split('|')
      if len(fields) == 4:
        sname, dbname, iname, atime = [f.strip() for f in fields]
        copyfile(join(esdir,sname), join(sdir,sname))
        copyfile(join(esdir,iname), join(self.thumbnail_directory,iname))
        sfile.write('%s|%s|%s|%s\n' % (join(sdir,sname), dbname, iname, atime))
    sfile.close()

    return True

  def write_history(self):

    if not self.changed:
      return

    f = open(self.history_file, 'w')
    f.write('\n'.join('%s|%s|%s|%d' % (spath,dbname,iname,atime)
                      for spath,atime,iname,dbname in self.files_sorted_by_access_time()))
    f.close()

  def files_sorted_by_access_time(self):

    s = [(spath, atime, iname, dbname) for spath,(atime,iname,dbname) in self.files.items()]
    s.sort(key = lambda sai: sai[1])
    s.reverse()
    return s

  def add_entry(self, path, from_database = None, replace_image = False, models = None):

    if not self.read:
      self.read_history()

    atime,iname,dbname = self.files.get(path, (None,None,None))
    if iname is None:
      iname = self.thumbnail_filename(path)
      self.save_thumbnail(iname, models)
    elif replace_image:
      self.save_thumbnail(iname, models)

    from time import time
    atime = time()

    dbname = '' if from_database is None else from_database
    self.files[path] = (atime,iname,dbname)
    self.changed = True

  def thumbnail_filename(self, path):

    p0 = path.split(',')[0]
    from os.path import splitext, basename
    bname = splitext(basename(p0))[0] + '.' + self.image_format.lower()
    iname = unique_file_name(bname, self.thumbnail_directory)
    return iname

  def save_thumbnail(self, iname, models = None):

    from os.path import join
    ipath = join(self.thumbnail_directory, iname)
    s = self.thumbnail_size
    v = self.session.view
    i = v.image(s, s, drawings = models)
    if not i is None:
      i.save(ipath, self.image_format)

  def recent_files_index(self):

    return user_settings_path('recent_files')

  def show_thumbnails(self):

    mw = self.session.main_window
    if self.history_shown():
      mw.show_graphics()
      return

    if not self.read:
      self.read_history()
    
    mw.show_text(self.html(), html=True, id = 'recent sessions',
                 anchor_callback = self.open_clicked_session)

  def open_clicked_session(self, href):

    # href is session file path
    path, db = href.split('@') if '@' in href else (href, None)

    p0 = path.split(',')[0]
    import os.path
    if db is None and not os.path.exists(p0):
      self.session.show_status('File not found: %s' % p0)
      return

    self.hide_history()
    from . import opensave
    opensave.open_data(path, self.session, from_database = db)

  def history_shown(self):

    mw = self.session.main_window
    return mw.showing_text() and mw.text_id == 'recent sessions'

  def hide_history(self):

    mw = self.session.main_window
    mw.show_graphics()

  def most_recent_directory(self):

    if not self.read:
      self.read_history()
    if len(self.files) == 0:
      return None
    p, atime, iname, dbname = self.files_sorted_by_access_time()[0]
    from os.path import dirname
    return dirname(p)
    
  def html(self):

    from os.path import basename, splitext, join
    lines = ['<html>', '<head>', '<style>',
             'body { background-color: black; color: white; }',
             'a { text-decoration: none; }',      # No underlining of links
             'a:link { color: #FFFFFF; }',        # Link text color white.
             'table { float:left; }',             # Multiple image/caption tables per row.
             'td { font-size:large; }',
             #           'td { text-align:center; }',        # Does not work in Qt 5.0.2
             '</style>', '</head>', '<body>']
    s = self.files_sorted_by_access_time()
    for spath, atime, iname, dbname in s:
      url = '%s@%s' % (spath, dbname) if dbname else spath
      sname = self.display_name(spath)
      ipath = join(self.thumbnail_directory, iname)
      lines.extend(['',
                    '<a href="%s">' % url,
                    '<table style="float:left;">',
                    '<tr><td><img src="%s" height=180>' % ipath,
                    '<tr><td><center>%s</center>' % sname,
                    '</table>',
                    '</a>'])
    lines.extend(['</body>', '</html>'])
    html = '\n'.join(lines)
    return html

  def display_name(self, path):

    from os.path import basename, splitext, join
    paths = path.split(',')
    np = len(paths)
    if np == 1:
      n = splitext(basename(paths[0]))[0]
    else:
      fmt = '%s %s' if np == 2 else '%s ... %s'
      n = fmt % tuple(splitext(basename(p))[0] for p in (paths[0],paths[-1]))
    return n

def unique_file_name(name, directory):
  from os.path import join, dirname, splitext, basename, isfile
  bname, suffix = splitext(name)
  uname = name
  upath = join(directory, uname)
  n = 1
  while isfile(upath):
    uname = '%s%d%s' % (bname, n, suffix)
    upath = join(directory, uname)
    n += 1
  return uname

def user_settings_path(filename = None, directory = False):
  from .. import ui
  from os.path import isdir, join
  if hasattr(ui, 'user_settings_path'):
    c2_dir = ui.user_settings_path()
  else:
    data_dir = ui.user_settings_directory()
    if not isdir(data_dir):
      return None
    c2_dir = join(data_dir, 'Hydra')
  if not isdir(c2_dir):
    import os
    os.mkdir(c2_dir)
  if filename is None:
    return c2_dir
  fpath = join(c2_dir, filename)
  if directory and not isdir(fpath):
    import os
    os.mkdir(fpath)
  return fpath
