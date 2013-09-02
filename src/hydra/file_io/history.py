def save_history(path, viewer, recent_sessions_directory = None, thumbnail_size = 256, format = 'JPG'):
  sdir = sessions_directory() if recent_sessions_directory is None else recent_sessions_directory
  from os.path import splitext, basename, join
  bname = splitext(basename(path))[0] + '.'+format.lower()
  iname = unique_file_name(bname, sdir)
  f = open(join(sdir, 'sessions'), 'a')
  f.write('%s|%s\n' % (path, iname))
  f.close()
  s = thumbnail_size
  i = viewer.image((s,s))
  i.save(join(sdir, iname), format)

def unique_file_name(name, directory):
  from os.path import join, dirname, splitext, basename, isfile
  bname, suffix = splitext(name)
  uname = name
  upath = join(directory, uname)
  n = 1
  while isfile(upath):
    uname = '%s%d.%s' % (bname, n, suffix)
    upath = join(directory, uname)
    n += 1
  return uname

def sessions_directory():
  return user_settings_path('RecentSessions', directory = True)

def user_settings_path(filename = None, directory = False):
  from ..ui.qt import QtCore
  data_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericDataLocation)
  from os.path import isdir, join
  if not isdir(data_dir):
    return None
  c2_dir = join(data_dir, 'Chimera')
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

def show_history_thumbnails(recent_sessions_directory = None):
  from ..ui.gui import main_window as mw
  if history_shown():
    mw.show_graphics()
    return

  sdir = sessions_directory() if recent_sessions_directory is None else recent_sessions_directory
  from os.path import join, isfile
  rspath = join(sdir, 'sessions')
  if not isfile(rspath):
    return
  f = open(rspath, 'r')
  lines = f.readlines()
  f.close()

  hlist = []
  found = set()
  keepers = []
  removed_some = False
  for line in reversed(lines):
    fields = line.rstrip().split('|')
    if len(fields) == 2:
      spath,iname = fields
      if not spath in found:
        ipath = join(sdir,iname)
        if isfile(ipath) and isfile(spath):
          hlist.append((spath, ipath))
          found.add(spath)
          keepers.append(line)
        else:
          removed_some = True

  if removed_some:
    f = open(rspath, 'w')
    f.write(''.join(reversed(keepers)))
    f.close()
    
  html = make_html(hlist)
  mw.show_text(html, html=True, id = 'recent sessions', anchor_callback = open_clicked_session)

def open_clicked_session(url):
  path = url.toString(url.PreferLocalFile)         # session file path
  import os.path
  if not os.path.exists(path):
    from ..ui.gui import show_status
    show_status('Session file not found: %s' % path)
    return
  hide_history()
  from . import opensave
  opensave.open_session(path)

def history_shown():
  from ..ui.gui import main_window as mw
  return mw.showing_text() and mw.text_id == 'recent sessions'

def hide_history():
  from ..ui.gui import main_window as mw
  mw.show_graphics()

def make_html(hlist):
  from os.path import basename, splitext
  lines = ['<html>', '<head>', '<style>',
           'body { background-color: black; }',
           'a { text-decoration: none; }',      # No underlining of links
           'a:link { color: #FFFFFF; }',        # Link text color white.
           'table { float:left; }',             # Multiple image/caption tables per row.
           'td { font-size:large; }',
#           'td { text-align:center; }',        # Does not work in Qt 5.0.2
           '</style>', '</head>', '<body>']
  for spath, ipath in hlist:
    sname = splitext(basename(spath))[0]
    lines.extend(['',
                  '<a href="%s">' % spath,
                  '<table style="float:left;">',
                  '<tr><td><img src="%s" height=180>' % ipath,
                  '<tr><td><center>%s</center>' % sname,
                  '</table>',
                  '</a>'])
  lines.extend(['</body>', '</html>'])
  html = '\n'.join(lines)
  return html
