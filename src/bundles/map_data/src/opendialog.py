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

# -----------------------------------------------------------------------------
#
def open_grid_files(paths_and_types, stack_images = True, log = None):

  grids = []
  error_message = ''
  unknown_type_paths = []
  from .fileformats import open_file, FileFormatError
  if stack_images:
    paths_and_types = batch_paths(paths_and_types)
  for path, file_type in paths_and_types:
    if file_type:
      try:
        glist = open_file(path, file_type, log = log)
        grids.extend(glist)
      except (IOError, SyntaxError, FileFormatError) as e:
        from os.path import basename
        if isinstance(path, (list,tuple)):
          descrip = '%s ... (%d files)' % (basename(path[0]), len(path))
        else:
          descrip = basename(path)
        msg = 'File %s, format %s\n%s\n' % (descrip, file_type, str(e))
        error_message = error_message + msg
    else:
      unknown_type_paths.append(path)

  if unknown_type_paths:
    import os.path
    file_names = [os.path.basename(p) for p in unknown_type_paths]
    files = ', '.join(file_names)
    msg = 'Unknown file types for %s' % files
    error_message = error_message + msg

  return grids, error_message

# ----------------------------------------------------------------------------
#
def batch_paths(paths_and_types):

  pt = []
  bp = []
  t2d = type_to_description_table()
  from chimera import fileInfo
  for path, ftype in paths_and_types:
    if isinstance(path, (tuple,list)) or not fileInfo.batch(t2d[ftype]):
      pt.append((path, ftype))
    else:
      bp.append((path, ftype))

  # Process file types where paths are batched to open one model.
  bpaths = {}
  for path, ftype in bp:
    if ftype in bpaths:
      bpaths[ftype].append(path)
    else:
      bpaths[ftype] = [path]
  for ftype, paths in bpaths.items():
    pt.append((tuple(paths), ftype))

  return pt
