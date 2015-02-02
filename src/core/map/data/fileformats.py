# -----------------------------------------------------------------------------
# Python readers for array file formats.
#

# -----------------------------------------------------------------------------
# File description, file reader module name, file prefixes, and file suffixes.
#
file_types = (
  ('Amira mesh', 'amira', ['amira'], ['am'], False),
  ('APBS potential', 'apbs', ['apbs'], ['dx'], False),
  ('BRIX or DSN6 density map', 'dsn6', ['dsn6'], ['brix','omap'], False),
  ('CCP4 density map', 'ccp4', ['ccp4'], ['ccp4','map'], False),
  ('Chimera map', 'cmap', ['cmap'], ['cmp','cmap'], False),
  ('CNS or XPLOR density map', 'xplor', ['xplor'], ['cns','xplor'], False),
  ('DelPhi or GRASP potential', 'delphi', ['delphi'], ['phi'], False),
  ('DOCK scoring grid', 'dock', ['dock'], ['bmp','cnt','nrg'], False),
  ('EMAN HDF map', 'emanhdf', ['emanhdf'], ['hdf', 'h5'], False),
  ('Gaussian cube grid', 'gaussian', ['cube'], ['cube','cub'], False),
  ('gOpenMol grid', 'gopenmol', ['gopenmol'], ['plt'], False),
  ('Image stack', 'imagestack', ['images'], ['tif', 'tiff', 'png'], True),
  ('IMOD map', 'imod', ['imodmap'], ['rec'], False),
  ('MacMolPlt grid', 'macmolplt', ['macmolplt'], ['mmp'], False),
  ('MRC density map', 'mrc', ['mrc'], ['mrc'], False),
  ('NetCDF generic array', 'netcdf', ['netcdf'], ['nc'], False),
  ('Priism microscope image', 'priism', ['priism'], ['xyzw'], False),
  ('PROFEC free energy grid', 'profec', ['profec'], ['profec'], False),
  ('Purdue image format', 'pif', ['pif'], ['pif'], False),
  ('SITUS map file', 'situs', ['situs'], ['sit','situs'], False),
  ('SPIDER volume data', 'spider', ['spider'], ['spi','vol'], False),
  ('TOM toolbox EM density map', 'tom_em', ['tom_em'], ['em'], False),
  ('UHBD grid, binary', 'uhbd', ['uhbd'], ['grd'], False),
  )

# -----------------------------------------------------------------------------
#
#from . import mrc, netcdf, cmap, dsn6
from . import mrc, cmap, dsn6
file_writers = (
  ('MRC density map', 'mrc', '.mrc', mrc.write_mrc2000_grid_data, ()),
#  ('NetCDF generic array', 'netcdf', '.nc', netcdf.write_grid_as_netcdf, ()),
  ('Chimera map', 'cmap', '.cmap', cmap.write_grid_as_chimera_map,
   ('chunk_shapes', 'append', 'compress', 'multigrid')),
  ('BRIX map', 'dsn6', '.brix', dsn6.write_brix, ()),
  )
  
# -----------------------------------------------------------------------------
# The electrostatics file types are opened using a surface colormapping tool
# instead of displaying a contour surface.
#
electrostatics_types = ('apbs', 'delphi', 'uhbd')

# -----------------------------------------------------------------------------
#
class Unknown_File_Type(Exception):

  def __init__(self, path):

    self.path = path
    Exception.__init__(self)
    
  def __str__(self):

    return suffix_warning([self.path])

# -----------------------------------------------------------------------------
#
class File_Format_Error(Exception):
  pass
  
# -----------------------------------------------------------------------------
#
def suffix_warning(paths):

  path_string = ' '.join(paths)

  if len(paths) > 1:
    pluralize = 'es'
  else:
    pluralize = ''
    
  suffixes = reduce(lambda list, s: list + s[3], file_types, [])
  suffix_string = ' '.join(map(lambda s: '.'+s, suffixes))

  prefixes = reduce(lambda list, s: list + s[2], file_types, [])
  prefix_string = ' '.join(map(lambda s: s+':', prefixes))
  
  msg = ('Warning: Unrecognized file suffix%s for %s.\n' %
         (pluralize, path_string) +
         '\tKnown suffixes: %s\n' % suffix_string +
         '\tYou can specify the file type by prepending one of\n' +
         '\t%s\n' % prefix_string +
         '\tto the file name (eg. mrc:mydata).\n')
  return msg
  
# -----------------------------------------------------------------------------
#
def open_file(path, file_type = None):

  if file_type == None:
    p = path if isinstance(path, str) else path[0]
    file_type = file_type_from_suffix(p)
    if file_type == None:
      file_type, path = file_type_from_colon_specifier(p)
      if file_type == None:
        raise Unknown_File_Type(p)

  module_name = file_type
  module = __import__(module_name, globals(), level = 1)

  apath = absolute_path(path) if isinstance(path,str) else [absolute_path(p) for p in path]

  try:
    data = module.open(apath)
  except SyntaxError as value:
    raise File_Format_Error(value)
  
  return data

# -----------------------------------------------------------------------------
#
def file_type_from_suffix(path):
    
  for descrip, mname, prefix_list, suffix_list, batch in file_types:
    for suffix in suffix_list:
      if has_suffix(path, suffix):
        return mname
  return None

# -----------------------------------------------------------------------------
#
def file_type_from_colon_specifier(path):

  try:
    colon_position = path.rindex(':')
  except ValueError:
    return None, path
  
  first_part = path[:colon_position]
  last_part = path[colon_position+1:]

  module_names = map(lambda s: s[1], file_types)
  if last_part in module_names:
    return last_part, first_part

  return None, path

# -----------------------------------------------------------------------------
#
def has_suffix(path, suffix):

  parts = path.split('.')
  if len(parts) >= 2:
    return parts[-1] == suffix
  return 0

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def absolute_path(path):

  from os.path import abspath
  if isinstance(path, (tuple, list)):
    apath = tuple([abspath(p) for p in path])
  elif isinstance(path, str) and len(path) > 0:
    apath = abspath(path)
  else:
    apath = path
  return apath
  
# -----------------------------------------------------------------------------
#
def file_writer(path, format = None):

  if format is None:
    for fw in file_writers:
      if path.endswith(fw[2]):
        return fw
  else:
    for fw in file_writers:
      if format == fw[1] or format == fw[0]:
        return fw
  return None
  
# -----------------------------------------------------------------------------
#
def save_map_command(cmdname, args, session):

    from ...commands.parse import path_arg, volumes_arg, parse_arguments
    req_args = (('path', path_arg),
                ('maps', volumes_arg),
                )
    opt_args = ()
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    vlist = kw['maps']
    kw['grids'] = [v.data for v in vlist]
    kw.pop('maps')

    save_grid_data(**kw)

    # Set file icon image on Mac
    from ...files import fileicon
    fileicon.set_file_icon(kw['path'], session, models = vlist)

    session.file_history.add_entry(kw['path'], models = vlist)
 
# -----------------------------------------------------------------------------
#
def save_grid_data(grids, path, session, format = None, options = {}):

  import os.path
  path = os.path.expanduser(path)
  
  fw = file_writer(path, format)
  if fw is None:
    raise ValueError('Unknown format "%s"' % (format or path))

  descrip, format, suffix, write_data_file, allowed_options = fw
  badopt = [k for k in options.keys() if not k in allowed_options]
  if badopt:
    raise ValueError(('Unsupported options for format %s: %s'
                      % (fw[1], ' ,'.join(badopt))))

  from .griddata import Grid_Data
  if isinstance(grids, Grid_Data):
    glist = [grids]
  else:
    glist = grids

  if len(glist) > 1 and not ('multigrid' in allowed_options):
    raise ValueError('Cannot write multiple volumes using format %s' % format)

  # Use a temporary file if a source file is being overwritten.
  tpath = path
  if not ('append' in options):
    if matching_grid_path(glist, path):
      from tempfile import mkstemp
      f, tpath = mkstemp(suffix)
      from os import close, remove
      close(f)
      remove(tpath)  # Want new file to have normal, not secure, permissions.

  g = glist[0]
  from os.path import basename
  operation = 'Writing %s to %s' % (g.name, basename(path))
  from .progress import Progress_Reporter
  p = Progress_Reporter(operation, g.size, g.value_type.itemsize)
  if 'multigrid' in allowed_options:
    garg = glist
  else:
    garg = g
  write_data_file(garg, tpath, options = options, progress = p)

  if tpath != path:
    import os, os.path, shutil
    if os.path.exists(path):
      os.remove(path)
    shutil.move(tpath, path)

  # Update path in grid data object.
  for g in glist:
    g.set_path(path, format)

  from os.path import basename
  p.message('Wrote file %s' % basename(path))

  return format
  
# -----------------------------------------------------------------------------
#
def matching_grid_path(glist, path):

  from os.path import realpath
  rp = realpath(path)
  for g in glist:
    for gp in grid_paths(g):
      if realpath(gp) == rp:
        return True
  return False
  
# -----------------------------------------------------------------------------
#
def grid_paths(g):

  if isinstance(g.path, str):
    gpaths = [g.path]
  else:
    gpaths = g.path
  return gpaths
