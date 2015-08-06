# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from . import structure
from .errors import UserError

_builtin_open = open
_initialized = False


def open_mmcif(session, filename, name, *args, **kw):
    # mmCIF parsing requires an uncompressed file
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name

    from . import _mmcif
    _mmcif.set_Python_locate_function(
        lambda name: _get_template(name, session.app_dirs, session.logger))
    pointers = _mmcif.parse_mmCIF_file(filename, session.logger)

    models = [structure.AtomicStructure(name, p) for p in pointers]
    for m in models:
        m.filename = filename

    return models, ("Opened mmCIF data containing %d atoms and %d bonds"
                    % (sum(m.num_atoms for m in models),
                       sum(m.num_bonds for m in models)))

def fetch_mmcif(session, pdb_id):
    if len(pdb_id) != 4:
        raise UserError("PDB identifiers are 4 characters long")
    import os
    # check on local system -- TODO: configure location
    lower = pdb_id.lower()
    subdir = lower[1:3]
    sys_filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, lower)
    if os.path.exists(sys_filename):
        return sys_filename, pdb_id

    filename = "~/Downloads/Chimera/PDB/%s.cif" % pdb_id.upper()
    filename = os.path.expanduser(filename)

    if os.path.exists(filename):
        return filename, pdb_id  # TODO: check if cache needs updating

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    from . import utils
    url = "http://www.pdb.org/pdb/files/%s.cif" % pdb_id.upper()
    request = Request(url, unverifiable=True, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    try:
        utils.retrieve_cached_url(request, filename, session.logger)
    except URLError as e:
        raise UserError(str(e))
    return filename, pdb_id


def _get_template(name, app_dirs, logger):
    """Get Chemical Component Dictionary (CCD) entry"""
    import os
    # check in local cache
    filename = "~/Downloads/Chimera/CCD/%s.cif" % name
    filename = os.path.expanduser(filename)

    if os.path.exists(filename):
        return filename  # TODO: check if cache needs updating

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    from . import utils
    url = "http://ligand-expo.rcsb.org/reports/%s/%s/%s.cif" % (name[0], name,
                                                                name)
    request = Request(url, unverifiable=True, headers={
        "User-Agent": utils.html_user_agent(app_dirs),
    })
    try:
        return utils.retrieve_cached_url(request, filename, logger)
    except URLError:
        if logger:
            logger.warning(
                "Unable to fetch template for '%s': might be missing bonds"
                % name)


def register():
    global _initialized
    if _initialized:
        return

    from . import io
    # mmCIF uses same file suffix as CIF
    # PDB uses chemical/x-cif when serving CCD files
    # io.register_format(
    #     "CIF", structure.CATEGORY, (), ("cif", "cifID"),
    #     mime=("chemical/x-cif"),
    #    reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
    io.register_format(
        "mmCIF", structure.CATEGORY, (".cif",), ("mmcif", "cif"),
        mime=("chemical/x-mmcif", "chemical/x-cif"),
        reference="http://mmcif.wwpdb.org/",
        requires_filename=True, open_func=open_mmcif, fetch_func=fetch_mmcif)

def read_mmcif_tables(mmcif_path, table_names):
  f = open(mmcif_path)
  tables = {}
  tname = None
  vcontinue = False
  semicolon_quote = False
  while True:
    line = f.readline()
    if tname is None:
      if line == '':
        break
      for tn in table_names:
        if line.startswith(tn + '.'):
          tname = tn
          tags = []
          values = []
          break
      if tname is None:
        continue
    if line.startswith(tname + '.'):
      tvalue = line.split('.', maxsplit=1)[1]
      tfields = tvalue.split(maxsplit = 1)
      tag = tfields[0]
      tags.append(tag)
      if len(tfields) == 2:
        value = remove_quotes(tfields[1])
        if values:            # Tags have values on same line without loop.
          values[0].append(value)
        else:
          values.append([value])
      elif values:
        # Other tags have values, so this one must have value on next line, e.g. 1afi _entity table.
        # Should really be looking for loop_.
        vcontinue = True
    elif line.startswith('#') or line == '':
      if [v for v in values if len(v) != len(tags)]:
        # Error: Number of values doesn't match number of tags.
        print (mmcif_path, tags, values)
      tables[tname] = mmCIF_Table(tname, tags, values)
      tname = None
    else:
      if line.startswith(';'):
        # Fields can extend onto next line if that line is preceded by a semicolon.
        # The whole line is treated as a single values as if quoted.
        lval = semicolon_quote = line[1:].rstrip()
        if lval:
          if values:
            values[-1].append(lval)
          else:
            values.append([lval])
      elif semicolon_quote:
        # Line that starts with semicolon continues on following lines until a line with only a semicolon.
        values[-1][-1] += line.rstrip()
      elif vcontinue:
        # Values simply continue on next line sometimes (e.g. 207l.cif _entity table).
        values[-1].extend(combine_quoted_values(line.split()))
      else:
        # New line of values
        values.append(combine_quoted_values(line.split()))
      vcontinue = (len(values[-1]) < len(tags))
        
  f.close()
  tlist = [tables.get(tn, None) for tn in table_names]
  return tlist

def combine_quoted_values(values):
  qvalues = []
  in_quote = False
  for e in values:
    if in_quote:
      if e.endswith(in_quote):
        qv.append(e[:-1])
        qvalues.append(' '.join(qv))
        in_quote = False
      else:
        qv.append(e)
    elif e.startswith("'") or e.startswith('"'):
      q = e[0]
      if e.endswith(q):
        qvalues.append(e[1:-1])
      else:
        in_quote = q
        qv = [e[1:]]
    else:
      qvalues.append(e)
  return qvalues

def remove_quotes(s):
  t = s.strip()
  if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
    return t[1:-1]
  return t

class mmCIF_Table:
  def __init__(self, table_name, tags, values):
    self.table_name = table_name
    self.tags = tags
    self.values = values
  def mapping(self, key_name, value_name, foreach = None):
    t = self.tags
    for n in (key_name, value_name, foreach):
      if n and not n in t:
        raise ValueError('Field "%s" not in table "%s", have fields %s' %
                         (n, self.table_name, ', '.join(t)))
    ki,vi = t.index(key_name), t.index(value_name)
    if foreach:
      fi = t.index(foreach)
      m = {}
      for f in set(v[fi] for v in self.values):
        m[f] = dict((v[ki],v[vi]) for v in self.values if v[fi] == f)
    else:
      m = dict((v[ki],v[vi]) for v in self.values)
    return m
  def fields(self, field_names):
    t = self.tags
    missing = [n for n in field_names if not n in t]
    if missing:
        raise ValueError('Fields %s not in table "%s", have fields %s' %
                         (', '.join(missing), self.table_name, ', '.join(t)))
    fi = tuple(t.index(f) for f in field_names)
    ftable = tuple(tuple(v[i] for i in fi) for v in self.values)
    return ftable 
      
