# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from . import structure
from ..errors import UserError

_builtin_open = open
_initialized = False

_additional_categories = (
    #    'pdbx_struct_assembly',
    #    'pdbx_struct_assembly_gen',
    #    'pdbx_struct_oper_list',
    #    'pdbx_poly_seq_scheme',
    #    'pdbx_nonpoly_scheme'
)


def open_mmcif(session, filename, name, *args, **kw):
    # mmCIF parsing requires an uncompressed file
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name

    from . import _mmcif
    _mmcif.set_Python_locate_function(
        lambda name, session=session: _get_template(session, name))
    pointers = _mmcif.parse_mmCIF_file(filename, _additional_categories, session.logger)

    lod = session.atomic_level_of_detail
    models = [structure.AtomicStructure(session, name=name, c_pointer=p, level_of_detail=lod)
              for p in pointers]
    for m in models:
        m.filename = filename

    return models, ("Opened mmCIF data containing %d atoms and %d bonds"
                    % (sum(m.num_atoms for m in models),
                       sum(m.num_bonds for m in models)))


def fetch_mmcif(session, pdb_id, ignore_cache=False):
    if len(pdb_id) != 4:
        raise UserError('PDB identifiers are 4 characters long, got "%s"' % pdb_id)
    import os
    # check on local system -- TODO: configure location
    lower = pdb_id.lower()
    subdir = lower[1:3]
    sys_filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, lower)
    if os.path.exists(sys_filename):
        return sys_filename, pdb_id

    pdb_name = "%s.cif" % pdb_id.upper()
    url = "http://www.pdb.org/pdb/files/%s" % pdb_name
    from ..fetch import fetch_file
    filename = fetch_file(session, url, 'mmCIF %s' % pdb_id, pdb_name, 'PDB',
                          ignore_cache=ignore_cache)
    # double check that a mmCIF file was downloaded instead of an
    # HTML error message saying the ID does not exist
    with open(filename, 'U') as f:
        line = f.readline()
        if not line.startswith(('data_', '#')):
            f.close()
            import os
            os.remove(filename)
            raise UserError("Invalid mmCIF identifier")

    from .. import io
    models, status = io.open_data(session, filename, format='mmcif', name=pdb_id)
    return models, status


def _get_template(session, name):
    """Get Chemical Component Dictionary (CCD) entry"""
    import os
    from chimerax import app_dirs, app_dirs_unversioned
    # check in local cache
    if len(_cache_dirs) == 0:
        _cache_dirs.append(os.path.join(
            app_dirs_unversioned.user_cache_dir, 'CCD'))
        old_cache_dir = os.path.join('~', 'Downloads', 'Chimera', 'CCD')
        old_cache_dir = os.path.expanduser(old_cache_dir)
        if os.path.isdir(old_cache_dir):
            _cache_dirs.append(old_cache_dir)

    filename = '%s.cif' % name
    for d in _cache_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path  # TODO: check if cache needs updating

    path = os.path.join(_cache_dirs[0], filename)
    os.makedirs(_cache_dirs[0], exist_ok=True)

    from urllib.request import URLError, Request
    from .. import fetch
    url = "http://ligand-expo.rcsb.org/reports/%s/%s/%s.cif" % (name[0], name,
                                                                name)
    request = Request(url, unverifiable=True, headers={
        "User-Agent": fetch.html_user_agent(app_dirs),
    })
    try:
        return fetch.retrieve_cached_url(request, path, session.logger)
    except URLError:
        session.logger.warning(
            "Unable to fetch template for '%s': might be missing bonds"
            % name)
_cache_dirs = []


def register_mmcif_format():
    global _initialized
    if _initialized:
        return

    from .. import io
    # mmCIF uses same file suffix as CIF
    # PDB uses chemical/x-cif when serving CCD files
    # io.register_format(
    #     "CIF", structure.CATEGORY, (), ("cif",),
    #     mime=("chemical/x-cif"),
    #    reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
    io.register_format(
        "mmCIF", structure.CATEGORY, (".cif",), ("mmcif",),
        mime=("chemical/x-mmcif",),
        reference="http://mmcif.wwpdb.org/",
        requires_filename=True, open_func=open_mmcif)


def register_mmcif_fetch(session):
    from .. import fetch
    fetch.register_fetch(session, 'pdb', fetch_mmcif, 'mmcif',
                         prefixes=['pdb'], default_format=True)


def get_mmcif_tables(model, table_names):
    from . import _mmcif
    data = _mmcif.extract_mmCIF_tables(model.filename, table_names)
    tlist = []
    for name in table_names:
        if name not in data:
            tlist.append(None)
        else:
            tags, values_1d = data[name]
            num_columns = len(tags)
            slices = [values_1d[i::num_columns] for i in range(num_columns)]
            values_2d = list(zip(*slices))
            tlist.append(MMCIFTable(name, tags, values_2d))
    return tlist


def get_mmcif_tables_from_metadata(model, table_names):
    raw_tables = model.metadata
    tlist = []
    for n in table_names:
        if n not in raw_tables or (n + ' data') not in raw_tables:
            tlist.append(None)
        else:
            tags = raw_tables[n]
            values_1d = raw_tables[n + ' data']
            num_columns = len(tags)
            slices = [values_1d[i::num_columns] for i in range(num_columns)]
            values_2d = list(zip(*slices))
            tlist.append(MMCIFTable(n, tags, values_2d))
    return tlist


class MMCIFTable:

    def __init__(self, table_name, tags, values):
        self.table_name = table_name
        self.tags = tags
        self.values = values

    def __eq__(self, other):
        # for debugging
        if self.tags != other.tags or len(self.values) != len(other.values):
            return False
        return all(tuple(self.values[i]) == tuple(other.values[i])
                   for i in range(len(self.values)))

    def __repr__(self):
        return "MMCIFTable(%s, %s, ...[%d])" % (self.table_name, self.tags, len(self.values))

    def mapping(self, key_name, value_name, foreach=None):
        t = self.tags
        for n in (key_name, value_name, foreach):
            if n and n not in t:
                raise ValueError(
                    'Field "%s" not in table "%s", have fields %s'
                    % (n, self.table_name, ', '.join(t)))
        ki, vi = t.index(key_name), t.index(value_name)
        if foreach:
            fi = t.index(foreach)
            m = {}
            for f in set(v[fi] for v in self.values):
                m[f] = dict((v[ki], v[vi]) for v in self.values if v[fi] == f)
        else:
            m = dict((v[ki], v[vi]) for v in self.values)
        return m

    def fields(self, field_names):
        t = self.tags
        missing = [n for n in field_names if n not in t]
        if missing:
            from chimerax.core.commands.cli import commas, plural_form
            missed = commas(missing, ' and')
            missed_noun = plural_form(missing, 'Field')
            missed_verb = plural_form(missing, 'is', 'are')
            have = commas(t, ' and')
            have_noun = plural_form(t, 'field')
            raise ValueError('%s %s %s not in table "%s", have %s %s' % (
                missed_noun, missed, missed_verb, self.table_name, have_noun,
                have))
        fi = tuple(t.index(f) for f in field_names)
        ftable = tuple(tuple(v[i] for i in fi) for v in self.values)
        return ftable
