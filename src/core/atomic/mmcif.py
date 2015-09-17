# vi: set expandtab shiftwidth=4 softtabstop=4:
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
    'pdbx_struct_assembly',
    'pdbx_struct_assembly_gen',
    'pdbx_struct_oper_list',
    'pdbx_poly_seq_scheme',
    'pdbx_nonpoly_scheme'
)


def open_mmcif(session, filename, name, *args, **kw):
    # mmCIF parsing requires an uncompressed file
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name

    from . import _mmcif
    from ..logger import Collator
    with Collator(session.logger, "Summary of problems reading mmCIF file",
                  kw.pop('log_errors', True)):
        _mmcif.set_Python_locate_function(
            lambda name: _get_template(name, session.app_dirs, session.logger))
        pointers = _mmcif.parse_mmCIF_file(filename, _additional_categories,
                                       session.logger)

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
    from .. import utils
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
    from .. import utils
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

    from .. import io
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


def get_mmcif_tables(model, table_names):
    raw_tables = model.metadata
    tlist = []
    for n in table_names:
        if n not in raw_tables:
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
            from chimera.core.commands.cli import commas
            missed, missed_suffix = commas(missing, ' and')
            have, have_suffix = commas(t, ' and')
            raise ValueError(
                'Field%s %s not in table "%s", have field%s %s'
                % (missed_suffix, missed, self.table_name, have_suffix, have))
        fi = tuple(t.index(f) for f in field_names)
        ftable = tuple(tuple(v[i] for i in fi) for v in self.values)
        return ftable
