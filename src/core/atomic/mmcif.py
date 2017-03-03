# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from ..errors import UserError

_builtin_open = open
_initialized = False

_additional_categories = (
    'pdbx_struct_assembly',
    'pdbx_struct_assembly_gen',
    #    'pdbx_struct_oper_list',
    "entity",
    "entity_src_gen",
    "entity_src_nat",
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

    smid = kw.get('auto_style', True)

    from .structure import AtomicStructure
    models = [AtomicStructure(session, name = name, c_pointer = p, auto_style = smid) for p in pointers]
    for m in models:
        m.filename = filename

    return models, ("Opened mmCIF data containing %d atoms and %d bonds"
                    % (sum(m.num_atoms for m in models),
                       sum(m.num_bonds for m in models)))


_mmcif_sources = {
    "rcsb": "http://www.pdb.org/pdb/files/%s.cif",
    "pdbe": "http://www.ebi.ac.uk/pdbe/entry-files/download/%s.cif",
    "pdbe_updated": "http://www.ebi.ac.uk/pdbe/entry-files/download/%s_updated.cif",
    "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=mmcif&id=%s",
}


def fetch_mmcif(session, pdb_id, fetch_source="rcsb", ignore_cache=False, **kw):
    if len(pdb_id) != 4:
        raise UserError('PDB identifiers are 4 characters long, got "%s"' % pdb_id)
    import os
    pdb_id = pdb_id.lower()
    filename = None
    if not fetch_source.endswith('updated'):
        # check on local system -- TODO: configure location
        subdir = pdb_id[1:3]
        filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, pdb_id)
        if os.path.exists(filename):
            session.logger.info("Fetching mmCIF %s from system cache: %s" % (pdb_id, filename))
        else:
            filename = None
    if filename is None:
        base_url = _mmcif_sources.get(fetch_source, None)
        if base_url is None:
            raise UserError('unrecognized mmCIF/PDB source "%s"' % fetch_source)
        url = base_url % pdb_id
        pdb_name = "%s.cif" % pdb_id
        session.logger.status("Fetching mmCIF %s from %s" % (pdb_id, url))
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
    models, status = io.open_data(session, filename, format='mmcif', name=pdb_id, **kw)
    return models, status


def fetch_mmcif_pdbe(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbe", **kw)


def fetch_mmcif_pdbe_updated(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbe_updated", **kw)


def fetch_mmcif_pdbj(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbj", **kw)


def _get_template(session, name):
    """Get Chemical Component Dictionary (CCD) entry"""
    from ..fetch import fetch_file
    filename = '%s.cif' % name
    url = "http://ligand-expo.rcsb.org/reports/%s/%s/%s.cif" % (name[0], name,
                                                                name)
    try:
        return fetch_file(session, url, 'CCD %s' % name, filename, 'CCD', log='status')
    except UserError:
        session.logger.warning(
            "Unable to fetch template for '%s': might be missing bonds"
            % name)
        return None


def register_mmcif_format():
    global _initialized
    if _initialized:
        return

    from .. import io
    from . import structure
    
    # mmCIF uses same file suffix as CIF
    # PDB uses chemical/x-cif when serving CCD files
    # io.register_format(
    #     "CIF", structure.CATEGORY, (), ("cif",),
    #     mime=("chemical/x-cif"),
    #    reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
    from .mmcif_write import write_mmcif
    io.register_format(
        "mmCIF", structure.CATEGORY, (".cif",), ("mmcif",),
        mime=("chemical/x-mmcif",),
        reference="http://mmcif.wwpdb.org/",
        requires_filename=True, open_func=open_mmcif, export_func=write_mmcif)


def register_mmcif_fetch():
    from .. import fetch
    fetch.register_fetch('pdb', fetch_mmcif, 'mmcif',
                         prefixes=['pdb'], is_default_format=True)
    fetch.register_fetch('pdbe', fetch_mmcif_pdbe, 'mmcif',
                         prefixes=['pdbe'], is_default_format=True)
    fetch.register_fetch('pdbe_updated', fetch_mmcif_pdbe_updated, 'mmcif',
                         prefixes=['pdbe_updated'], is_default_format=True)
    fetch.register_fetch('pdbj', fetch_mmcif_pdbj, 'mmcif',
                         prefixes=['pdbj'], is_default_format=True)


def get_mmcif_tables(filename, table_names):
    from os import path
    if path.exists(filename):
        from . import _mmcif
        data = _mmcif.extract_mmCIF_tables(filename, table_names)
    else:
        data = {}
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

    def fields(self, field_names, allow_missing_fields = False):
        t = self.tags
        if allow_missing_fields:
            fi = tuple((t.index(f) if f in t else -1) for f in field_names)
            ftable = tuple(tuple((v[i] if i >= 0 else '') for i in fi) for v in self.values)
        else:
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
