# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
pdb: PDB format support
=======================

Read Protein DataBank (PDB) files.
"""

from . import structure
from ..errors import UserError

_builtin_open = open


def open_pdb(session, filename, name, *args, **kw):

    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    log = session.logger
    from ..logger import Collator
    with Collator(log, "Summary of problems reading PDB file", kw.pop('log_errors', True)):
        from . import pdbio
        pointers = pdbio.read_pdb_file(input, log=log)
        if input != filename:
            input.close()

    lod = session.atomic_level_of_detail
    models = [structure.AtomicStructure(name, session, c_pointer = p, level_of_detail = lod)
        for p in pointers]

    return models, ("Opened PDB data containing %d atoms and %d bonds"
                    % (sum(m.num_atoms for m in models),
                       sum(m.num_bonds for m in models)))


def fetch_pdb(session, pdb_id):
    if len(pdb_id) != 4:
        raise UserError("PDB identifiers are 4 characters long")
    import os
    # check on local system -- TODO: configure location
    lower = pdb_id.lower()
    subdir = lower[1:3]
    sys_filename = "/databases/mol/pdb/%s/pdb%s.ent" % (subdir, lower)
    if os.path.exists(sys_filename):
        return sys_filename, pdb_id

    pdb_name = "%s.pdb" % pdb_id.upper()
    url = "http://www.pdb.org/pdb/files/%s" % pdb_name
    from ..fetch import fetch_file
    filename = fetch_file(session, url, 'PDB %s' % pdb_id, pdb_name, 'PDB')
    return filename, pdb_id

def register():
    from .. import io
    io.register_format(
        "PDB", structure.CATEGORY, (".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
        mime=("chemical/x-pdb", "chemical/x-spdbv"),
        reference="http://wwpdb.org/docs.html#format",
        open_func=open_pdb, fetch_func=fetch_pdb)
