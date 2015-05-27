# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
pdb: PDB format support
=======================

Read Protein DataBank (PDB) files.
"""

from . import structure
from .cli import UserError

_builtin_open = open


def open_pdb(session, filename, name, *args, **kw):

    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    from . import pdbio
    mol_blob = pdbio.read_pdb_file(input, log=session.logger)
    if input != filename:
        input.close()

    structures = mol_blob.structures
    models = []
    num_atoms = 0
    num_bonds = 0
    for structure_blob in structures:
        model = structure.StructureModel(name, structure_blob)
        models.append(model)
        model.make_drawing()

        num_atoms += structure_blob.num_atoms
        num_bonds += structure_blob.num_bonds

    return models, ("Opened PDB data containing %d atoms and %d bonds"
                    % (num_atoms, num_bonds))


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

    filename = "~/Downloads/Chimera/PDB/%s.pdb" % pdb_id.upper()
    filename = os.path.expanduser(filename)

    if os.path.exists(filename):
        return filename, pdb_id  # TODO: check if cache needs updating

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    from . import utils
    url = "http://www.pdb.org/pdb/files/%s.pdb" % pdb_id.upper()
    request = Request(url, unverifiable=True, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    try:
        return utils.retrieve_cached_url(request, filename, session.logger), pdb_id
    except URLError as e:
        raise UserError(str(e))


def register():
    from . import io
    io.register_format(
        "PDB", structure.CATEGORY, (".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
        mime=("chemical/x-pdb", "chemical/x-spdbv"),
        reference="http://wwpdb.org/docs.html#format",
        open_func=open_pdb, fetch_func=fetch_pdb)
