# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from . import structure
from .cli import UserError

_builtin_open = open


def open_mmCIF(session, filename, *args, **kw):
    # mmCIF parsing requires an uncompressed file
    name = kw['name'] if 'name' in kw else None
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name
    if name is None:
        name = filename

    from . import _mmcif
    mol_blob = _mmcif.parse_mmCIF_file(filename)

    model = structure.StructureModel(name)
    model.mol_blob = mol_blob
    model.make_drawing()

    atom_blob, bond_list = model.mol_blob.atoms_bonds
    num_atoms = len(atom_blob.coords)
    num_bonds = len(bond_list)

    return [model], ("Opened mmCIF data containing %d atoms and %d bonds"
                     % (num_atoms, num_bonds))


def fetch_mmcif(session, pdb_id):
    if len(pdb_id) != 4:
        raise UserError("PDB identifiers are 4 characters long")
    import os
    # TODO: use our own cache
    # check in local cache
    filename = "~/Downloads/Chimera/PDB/%s.cif" % pdb_id.upper()
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        return _builtin_open(filename, 'rb')
    # check on local system -- TODO: configure location
    lower = pdb_id.lower()
    subdir = lower[1:3]
    filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, lower)
    if os.path.exists(filename):
        return _builtin_open(filename, 'rb')
    from urllib.request import URLError, Request, urlopen
    url = "http://www.rcsb.org/pdb/files/%s.cif" % pdb_id.upper()
    # TODO: save in local cache
    from . import utils
    request = Request(url, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    try:
        return urlopen(request)
    except URLError as e:
        raise UserError(str(e))


def register():
    from . import io
    # mmCIF uses same file suffix as CIF
    # io.register_format(
    #     "CIF", structure.CATEGORY, (), ("cif", "cifID"),
    #     mime=("chemical/x-cif"),
    #    reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
    io.register_format(
        "mmCIF", structure.CATEGORY, (".cif",), ("mmcif", "cif"),
        mime=("chemical/x-mmcif",), reference="http://mmcif.wwpdb.org/",
        requires_seeking=True, open_func=open_mmCIF)
