# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from . import structure
from .cli import UserError

_builtin_open = open
_initialized = False


def open_mmCIF(session, filename, *args, **kw):
    # mmCIF parsing requires an uncompressed file
    name = kw['name'] if 'name' in kw else None
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name
    if name is None:
        name = filename

    from . import _mmcif
    _mmcif.set_Python_locate_function(
        lambda x: _get_template(x, session.app_dirs))
    mol_blob = _mmcif.parse_mmCIF_file(filename)

    model = structure.StructureModel(name)
    model.mol_blob = mol_blob
    model.make_drawing()

    coords = model.mol_blob.atoms.coords
    bond_list = mol_blob.bond_indices
    num_atoms = len(coords)
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
    from . import utils
    url = "http://www.rcsb.org/pdb/files/%s.cif" % pdb_id.upper()
    # TODO: save in local cache
    request = Request(url, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    try:
        return urlopen(request)
    except URLError as e:
        raise UserError(str(e))


def _get_template(name, app_dirs):
    """Get Chemical Component Dictionary (CCD) entry"""
    import os
    # check in local cache
    filename = "~/Downloads/Chimera/CCD/%s.cif" % name.upper()
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        return filename

    from urllib.request import URLError, Request, urlopen
    from . import utils
    url = "http://www.rbvi.ucsf.edu/ccdcache/%s" % name
    request = Request(url, headers={
        "User-Agent": utils.html_user_agent(app_dirs),
    })
    try:
        data = urlopen(request).read()
    except URLError as e:
        raise UserError("Unable to fetch template for '%s'" % name)

    dirname = os.path.dirname(filename)
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    with _builtin_open(filename, 'wb') as output:
        output.write(data)
    return filename


def register():
    global _initialized
    if _initialized:
        return

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
