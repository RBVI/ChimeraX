# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

from . import structure
from .cli import UserError

_builtin_open = open
_initialized = False


def open_mmcif(session, filename, name, *args, **kw):
    from time import time
    t0 = time()
    # mmCIF parsing requires an uncompressed file
    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name

    from . import _mmcif
    _mmcif.set_Python_locate_function(
        lambda name: _get_template(name, session.app_dirs, session.logger))
    t1 = time()
    mol_blob = _mmcif.parse_mmCIF_file(filename)
    t2 = time()

    structures = mol_blob.structures
    models = []
    num_atoms = 0
    num_bonds = 0
    t3 = time()
    for structure_blob in structures:
        model = structure.StructureModel(name)
        t4 = time()
        models.append(model)
        model.mol_blob = structure_blob
        t5 = time()
        model.make_drawing()
        t6 = time()

        num_atoms += model.mol_blob.num_atoms
        num_bonds += model.mol_blob.num_bonds
        t7 = time()

    print("open_mmcif:")
    print("\tsetup: {}".format(t1 - t0))
    print("\treading mmcif: {}".format(t2 - t1))
    print("\tindividual blobs: {}".format(t3 - t2))
    print("\tStructureModel: {}".format(t4 - t3))
    print("\tappend model: {}".format(t5 - t4))
    print("\tmake drawing: {}".format(t6 - t5))
    print("\tcoords: {}".format(t7 - t6))
    return models, ("Opened mmCIF data containing %d atoms and %d bonds"
                    % (num_atoms, num_bonds))


def fetch_mmcif(session, pdb_id):
    from time import time
    t0 = time()
    if len(pdb_id) != 4:
        raise UserError("PDB identifiers are 4 characters long")
    import os
    # check on local system -- TODO: configure location
    lower = pdb_id.lower()
    subdir = lower[1:3]
    sys_filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, lower)
    if os.path.exists(sys_filename):
        return _builtin_open(sys_filename, 'rb')

    filename = "~/Downloads/Chimera/PDB/%s.cif" % pdb_id.upper()
    filename = os.path.expanduser(filename)

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    from . import utils
    url = "http://www.pdb.org/pdb/files/%s.cif" % pdb_id.upper()
    request = Request(url, unverifiable=True, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    t1 = time()
    try:
        utils.retrieve_cached_url(request, filename, session.logger)
    except URLError as e:
        raise UserError(str(e))
    t2 = time()
    print("fetch mmcif:")
    print("\tpre-fetch/URL: {}".format(t1 - t0))
    print("\tfetch/URL: {}".format(t2 - t1))
    return filename, pdb_id


def _get_template(name, app_dirs, logger):
    """Get Chemical Component Dictionary (CCD) entry"""
    import os
    # check in local cache
    filename = "~/Downloads/Chimera/CCD/%s.cif" % name
    filename = os.path.expanduser(filename)

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
