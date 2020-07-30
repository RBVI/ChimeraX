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
pdb_bio: Fetch biological assemblies from PDB
"""

# --------------------------------------------------------------------------------------
#
def fetch_pdb_biological_assemblies(session, pdb_id, *,
                                    max_assemblies=None, format_name=None,
                                    ignore_cache=False, **kw):
    from chimerax.core.errors import UserError
    if len(pdb_id) != 4:
        raise UserError('PDB identifiers are 4 characters, got "%s"' % pdb_id)

    # TODO: Manager provider code is always giving mmCIF.  Ask Eric how to make this work.
    format_name = None
    
    models = []
    if format_name is None or format_name.lower() == 'mmcif':
        models = _fetch_mmcif_assemblies(session, pdb_id,
                                         max_assemblies = max_assemblies,
                                         ignore_cache = ignore_cache, **kw)
    if len(models) == 0 and (format_name is None or format_name.lower() == 'pdb'):
        models = _fetch_pdb_assemblies(session, pdb_id,
                                       max_assemblies = max_assemblies,
                                       ignore_cache = ignore_cache, **kw)
    if len(models) == 0:
        msg = 'No biogical assemblies for %s' % pdb_id
    else:
        msg = 'Opened %d biological assemblies for %s' % (len(models), pdb_id)
    session.logger.status(msg)

    return models, msg

# --------------------------------------------------------------------------------------
#
def _fetch_mmcif_assemblies(session, pdb_id, *,
                            max_assemblies=None, ignore_cache=False, **kw):
    id = pdb_id.upper()
    mmcif_file = '%s-assembly%d.cif'
    mmcif_url = 'https://files.rcsb.org/download/%s'

    models = []
    n = 1
    from chimerax.core.fetch import fetch_file
    from chimerax.core.errors import UserError
    while max_assemblies is None or n <= max_assemblies:
        filename = mmcif_file % (id, n)
        url = mmcif_url % filename
        status_name = '%s bioassembly %d' % (pdb_id, n)
        try:
            path = fetch_file(session, url, status_name, filename, 'PDB', ignore_cache=ignore_cache)
        except UserError:
            break
        model_name = status_name
        mlist, status = session.open_command.open_data(path, format='mmcif', name=model_name, **kw)
        if len(mlist) > 1:
            models.append(_group_subunit_models(session, mlist, status_name)) 
        else:
            models.extend(mlist)
        n += 1

    return models

# --------------------------------------------------------------------------------------
#
def _fetch_pdb_assemblies(session, pdb_id, *,
                          max_assemblies=None, ignore_cache=False, **kw):
    id = pdb_id.upper()
    pdb_file = '%s.pdb%d.gz'
    pdb_url = 'https://files.rcsb.org/download/%s'
    models = []
    n = 1
    from chimerax.core.fetch import fetch_file
    from chimerax.core.errors import UserError
    while max_assemblies is None or n <= max_assemblies:
        filename = pdb_file % (id, n)
        url = pdb_url % filename
        save_name = '%s-assembly%d.pdb' % (id, n)
        status_name = '%s bioassembly %d' % (pdb_id, n)
        try:
            path = fetch_file(session, url, status_name, save_name, 'PDB',
                              ignore_cache=ignore_cache, uncompress=True)
        except UserError:
            break
        model_name = status_name
        mlist, status = session.open_command.open_data(path, format='pdb', name=model_name, **kw)
        if len(mlist) > 1:
            models.append(_group_subunit_models(session, mlist, status_name)) 
        else:
            models.extend(mlist)
        n += 1

    return models

def _group_subunit_models(session, subunit_models, name):
    from chimerax.core.models import Model
    group = Model(name, session)
    for i,m in enumerate(subunit_models):
        m.name = 'subunit %d' % (i+1)
    group.add(subunit_models)
    return group
