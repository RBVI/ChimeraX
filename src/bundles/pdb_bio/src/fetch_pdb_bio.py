# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
pdb_bio: Fetch biological assemblies from PDB
"""

# --------------------------------------------------------------------------------------
#
def fetch_pdb_biological_assemblies(session, pdb_id, *,
                                    max_assemblies=None, site='pdbe',
                                    ignore_cache=False, **kw):
    from chimerax.core.errors import UserError
    if len(pdb_id) != 4:
        raise UserError('PDB identifiers are 4 characters, got "%s"' % pdb_id)

    if site == 'pdbe':
        models = _fetch_pdbe_assemblies(session, pdb_id,
                                        max_assemblies = max_assemblies,
                                        ignore_cache = ignore_cache, **kw)
    elif site == 'rcsb':
        # First try to get mmCIF biological assemblies.
        models = _fetch_rcsb_mmcif_assemblies(session, pdb_id,
                                              max_assemblies = max_assemblies,
                                              ignore_cache = ignore_cache, **kw)
        if len(models) == 0:
            # No mmCIF format biological assemblies so try pdb format.
            models = _fetch_rcsb_pdb_assemblies(session, pdb_id,
                                                max_assemblies = max_assemblies,
                                                ignore_cache = ignore_cache, **kw)

    if len(models) == 0:
        msg = 'No biogical assemblies available for %s' % pdb_id
    else:
        msg = 'Opened %d biological assemblies for %s' % (len(models), pdb_id)
    session.logger.status(msg)

    return models, msg

# --------------------------------------------------------------------------------------
# https://www.ebi.ac.uk/pdbe/static/entry/download/6ts0-assembly-1.cif.gz
#
def _fetch_pdbe_assemblies(session, pdb_id, *,
                           max_assemblies=None, ignore_cache=False, **kw):
    mmcif_file = '%s-assembly-%d.cif.gz'
    save_name = '%s-assembly-%d.cif'
    mmcif_url = 'https://www.ebi.ac.uk/pdbe/static/entry/download/%s'
    models = _fetch_assemblies(session, pdb_id, mmcif_url, mmcif_file,
                               save_template=save_name,
                               max_assemblies=max_assemblies,
                               ignore_cache=ignore_cache,
                               transmit_compressed = False,  # PDBe assemblies are twice gzip compressed (July 2020)
                               format="mmcif", # disambiguate from small-molecule .cif
                               **kw)
    return models

# --------------------------------------------------------------------------------------
#
def _fetch_rcsb_mmcif_assemblies(session, pdb_id, *,
                                 max_assemblies=None, ignore_cache=False, **kw):
    mmcif_file = '%s-assembly%d.cif'
    mmcif_url = 'https://files.rcsb.org/download/%s'
    models = _fetch_assemblies(session, pdb_id, mmcif_url, mmcif_file,
                               max_assemblies=max_assemblies,
                               ignore_cache=ignore_cache,
                               **kw)
    return models

# --------------------------------------------------------------------------------------
#
def _fetch_rcsb_pdb_assemblies(session, pdb_id, *,
                               max_assemblies=None, ignore_cache=False, **kw):
    pdb_file = '%s.pdb%d.gz'
    save_name = '%s-assembly%d.pdb'
    pdb_url = 'https://files.rcsb.org/download/%s'
    models = _fetch_assemblies(session, pdb_id, pdb_url, pdb_file,
                               save_template=save_name,
                               max_assemblies=max_assemblies,
                               ignore_cache=ignore_cache, **kw)
    return models

# --------------------------------------------------------------------------------------
#
def _fetch_assemblies(session, pdb_id, url_template, file_template, *,
                      save_template=None, max_assemblies=None, ignore_cache=False,
                      transmit_compressed=True, **kw):
    models = []
    n = 1
    id = pdb_id.lower()
    from chimerax.core.fetch import fetch_file
    from chimerax.core.errors import UserError
    while max_assemblies is None or n <= max_assemblies:
        filename = file_template % (id, n)
        url = url_template % filename
        status_name = '%s bioassembly %d' % (pdb_id, n)
        save_name = filename if save_template is None else (save_template % (id, n))
        uncompress = filename.endswith('.gz')
        try:
            path = fetch_file(session, url, status_name, save_name, 'PDB',
                              uncompress=uncompress, transmit_compressed=transmit_compressed,
                              ignore_cache=ignore_cache, error_status=False)
        except UserError:
            break
        model_name = status_name
        mlist, status = session.open_command.open_data(path, name=model_name, **kw)
        if len(mlist) > 1:
            models.append(_group_subunit_models(session, mlist, status_name)) 
        else:
            models.extend(mlist)
        n += 1

    return models

# --------------------------------------------------------------------------------------
#
def _group_subunit_models(session, subunit_models, name):
    from chimerax.core.models import Model
    group = Model(name, session)
    for i,m in enumerate(subunit_models):
        m.name = 'subunit %d' % (i+1)
    group.add(subunit_models)
    return group
