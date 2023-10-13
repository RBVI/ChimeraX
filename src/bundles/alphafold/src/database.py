# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
#
from chimerax.core.settings import Settings

class _AlphaFoldDatabaseSettings(Settings):
    EXPLICIT_SAVE = {
        'database_url': 'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.cif',
        'database_version': '4',
        'last_update_time': 0.0,	# seconds since 1970 epoch
        'update_interval': 86400.0,	# seconds
        'update_url': 'https://www.rbvi.ucsf.edu/chimerax/data/status/alphafold_database3.json',
    }

# -----------------------------------------------------------------------------
#
def alphafold_model_url(session, uniprot_id, database_version = None):
    settings = _alphafold_database_settings(session)
    url_template = settings.database_url
    if database_version is None:
        database_version = settings.database_version
    url = url_template.format(uniprot_id = uniprot_id, version = database_version)
    return url

# -----------------------------------------------------------------------------
#
def alphafold_pae_url(session, uniprot_id, database_version = None):
    model_url = alphafold_model_url(session, uniprot_id, database_version)
    url = model_url.replace('model', 'predicted_aligned_error').replace('.cif', '.json')
    return url

# -----------------------------------------------------------------------------
#
def uniprot_id_from_filename(filename):
    fields = filename.split('-')
    if len(fields) >= 4 and fields[0] == 'AF' and fields[2] == 'F1' and fields[3].startswith('model'):
        return fields[1]
    return None

# -----------------------------------------------------------------------------
#
def default_database_version(session):
    settings = _alphafold_database_settings(session)
    return settings.database_version

# -----------------------------------------------------------------------------
#
def _alphafold_database_settings(session):
    settings = getattr(session, '_alphafold_database_settings', None)
    if settings is None:
        settings = _AlphaFoldDatabaseSettings(session, "alphafold_database")
        session._alphafold_database_settings = settings
    _check_for_database_update(session, settings)
    return settings

# -----------------------------------------------------------------------------
#
def _check_for_database_update(session, settings):
    from time import time
    t = time()
    if t < settings.last_update_time + settings.update_interval:
        return
    
    url = settings.update_url
    if not url:
        return
    
    settings.last_update_time = t
    try:
        from chimerax.core.fetch import fetch_file
        filename = fetch_file(session, url, 'AlphaFold database settings',
                              'alphafold_database.json', 'AlphaFold',
                              ignore_cache=True, error_status = False)
        with open(filename, 'r') as f:
            import json
            db_settings = json.load(f)
            for key, value in db_settings.items():
                setattr(settings, key, value)

    except Exception:
        pass		# Could not reach update site

    settings.save()

