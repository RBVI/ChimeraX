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

class _ESMFoldDatabaseSettings(Settings):
    EXPLICIT_SAVE = {
        'structure_url': 'https://api.esmatlas.com/fetchPredictedStructure/{mgnify_id}',
        'pae_url': 'https://api.esmatlas.com/fetchConfidencePrediction/{mgnify_id}',
        'database_version': '0',
        'last_update_time': 0.0,	# seconds since 1970 epoch
        'update_interval': 86400.0,	# seconds
        'update_url': 'https://www.rbvi.ucsf.edu/chimerax/data/status/esmfold_database.json',
    }

# -----------------------------------------------------------------------------
#
def esmfold_model_url(session, mgnify_id, database_version = None):
    settings = _esmfold_database_settings(session)
    url_template = settings.structure_url
    if database_version is None:
        database_version = settings.database_version
    url = url_template.format(mgnify_id = mgnify_id, version = database_version)
    file_name = f'{mgnify_id}_v{database_version}.pdb'
    return url, file_name

# -----------------------------------------------------------------------------
#
def esmfold_pae_url(session, mgnify_id, database_version = None):
    settings = _esmfold_database_settings(session)
    url_template = settings.pae_url
    if database_version is None:
        database_version = settings.database_version
    url = url_template.format(mgnify_id = mgnify_id, version = database_version)
    file_name = f'{mgnify_id}_v{database_version}.json'
    return url, file_name

# -----------------------------------------------------------------------------
#
def default_database_version(session):
    settings = _esmfold_database_settings(session)
    return settings.database_version

# -----------------------------------------------------------------------------
#
def _esmfold_database_settings(session):
    settings = getattr(session, '_esmfold_database_settings', None)
    if settings is None:
        settings = _ESMFoldDatabaseSettings(session, "esmfold_database")
        session._esmfold_database_settings = settings
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
        filename = fetch_file(session, url, 'ESMFold database settings',
                              'esmfold_database.json', 'ESMFold',
                              ignore_cache=True, error_status = False)
        with open(filename, 'r') as f:
            import json
            db_settings = json.load(f)
            for key, value in db_settings.items():
                setattr(settings, key, value)

    except Exception:
        pass		# Could not reach update site

    settings.save()

