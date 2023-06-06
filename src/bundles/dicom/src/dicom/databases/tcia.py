# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2023 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import logging
import os
import json
from pathlib import Path
import string
import datetime

from collections import defaultdict
from typing import List, Dict

from chimerax.core.fetch import cache_directories
from chimerax.core.commands import run
from tcia_utils import nbia

from ..dicom import DICOM

_nbia_base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"

logging.getLogger('tcia_utils').setLevel(100)

NPEXSpecies = {
    "337915000": "Human"
    , "447612001": "Mouse"
    , "448771007": "Dog"
}

class TCIADatabase:

    data_usage_disclaimer = """\
The data presented in the Download DICOM tool is provided by The Cancer Imaging Archive (TCIA), an effort funded
by the National Cancer Institute's Cancer Imaging Program. When using these datasets it is up to you to abide by
TCIA's Data Usage Policy, which is provided
<a href="https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions">here</a>.
<br>
<br>
Please note that each collection carries its own citation and data usage policy, which you can find from ChimeraX.
To see the citation and data usage policy of one or more collections: highlight the collection(s) you are interested
in, then right click and click on "Load Webpage for Chosen Entries", or click on the "Load Webpage" button at the
bottom of the tool. ChimeraX's browser will open the collections' corresponding webpages on TCIA's website, where you
can find the information you need.
<br>
<br>
Please read the usage policy. If you agree to it and to abide by the citation and data usage policies of each collection
you use, hit 'OK' to close this dialog and continue. If you do not agree, please hit 'Cancel'.
"""
    @staticmethod
    def get_collections(session = None):
        from chimerax import app_dirs # noqa
        tcia_cache_dir = os.path.join(app_dirs.user_cache_dir, "dicom")
        if not os.path.exists(tcia_cache_dir):
            Path(tcia_cache_dir).mkdir(parents=True, exist_ok=True)
        collections_cache_file = os.path.join(tcia_cache_dir, "tcia_collections")
        need_to_cache = False
        if not os.path.exists(collections_cache_file):
            need_to_cache = True
        else:
            time = os.path.getmtime(collections_cache_file)
            last_mod_date = datetime.datetime.fromtimestamp(time).strftime("%Y/%m/%d")
            today = datetime.datetime.now().strftime("%Y/%m/%d")
            last_mod_date_is_today = (last_mod_date == today)
            if not last_mod_date_is_today:
                new_collections = nbia.getUpdatedSeries(last_mod_date)
                if new_collections and not last_mod_date_is_today:
                    need_to_cache = True
        if not need_to_cache:
            with open(collections_cache_file, "r") as f:
                collections_dict = json.loads(f.read())
            session.ui.thread_safe(session.logger.status, f"Loaded TCIA collections using cached data")
        else:
            collections = nbia.getCollections()
            collection_descs = nbia.getCollectionDescriptions()

            uris = defaultdict(str)
            collections_dict = defaultdict(dict)
            for entry in collection_descs:
                uris[entry['collectionName']] = entry['descriptionURI']
            for entry in collections:
                name = entry['Collection']
                collections_dict[name] = {
                    'name': name
                    , 'url': uris.get(name, '')
                }
            # Now get modalities, species, etc
            num_collections = len(collections)
            failed_to_fetch = False
            for index, collection in enumerate(collections_dict):
                if session:
                    session.ui.thread_safe(session.logger.status, f"Loading collection {index+1}/{num_collections}")
                data = nbia.getSimpleSearchWithModalityAndBodyPartPaged(collection=collection)
                if data:
                    try:
                        collections_dict[collection]['patients'] = data['totalPatients']
                        collections_dict[collection]['body_parts'] = [string.capwords(x['value']) for x in data['bodyParts']]
                        collections_dict[collection]['modalities'] = [m['value'] for m in data['modalities']]
                        species_list = []
                        if data.get('species', []) is not None:
                            for species in data.get('species', []):
                                id = species['value']
                                species_list.append(NPEXSpecies.get(id, id))
                        collections_dict[collection]['species'] = species_list
                    except KeyError:
                        failed_to_fetch = True
                else:
                    failed_to_fetch = True
            if failed_to_fetch:
                session.ui.thread_safe(session.logger.warning, "Failed to fetch some collections' metadata; won't cache these results.")
            else:
                with open(collections_cache_file, 'w') as f:
                    f.write(json.dumps(collections_dict))
        return collections_dict.values()

    @staticmethod
    def get_study(collection, patientId="", studyUid=""):
        return nbia.getStudy(collection, patientId, studyUid)

    @staticmethod
    def get_series(studyUid) -> List[Dict[str, str]]:
        return nbia.getSeries(studyUid=studyUid)

    @staticmethod
    def getImages(session, studyUID, ignore_cache, **kw):
        old_cwd = os.getcwd()
        download_dir = cache_directories()[0]
        if not os.path.exists(download_dir):
            Path(download_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(download_dir)
        # Unfortunately the return type of this function is a Pandas dataframe, but we can
        # predict where tcia_utils will put the images
        final_download_dir = os.path.join(
            download_dir, "tciaDownload", studyUID)
        nbia.downloadSeries([{"SeriesInstanceUID": studyUID}])
        os.chdir(old_cwd)
        status_message = ""
        if os.path.exists(final_download_dir):
            models, msg = DICOM.from_paths(session, final_download_dir).open()
            msg += "Images from TCIA may be rotated so that flat planes appear invisible. If the screen looks black but no error message has been issued, try rotating the model into view with your mouse."
        else:
            models = []
            msg = "No images were returned by TCIA"
        return models, msg
