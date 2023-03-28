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
import string

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
    @staticmethod
    def get_collections(session = None):
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
        # Workaround for the fact that this is the only entry in TCIA's dataset that doesn't have
        # a link
        collections_dict["CTpred-Sunitinib-panNET"]['url'] = "https://doi.org/10.7937/spgk-0p94"
        # Now get modalities, species, etc
        num_collections = len(collections)
        for index, collection in enumerate(collections_dict):
            if session:
                session.ui.thread_safe(session.logger.status, f"Loading collection {index+1}/{num_collections}")
            data = nbia.getSimpleSearchWithModalityAndBodyPartPaged(collection=collection)
            collections_dict[collection]['patients'] = data['totalPatients']
            collections_dict[collection]['body_parts'] = [string.capwords(x['value']) for x in data['bodyParts']]
            collections_dict[collection]['modalities'] = [m['value'] for m in data['modalities']]
            species_list = []
            for species in data['species']:
                id = species['value']
                species_list.append(NPEXSpecies.get(id, id))
            collections_dict[collection]['species'] = species_list
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
