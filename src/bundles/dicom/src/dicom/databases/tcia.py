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
import os
import logging

from collections import defaultdict
from typing import List, Dict

from bs4 import BeautifulSoup

from chimerax.core.fetch import cache_directories
from chimerax.core.commands import run
from tcia_utils import nbia

_nbia_base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"

logging.getLogger('tcia_utils').setLevel(100)

class TCIADatabase:
    @staticmethod
    def get_collections():
        collections = nbia.getCollectionPatientCounts()
        collections_dict = defaultdict(dict)
        for entry in collections:
            name = entry['criteria']
            patient_count = entry['count']
            collections_dict[name] = {
                'name': name
                , 'patients': patient_count
            }
        collection_descs = nbia.getCollectionDescriptions()
        uris = defaultdict(str)
        for entry in collection_descs:
            uris[entry['collectionName']] = entry['descriptionURI']
        for name in collections_dict:
            if name in uris:
                collections_dict[name]['url'] = uris[name]
        # Workaround for the fact that this is the only entry in TCIA's dataset that doesn't have
        # a link
        collections_dict["CTpred-Sunitinib-panNET"]['url'] = "https://doi.org/10.7937/spgk-0p94"
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
            run(session, f"open {final_download_dir} format dicom")
        else:
            status_message = "No images were returned by TCIA"
        return [], status_message
