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

from typing import List, Dict
from contextlib import redirect_stdout
from chimerax.core.fetch import cache_directories
from chimerax.core.commands import run
from tcia_utils import nbia

_nbia_base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"


class TCIADatabase:
    @staticmethod
    def getCollections(patient_counts=False):
        with redirect_stdout(None):
            if patient_counts:
                return nbia.getCollectionPatientCounts()
            else:
                return nbia.getCollections()

    @staticmethod
    def getStudy(collection, patientId="", studyUid=""):
        with redirect_stdout(None):
            return nbia.getStudy(collection, patientId, studyUid)

    @staticmethod
    def getSeries(studyUid) -> List[Dict[str, str]]:
        with redirect_stdout(None):
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
