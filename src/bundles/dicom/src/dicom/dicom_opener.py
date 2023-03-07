from .dicom import DICOM
from chimerax.open_command import OpenerInfo

class DicomOpener(OpenerInfo):
    def open(self, session, data, file_name, **kw):
        dcm = DICOM.from_paths(session, data)
        return dcm.open()
