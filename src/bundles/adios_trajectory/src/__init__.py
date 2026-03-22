"""
ADIOS2 BP5 streaming trajectory reader for UCSF ChimeraX.

This bundle registers the 'bp5' data format and provides an open-command
provider that returns a BP5Trajectory object.  The trajectory reads one
step at a time from the BP5 archive via ADIOS2, maintaining only a small
ring buffer of fp32 coordinate frames in memory regardless of how long the
simulation is.
"""

from chimerax.core.toolshed import BundleAPI


class _BP5BundleAPI(BundleAPI):

    @staticmethod
    def open_file(session, data, file_name, **kw):
        from .reader import open_bp5
        return open_bp5(session, data, file_name, **kw)

    @staticmethod
    def get_class(class_name):
        if class_name == "BP5Trajectory":
            from .trajectory import BP5Trajectory
            return BP5Trajectory
        return None


bundle_api = _BP5BundleAPI()
