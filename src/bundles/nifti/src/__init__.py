# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
__version__ = "1.2"

import nibabel
from .nifti import NifTI, NiftiData, NiftiGrid

from chimerax.core.toolshed import BundleAPI
from chimerax.open_command import OpenerInfo
from chimerax.save_command import SaverInfo


class _NifTIBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:

            class NiftiOpenerInfo(OpenerInfo):
                def open(self, session, data, *args, **kw):
                    nifti = NifTI.from_paths(session, data)
                    return nifti.open()

            return NiftiOpenerInfo()

        elif mgr == session.save_command:

            class NiftiSaverInfo(SaverInfo):
                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg

                    return {"models": ModelsArg}

                def save(self, _, path, *, models=None):
                    """Save a volume in NIfTI format."""
                    import numpy as np

                    reference_volume = models[0].reference_volume
                    original_grid = reference_volume.data
                    affine = None
                    if isinstance(original_grid, NiftiGrid):
                        affine = original_grid.nifti_data._raw_data.affine
                    else:
                        session.logger.warning(
                            "Source data is not NIfTI; the saved segmentation will have the default rotation for NIfTI files."
                        )
                    img = nibabel.nifti2.Nifti2Image(models[0].data.array, affine)
                    nibabel.save(img, path)

            return NiftiSaverInfo()


bundle_api = _NifTIBundle()
