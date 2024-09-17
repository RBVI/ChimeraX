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
from .nrrd import NRRD, NRRDData, NRRDGrid

from chimerax.core.toolshed import BundleAPI
from chimerax.open_command import OpenerInfo
from chimerax.save_command import SaverInfo


class _NRRDBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:

            class NRRDOpenerInfo(OpenerInfo):
                def open(self, session, data, *args, **kw):
                    nrrd = NRRD.from_paths(session, data)
                    return nrrd.open()

            return NRRDOpenerInfo()

        elif mgr == session.save_command:

            class NRRDSaverInfo(SaverInfo):
                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg

                    return {"models": ModelsArg}

                def save(self, _, path, *, models=None):
                    """Save a volume in NIfTI format."""
                    import nrrd
                    import numpy as np

                    reference_volume = models[0].reference_volume
                    original_grid = reference_volume.data
                    if not isinstance(original_grid, NRRDGrid):
                        session.logger.warning(
                            "Source data is not NRRD; the saved segmentation may not necessarily line up with the source data if opened again."
                        )

                    data = np.asfortranarray(models[0].data.array)

                    nrrd.write(
                        file=path,
                        data=data,
                        header={
                            "dimension": 3,
                            "sizes": models[0].data.size[::-1],
                            "space origin": models[0].data.origin,
                            "spacings": models[0].data.step[::-1],
                            "kinds": ["domain", "domain", "domain"],
                            "type": data.dtype.name,
                        },
                    )

            return NRRDSaverInfo()


bundle_api = _NRRDBundle()
