# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

class DICOMBrowserTool:
    def __init__(self, session, model):
        """Bring up a tool to explore DICOM models open in the session.
        session: A ChimeraX session
        model: The root model containing patients, series, and files underneath
        """
        self.session = session
        self.model = model

    def build_ui(self):
        pass
