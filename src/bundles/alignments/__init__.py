# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _AlignmentsBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "AlignmentsManager":
            from . import manager
            return manager.AlignmentsManager
        elif class_name == "Alignment":
            from . import alignment
            return alignment.Alignment

    @staticmethod
    def initialize(session, bundle_info):
        """Install alignments manager into existing session"""
        from . import settings
        settings.init(session)

        from .manager import AlignmentsManager
        session.alignments = AlignmentsManager(session, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        """De-install alignments manager from existing session"""
        del session.alignments

    @staticmethod
    def open_file(session, stream, fname, format_name="FASTA", alignment=True,
            ident=None, auto_associate=True):
        from .parse import open_file
        return open_file(session, stream, fname, format_name=format_name, alignment=alignment,
            ident=ident, auto_associate=auto_associate)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_seqalign_command(logger)

bundle_api = _AlignmentsBundleAPI()
