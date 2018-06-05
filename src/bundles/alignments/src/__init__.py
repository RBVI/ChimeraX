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

from .cmd import get_alignment_sequence, SeqArg, AlignmentArg, AlignSeqPairArg

from chimerax.core.toolshed import BundleAPI

class _AlignmentsBundleAPI(BundleAPI):

    # so toolshed can find it...
    AlignmentArg = AlignmentArg

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
    def open_file(session, stream, file_name, format_name, alignment=True,
            ident=None, auto_associate=True):
        from .parse import open_file
        return open_file(session, stream, file_name, format_name=format_name.upper(),
            alignment=alignment, ident=ident, auto_associate=auto_associate)

    @staticmethod
    def save_file(session, path, format_name="fasta", alignment=None):
        if not alignment:
            alignments = list(session.alignments.alignments.values())
            from chimerax.core.errors import UserError
            if not alignments:
                raise UserError("No alignments open!")
            elif len(alignments) != 1:
                raise UserError("More than one alignment open;"
                    " use 'alignment' keyword to specify one")
            alignment = alignments[0]
        alignment.save(path, format_name=format_name)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_seqalign_command(logger)

bundle_api = _AlignmentsBundleAPI()
