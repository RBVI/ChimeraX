# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .cmd import (get_alignment_sequence, SeqArg, AlignmentArg, AlignSeqPairArg, SeqRegionArg,
                    AlignmentViewerArg, SequenceViewerArg)
from .alignment import clustal_strong_groups, clustal_weak_groups

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
    def init_manager(session, bundle_info, name, **kw):
        """Initialize alignments manager"""
        from . import settings
        settings.init(session)

        if not hasattr(session, 'alignments'):
            from .manager import AlignmentsManager
            session.alignments = AlignmentsManager(session, name, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        """De-install alignments manager from existing session"""
        del session.alignments

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_seqalign_command(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class SeqInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from .parse import open_file, NoSequencesError
                    try:
                        return open_file(session, data, file_name,
                            format_name=name.upper(), **kw)
                    except NoSequencesError as e:
                        from chimerax.core.errors import UserError
                        raise UserError(str(e))

                @property
                def open_args(self, *, session=session):
                    from chimerax.core.commands import BoolArg, StringArg, Or
                    return {
                        'alignment': BoolArg,
                        'auto_associate': BoolArg,
                        'ident': StringArg,
                        'viewer': Or(BoolArg, AlignmentViewerArg(session), SequenceViewerArg(session)),
                    }
        else:
            from chimerax.save_command import SaverInfo
            class SeqInfo(SaverInfo):
                def save(self, session, path, *, alignment=None, **kw):
                    if not alignment:
                        alignments = session.alignments.alignments
                        from chimerax.core.errors import UserError
                        if not alignments:
                            raise UserError("No alignments open!")
                        elif len(alignments) != 1:
                            raise UserError("More than one alignment open;"
                                " use 'alignment' keyword to specify one")
                        alignment = alignments[0]
                    alignment.save(path, format_name=name)

                @property
                def save_args(self):
                    return { 'alignment': AlignmentArg }

        return SeqInfo()

bundle_api = _AlignmentsBundleAPI()
