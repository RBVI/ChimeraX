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

from chimerax.core.state import State
class AlignmentsManager(State):
    """Manager for sequence alignments"""
    def __init__(self, session, bundle_info):
        self.alignments = {}
        # bundle_info needed for session save
        self.bundle_info = bundle_info
        self.session = session
        self.viewer_synonyms = {}

    def delete_alignment(self, alignment):
        del self.alignments[alignment.name]
        alignment._close()

    def deregister_viewer(self, bundle_info):
        del self.viewer_synonyms[bundle_info]

    def new_alignment(self, seqs, identify_as, align_attrs=None, align_markups=None,
            autodestroy=None, viewer=None, **kw):
        """Create new alignment from 'seqs'

        Parameters
        ----------
        seqs : list of :py:class:`~chimerax.core.atomic.Sequence` instances
            Contents of alignment
        autodestroy : boolean or None
            Whether to automatically destroy the alignment when the last viewer for it
            is closed.  If None, then treated as False if the value of the 'viewer' keyword
            results in no viewer being launched, else True.
        viewer : str, False or None
           What alignment viewer to launch.  If False, do not launch a viewer.  If None,
           use the current preference setting for the user.  The string must either be
           the viewer's tool display_name or a synonym registered by the viewer (during
           its register_viewer call).
        """
        if viewer is None:
            from .settings import settings
            viewer = settings.viewer
        if viewer:
            for bundle_info, syms in self.viewer_synonyms.items():
                if bundle_info.display_name == viewer:
                    break
                if viewer in syms:
                    break
            else:
                self.session.logger.warning("No registered alignment viewer corresponds to '%s'"
                    % viewer)
                viewer = False

        from .alignment import Alignment
        i = 1
        disambig = ""
        while identify_as+disambig in self.alignments:
            i += 1
            disambig = "[%d]" % i
        final_identify_as = identify_as+disambig
        alignment = Alignment(self.session, seqs, final_identify_as,
            align_attrs, align_markups, autodestroy)
        self.alignments[final_identify_as] = alignment
        if viewer:
            bundle_info.start(alignment, **kw)
        return alignment

    def register_viewer(self, bundle_info, synonyms=[]):
        """Register an alignment viewer for possible use by the user.

        Parameters
        ----------
        bundle_info : BundleInfo
            The toolshed BundleInfo for your tool.
        synonyms : list of str
           Shorthands that the user could type instead of standard_name to refer to your tool
           in commands.  Example:  ['mav', 'multalign']
        """
        self.viewer_synonyms[bundle_info] = synonyms

    @property
    def registered_viewers(self):
        return self.viewer_synonyms

    def reset_state(self, session):
        for alignment in self.alignments.values():
            alignment._close()
        self.alignments.clear()

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.alignments
        mgr._ses_restore(data)
        return mgr

    def take_snapshot(self, session, flags):
        # viewer_synonyms are "session independent"
        return {
            'version': 1,

            'alignments': self.alignments,
        }

    def _ses_restore(self, data):
        for am in self.alignments.values():
            am.close()
        self.alignments = data['alignments']
