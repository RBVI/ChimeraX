# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.state import State
class AlignmentsManager(State):
    """Manager for sequence alignments"""
    def __init__(self, session):
        self.alignments = {}
        self.session = session
        self.viewer_info = {}

    def new_alignment(self, seqs, identify_as, align_attrs=None, align_markups=None, **kw):
        from .alignment import Alignment
        i = 1
        disambig = ""
        while identify_as+disambig in self.alignments:
            i += 1
            disambig = "[%d]" % i
        final_identify_as = identify_as+disambig
        alignment = Alignment(self.session, seqs, final_identify_as,
            align_attrs, align_markups, **kw)
        self.alignments[final_identify_as] = alignment
        return alignment

    def register_viewer(self, standard_name, import_info, synonyms=[]):
        """Register an alignment viewer for possible use by the user.


        Parameters
        ----------
        standard_name : str
            The toolshed / tools-menu name of your tool, i.e. name used in interfaces.
            Example:  "Multalign Viewer"
        import_info : str
           How to import your viewer class.  Namely the 'X' in "from X import Viewer".
           If the user requests to view an alignment using your tool, then after executing
           the quoted import statement, "Viewer(alignment)" will be called.
           Example:  "MultalignViewer.viewer"
        synonyms : list of str
           Shorthands that the user could type instead of standard_name to refer to your tool
           in commands.  Example:  ['mav', 'multalign']
        """
        self.viewer_info[standard_name] = (import_info, synonyms)

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.alignments
        mgr._ses_restore(data)
        return mgr

    def take_snapshot(self, session, flags):
        return { 'version': 1, 'alignments': self.alignments }

    def _ses_restore(self, data):
        for am in self.alignments.values():
            am.close()
        self.alignments = data['alignments']
