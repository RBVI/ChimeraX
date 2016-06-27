# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.state import State
class AlignmentsManager(State):
    """Manager for sequence alignments"""
    def __init__(self, session):
        self.alignments = {}
        self.session = session

    def new_alignment(self, seqs, identify_as, align_attrs=None, align_markups=None):
        from .alignment import Alignment
        i = 1
        disambig = ""
        while identify_as+disambig in self.alignments:
            i += 1
            disambig = "[%d]" % i
        final_identify_as = identify_as+disambig
        alignment = Alignment(self.session, seqs, final_identify_as, align_attrs, align_markups)
        self.alignments[final_identify_as] = alignment
        return alignment

    def ses_restore(self, data):
        for am in self.alignments.values():
            am.close()
        self.alignments = data['alignments']

    def take_snapshot(self, session, flags):
        return { 'version': 1, 'alignments': self.alignments }

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.alignments
        mgr.ses_restore(data)
        return mgr
