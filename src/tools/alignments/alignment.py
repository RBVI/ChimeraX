# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment"""
    def __init__(self, session, seqs, name, file_attrs=None, file_markups=None):
        self.seqs = seqs
        self.name = name
        self.file_attrs = file_attrs
        self.file_markups = file_markups

    def take_snapshot(self, session, flags):
        return { 'version': 1, 'seqs': self.seqs, 'name': self.name
            'file_attrs': self.file_atts, 'file_markups': self.file_markups }

    @staticmethod
    def restore_snapshot(session, data):
        return Alignment(data['seqs'], data['name'], data['file_attrs'], data['file_markups'])

    def _close(self):
        """Called by alignments manager so alignment can clean up (notify viewers, etc.)"""
        pass
