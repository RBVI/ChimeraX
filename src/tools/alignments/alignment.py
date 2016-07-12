# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment"""
    def __init__(self, session, seqs, name, file_attrs=None, file_markups=None, viewer=None):
        """'viewer' is the viewer to use:  None means use preference;
           False means no graphical viewer (accessible by command only).  If a graphical
           viewer is used, then when the last viewer goes away the alignment is automatically
           destroyed.  Otherwise, it must be explicitly destroyed."""
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
