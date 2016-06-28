# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment"""
    def __init__(self, session, seqs, identify_as, file_attrs=None, file_markups=None):
        self.seqs = seqs
