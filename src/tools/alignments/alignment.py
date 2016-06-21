# vim: set expandtab shiftwidth=4 softtabstop=4:

class Alignment:
    """A sequence alignment"""
    def __init__(self, session, seqs, file_attrs=None, file_markups=None):
        self.seqs = seqs
