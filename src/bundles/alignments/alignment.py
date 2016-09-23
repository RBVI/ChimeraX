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
class Alignment(State):
    """A sequence alignment,
    
    Should only be created through new_alignment method of the alignment manager
    """
    def __init__(self, session, seqs, name, file_attrs, file_markups, autodestroy):
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
