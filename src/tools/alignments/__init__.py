# vim: set expandtab shiftwidth=4 softtabstop=4:

def finish(bundle_info): pass

def initialize(bundle_info):
   """Install alignments managers into existing sessions and future sessions"""
    from .manager import AlignmentsManager
    from chimerax.core import chimerax_sessions, chimerax_triggers
    for sess in chimerax_sessions:
        am = AlignmentsManager()
        sess.add_state_manager('alignments', am)
        sess.alignments = am


from .parse import open_file
