# vim: set expandtab shiftwidth=4 softtabstop=4:

def finish(bundle_info, session):
   """De-install alignments manager from existing session"""
   session.remove_state_manager(session.alignments)
   session.delete_attribute('alignments')

def initialize(bundle_info, session):
   """Install alignments manager into existing session"""
    from .manager import AlignmentsManager
    am = AlignmentsManager(session)
    session.add_state_manager('alignments', am)
    session.alignments = am


from .parse import open_file
