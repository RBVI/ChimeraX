# vim: set expandtab shiftwidth=4 softtabstop=4:

def finish(bundle_info, session):
    """De-install alignments manager from existing session"""
    del session.alignments

def initialize(bundle_info, session):
    """Install alignments manager into existing session"""
    from . import settings
    settings.init(session)

    from .manager import AlignmentsManager
    session.alignments = AlignmentsManager(session)

from .parse import open_file
