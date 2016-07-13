# vim: set expandtab shiftwidth=4 softtabstop=4:

def finish(bundle_info, session):
    """De-install alignments manager from existing session"""
    del session.alignments

def initialize(bundle_info, session):
    """Install alignments manager into existing session"""
    print("alignments bundle info:", type(bundle_info), bundle_info.__class__.__name__)
    from . import settings
    settings.init(session)

    from .manager import AlignmentsManager
    session.alignments = AlignmentsManager(session)

from .parse import open_file
