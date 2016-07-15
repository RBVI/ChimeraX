# vim: set expandtab shiftwidth=4 softtabstop=4:

def finish(bundle_info, session):
    """De-install alignments manager from existing session"""
    del session.alignments

def get_class(class_name):
    if class_name == "AlignmentsManager":
        from . import manager
        return manager.AlignmentsManager

def initialize(bundle_info, session):
    """Install alignments manager into existing session"""
    from . import settings
    settings.init(session)

    from .manager import AlignmentsManager
    session.alignments = AlignmentsManager(session, bundle_info)

from .parse import open_file
