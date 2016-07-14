# vim: set expandtab ts=4 sw=4:

def finish(bundle_info, session):
    """De-register MAV from alignments manager"""
    session.alignments.deregister_viewer(bundle_info)

def initialize(bundle_info, session):
    """Register MAV with alignments manager"""
    session.alignments.register_viewer(bundle_info, ["mav", "multalign"])
