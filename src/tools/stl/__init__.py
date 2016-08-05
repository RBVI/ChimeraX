# vim: set expandtab shiftwidth=4 softtabstop=4:

def initialize(bundle_info, session):
    """Register STL file format."""
    from . import stl
    stl.register()
