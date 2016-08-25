# vim: set expandtab shiftwidth=4 softtabstop=4:

def initialize(bundle_info, session):
    """Register IHM file format."""
    from . import ihm
    ihm.register()
