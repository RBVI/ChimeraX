# vim: set expandtab shiftwidth=4 softtabstop=4:

def initialize(bundle_info, session):
    """Register STL file format."""
    from . import stl
    stl.register()

    # Configure STLModel for session saving
    stl.STLModel.bundle_info = bundle_info

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'STLModel':
        from . import stl
        return stl.STLModel
    return None
