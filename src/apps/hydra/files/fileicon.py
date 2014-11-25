def set_file_icon(path, session, size = 512, models = None):

    import sys
    if not sys.platform == 'darwin':
        return False
    from .. import mac_os_cpp
    if not mac_os_cpp.can_set_file_icon():
        return False
    i = session.view.image(size, size, drawings = models)
    if i is None:
        return False
    from .. import scenes
    s = scenes.encode_image(i)      # Convert to jpeg image as a string
    return mac_os_cpp.set_file_icon(path, s)
