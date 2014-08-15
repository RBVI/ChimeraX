def set_file_icon(path, session, size = 512, models = None):

    import sys
    if not sys.platform == 'darwin':
        return False
    from .. import mac_os_cpp
    if not mac_os_cpp.can_set_file_icon():
        return False
    v = session.view
    from .. import graphics
    c = graphics.camera_framing_models(models) if models else v.camera
    i = v.image(size, size, camera = c, models = models)
    from .. import scenes
    s = scenes.encode_image(i)      # Convert to jpeg image as a string
    return mac_os_cpp.set_file_icon(path, s)
