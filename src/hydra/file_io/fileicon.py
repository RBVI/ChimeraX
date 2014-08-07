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
    qi = v.image(size, size, camera = c, models = models)
    from .. import scenes
    image = scenes.image_as_bytes(qi)
    return mac_os_cpp.set_file_icon(path, image)
