def set_file_icon(path, session, size = 512, models = None):

    from .. import scenes, _image3d
    if not _image3d.can_set_file_icon():
        return False
    v = session.view
    from .. import graphics
    c = graphics.camera_framing_models(size,size,models) if models else v.camera
    qi = v.image(size,size, camera = c, models = models)
    image = scenes.image_as_bytes(qi)
    return _image3d.set_file_icon(path, image)
