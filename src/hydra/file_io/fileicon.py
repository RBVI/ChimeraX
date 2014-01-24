def set_file_icon(path, session, size = 512):

    from .. import scenes, _image3d
    if not _image3d.can_set_file_icon():
        return False
    qi = session.view.image((size,size))
    image = scenes.image_as_bytes(qi)
    return _image3d.set_file_icon(path, image)
