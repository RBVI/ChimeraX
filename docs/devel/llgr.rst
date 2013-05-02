.. automodule:: llgr
    :members:
    :show-inheritance:

Example
-------

    This example shows how to add an object made of triangles::

        # va: numpy array of 32-bit floating point XYZ coordinates
        # na: numpy array of 32-bit floating point XYZ normals
        # ta: numpy array of 16-bit unsigned integer indices of triangles

        vn = numpy.concatenate([va, na])     # combine coordinates and normals
        import llgr

        vn_id = llgr.next_data_id()
        llgr.create_buffer(vn_id, llgr.ARRAY, vn)

        tri_id = llgr.next_data_id()
        llgr.create_buffer(tri_id, llgr.ELEMENT_ARRAY, ta)

        color_id = llgr.next_data_id()
        rgba = array(cur_color, dtype=float32)
        llgr.create_singleton(color_id, rgba)

        uniform_scale_id = llgr.next_data_id()
        llgr.create_singleton(uniform_scale_id, array([1, 1, 1], dtype=float32))

        obj_id = llgr.next_object_id()
        AI = llgr.AttributeInfo
        mai = [
                AI("color", color_id, 0, 0, 4, llgr.Float),
                AI("position", vn_id, 0, 0, 3, llgr.Float),
                AI("normal", vn_id, va.nbytes, 0, 3, llgr.Float),
                AI("instanceScale", uniform_scale_id, 0, 0, 3, llgr.Float),
        ]

        llgr.create_object(obj_id, scene._program_id, 0, mai, llgr.Triangles,
                0, ta.size, tri_id, llgr.UShort)
