.. automodule:: llgr
    :members:
    :show-inheritance:

Example
-------

    This example show how to add a 

        # va: numpy array of 32-bit floating point XYZ vertices
        # na: numpy array of 32-bit floating point XYZ normals
        # ta: numpy array of integer indices of triangles

        import llgr
        vn_id = llgr.next_data_id()
        vn = concatenate([va, na])
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

        tc = len(ta)
        if tc >= pow(2, 16):
                index_type = llgr.UInt
        elif tc >= pow(2, 8):
                index_type = llgr.UShort
        else:
                index_type = llgr.UByte
        llgr.create_object(obj_id, scene._program_id, 0, mai, llgr.Triangles,
                0, ta.size, tri_id, index_type)
