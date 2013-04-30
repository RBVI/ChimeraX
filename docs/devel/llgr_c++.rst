Low-Level Graphics Library
==========================

The core of the LLGR, a Low-Level Graphics Library, is designed to be an
output-only API for rendering with WebGL_ or OpenGL_,
and to support picking, selection highlights, and translucency.
Shadows are also desirable but will be added later.
Also support for various optimizations,
e.g., instancing and level-of-detail, is provided.
Except for the rendering code,
it is a thin-layer over the underlying graphics library,
and exposes shader programs, program uniforms, vertex attributes,
and vertex attribute buffers.
Additional OpenGL concepts, eg., uniform buffers, may be exposed later.

.. _WebGL: http://www.webgl.org/
.. _OpenGL: http://www.opengl.org/

Typical Usage
-------------

To display the contents of those buffers,
a flat scene graph of objects is used.
Objects contain references to a particular shader program,
a transformation matrix,
and the attributes needed to draw the object.
All attributes refer to buffers, but should refer to a singleton buffer
when the value of the attribute is constant within the object.
Objects that only differ only by their singleton attributes
can be automatically instanced for faster drawing.

The default rendering code expects various uniforms and attributes
to be present, and they are documented below.

Typical Usage
-------------

A reasonable scenario would be to:

    #. Instantiate the shader programs with :cpp:func:`create_program`

    #. Instantiate the vertex coordinate (and vertex index) buffers
           with :cpp:func:`create_buffer`

    #. Instantiate other vertex attributes, such as color,
           as buffers with :cpp:func:`create_buffer`
           or singletons with :cpp:func:`create_singleton`

    #. For each object create a std::vector describing its attributes
           with :cpp:class:`AttributeInfo`
           and instantiate it with :cpp:func:`create_object`

    #. Set shader program uniforms with :cpp:func:`set_uniform`
           and :cpp:func:`set_uniform_matrix`

    #. :cpp:func:`render`

C++ API
-------

All data types are limited to 32-bit quantities or less.
In the future, we might allow 64-bit quantities for desktop OpenGL.

Types
~~~~~

.. cpp:namespace: llgr

    All of the public symbols are in the **llgr** namespace.

.. cpp:type:: Bytes

    For C++, Bytes is just **'void *'**.
    It is a separate type,
    so a Python wrapper generator can easily find byte arguments.

.. cpp:type:: Id

    API client provided identifier.
    Negative values are reserved for internal use
    and should not appear in client code.
    Zero has special meaning depending on where it is used,
    so all user-generated content should have positive identifiers.

.. cpp:type:: BufferTarget

    Buffer array types:

    ARRAY
        for array of data,

    ELEMENT_ARRAY
        for array of indices

.. cpp:type:: DataType

    Buffer data types:

    Byte
        8-bit integer

    UByte
        8-bit unsigned integer

    Short
        16-bit integer

    UShort
        16-bit unsigned integer

    Int
        32-bit integer

    UInt
        32-bit unsigned integer

    Float
        32-bit IEEE floating point

.. cpp:type:: ShaderType

    Shader variable types:

    IVec1, IVec2, IVec3, IVec4
        Integer vectors of 1-4 elements


    UVec1, UVec2, UVec3, UVec4
        Unsigned integer vectors of 1-4 elements
        *Not implemented.
        Reserved for forward compatibility
        with a WebGL that is based on OpenGL ES 3.0.*

    FVec1, FVec2, FVec3, FVec4
        Floating point vectors of 1-4 elements

    Mat2x2, Mat3x3, Mat4x4
        Square matrices

    Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3
        Rectangular matrices.
        *Not implemented.
        Reserved for forward compatibility
        with a WebGL that is based on OpenGL ES 3.0.*

    .. Note:

        [IUF]Vec1 instead of Int, UInt, Float to avoid clash with DataType's
        identifiers.

.. cpp:type:: PrimitiveType

    Drawing primitive types:

    Points, Lines, Line_loop, Line_strip, Triangles, Triangle_strip, Triangle_fan
        Same primitives that WebGL provides.

.. cpp:type:: Objects

    A std::vector of object identifiers

Shader Programs
~~~~~~~~~~~~~~~

Shaders problems need to be compatible with the rendering code.
Since the rendering code may change,
or there might be more than one way to render objects,
those requirements are documented below with the rendering code.

Managing shader programs is expected to be done
by a library layered on top of llgr.

.. Todo:

    Decide whether to annotate shader programs with expected OpenGL state,
    e.g., GL_DEPTH_TEST, and/or names of well-known uniform or vertex attributes,
    e.g., instance matrix, position, normal.

.. cpp:function:: void create_program(Id program_id, const char *vertex_shader, const char *fragment_shader)

    :param program_id: user-provided identifier to reference in other functions
        (zero is reserved, see :cpp:func:`set_uniform`)
    :param vertex_shader: vertex shader text
    :param fragment_shader: fragment shader text

    To reuse a program_id, just recreate it.

.. cpp:function:: void delete_program(Id program_id)

    :param program_id: existing program identifier

    Remove resources associated with program identifier.

.. cpp:function:: void clear_programs()

    Remove all existing programs.

.. cpp:function:: void set_uniform(Id program_id, const char *name, DataType type, uint32_t data_length, Bytes data)

    :param program_id: existing program identifier
        (program id zero means to set uniform in all existing programs)
    :param name: uniform name
    :param type: data type
    :param data_length: size of the data in bytes
    :param data: the actual data

.. cpp:function:: template \<typename T> void set_uniform(Id program_id, const char *name, const T *data)

    Template versions for all of the shader variable types,
    where the type and size are inferred from the data argument's type.

Buffers
~~~~~~~

Buffers contain coordinate and attribute data.

.. cpp:function:: void create_buffer(Id data_id, BufferTarget target, uint32_t data_length, Bytes data)

    :param data_id: provided buffer data id
    :param target: type of buffer
    :param data_length: size of the data in bytes
    :param data: the actual data

    Create buffer data.

.. cpp:function:: void create_singleton(Id data_id, uint32_t data_length, Bytes data)

    :param data_id: provided buffer data id
    :param data_length: size of the data in bytes
    :param data: the actual data

.. cpp:function:: void update_buffer(Id data_id, uint32_t offset, uint32_t stride, uint32_t data_length, Bytes data)

    TODO: future function to update column of existing buffer

.. cpp:function:: void delete_buffer(Id buffer_id)

    :param buffer_id: existing buffer identifier

    Remove resources associated with buffer identifier.

.. cpp:function:: void clear_buffers()

    Remove all existing buffers.

Matrices
~~~~~~~~

A matrix_id of zero is always the identity matrix.
Matrices are a separate kind of data

.. cpp:function:: void create_matrix(Id matrix_id, const float matrix[4][4], bool renormalize = false)

    :param data_id: provided matrix id
    :param matrix: the matrix
    :param renormalize: true if shear or scale matrix (*TODO: not implemented*)

.. cpp:function:: void delete_matrix(Id matrix_id)

    :param matrix_id: existing matrix identifier

    Remove resources associated with matrix identifier.

.. cpp:function:: void clear_matrices()

    Remove all existing matrices.

Objects
~~~~~~~

.. cpp:type:: AttributeInfo

.. cpp:member:: std::string name

    name of attribute

.. cpp:member:: Id data_id

    Data to use for attribute

.. cpp:member:: uint32_t offset

    Byte offset into data for first attribute value

.. cpp:member:: uint32_t stride

    Byte stride through data to next attribute value

.. cpp:member:: uint32_t count

    Number of data type (1-4)

.. cpp:member:: DataType type

    Type of attribute

.. cpp:member:: bool normalized

    For integer types, true if attribute values should be normalized to 0.0-1.0

.. cpp:type:: AttributeInfos

    std::vector\<AttributeInfo>

.. cpp:function void create_object(Id obj_id, Id program_id, Id matrix_id, \
	const AttributeInfos\& ais, PrimitiveType pt, \
	uint32_t first, uint32_t count, \
	Id index_data_id = 0, DataType index_data_type = Byte)

    :param obj_id: provided object identifier
    :param program_id: provided program identifier
    :param matrix_id: provided matrix identifier
    :param ais: vector of attribute information
    :param pt: primitive type
    :param first:
    :param count:
    :param index_data_id: provided data identifier for index data, zero if none
    :param index_data_type:

.. cpp:function:: void delete_object(Gluint obj_id)

    :param obj_id: existing object identifier

    Remove resources associated with object identifier.

.. cpp:function:: void clear_objects()

    Remove all existing objects.

Object annotations
~~~~~~~~~~~~~~~~~~

.. cpp:function:: void hide_objects(const Objects& objs)

    Don't draw given objects.

.. cpp:function:: void show_objects(const Objects& objs)

    Draw given objects (default).

.. cpp:function:: void transparent(const Objects& objs)

    Object is transparent, so draw it with extra code.

.. cpp:function:: void opaque(const Objects& objs)

    Object is opaque, so draw it normally (default).

.. cpp:function:: void selection_add(const Objects& objs)

    Add objects to selection set.

.. cpp:function:: void selection_remove(const Objects& objs)

    Remove objects from selection set.

.. cpp:function:: void selection_clear()

    Clear selection set.

LOD primitives
~~~~~~~~~~~~~~

Level-of-detail primitives. *TODO: implement LOD*

.. cpp:function:: void add_sphere(Id obj_id, float radius, \
	Id program_id, Id matrix_id, const AttributeInfos& ais, \
	const char *position = "position", const char *normal = "normal")

     Add sphere.

    :param obj_id: provided object identifier
    :param radius: the sphere's radius
    :param program_id: provided program identifier
    :param matrix_id: provided matrix identifier
    :param ais: vector of attribute information
    :param position: optional override for shader program's postion attribute
    :param normal: optional override for shader program's normal attribute

.. cpp:function:: void add_cylinder(Id obj_id, float radius, float length, \
	Id program_id, Id matrix_id, const AttributeInfos& ais, \
	const char *position = "position", const char *normal = "normal")

     Add cylinder.

    :param obj_id: provided object identifier
    :param radius: the cylinder's radius
    :param length: the cylinder's length
    :param program_id: provided program identifier
    :param matrix_id: provided matrix identifier
    :param ais: vector of attribute information
    :param position: optional override for shader program's postion attribute
    :param normal: optional override for shader program's normal attribute

.. cpp:function:: void clear_primitives()

    Remove all existing primitive objects and associated internal data.

Miscellaneous
~~~~~~~~~~~~~

.. cpp:function:: void clear_all()

    Remove data for all existing identifiers.

.. cpp:function:: void set_clear_color(float red, float green, float blue, float alpha)

    Set background clear color.

.. cpp:function:: void render()

    Render objects.
    Will invoke optimizer if some types of data have changed.
