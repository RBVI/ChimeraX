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

To display the contents of those buffers,
a flat scene graph of objects is used.
Objects contain references to a particular shader program
and the attributes buffers needed to draw the object.

All data is limited to 32-bits.

.. _WebGL: http://www.webgl.org/
.. _OpenGL: http://www.opengl.org/

C++ API
-------

Types
~~~~~

.. cpp:namespace: llgr

.. cpp:type:: Id

    provided identifier

.. cpp:type:: BufferTarget

    buffer array types: ARRAY for array of data,
    ELEMENT_ARRAY for array of indices

.. cpp:type:: DataType

    buffer data types: Byte, UByte, Short, UShort, Int, UInt, Float

.. cpp:type:: ShaderType

    shader variable types: IVec1, IVec2, IVec3, IVec4,
    UVec1, UVec2, UVec3, UVec4,
    FVec1, FVec2, FVec3, FVec4,
    Mat2x2, Mat3x3, Mat4x4,
    Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3

    Note: [IUF]Vec1 instead of Int, UInt, Float to avoid clash with DataType's
    values.
    Note: UVec[1234] (unsigned integers)
    and asymmetrical matrices are for a future compatibility with a  WebGL 
    that is based on OpenGL ES 3.0.

.. cpp:type:: PrimitiveType

    drawing primitive types: Points, Lines, Line_loop, Line_strip,
    Triangles, Triangle_strip, Triangle_fan

Shader Programs
~~~~~~~~~~~~~~~

TODO: eventually the shader support will be constrained to those
that support the features (e.g., shadows, picking) that the
rendering engine supports
Perhaps like:
http://svn.code.sf.net/p/castle-engine/code/trunk/castle_game_engine/src/x3d/opengl/glsl

TODO: decide whether to annotate shader programs with expected OpenGL state,
e.g., GL_DEPTH_TEST, and/or names of well-known uniform or vertex attributes,
e.g., instancematrix, position, normal.

.. cpp:function:: void create_program(Id program_id, const char *vertex_shader, const char *fragment_shader)

    :param program_id: user-provided identifier to reference in other functions
        (zero is reserved)
    :param vertex_shader: vertex shader text
    :param fragment_shader: fragement shader text

.. cpp:function:: void delete_program(Id program_id)

    :param program_id: existing program identifier

    Remove resources associated with program identifier.

.. cpp:function:: void clear_programs()

    Remove all existing programs.

.. cpp:function:: void set_uniform(Id program_id, const char *name, DataType type, uint32_t data_length, void *data)

    :param program_id: existing program identifer
        (program id zero means to set uniform in all existing programs)
    :param name: uniform name
    :param type: data type
    :param data_length: size of the data in bytes
    :param data: the actual data

.. cpp:function:: template \<typename T> void set_uniform(Id program_id, const char *name, const T *data)

    Templated versions for all of the shader variable types,
    where the type and size are infered from the data argument's type.

Buffers
~~~~~~~

.. cpp:function:: void create_buffer(Id data_id, BufferTarget target, uint32_t data_length, void *data)

    :param data_id: provided buffer data id
    :param target: type of buffer
    :param data_length: size of the data in bytes
    :param data: the actual data

    Create buffer data.
    Data length in bytes = length * size * "sizeof"(type).
    So buffer only contains one type, unlike OpenGL.

.. cpp:function:: void create_singleton(Id data_id, uint32_t data_length, type, Bytes *data)

    :param data_id: provided buffer data id
    :param data_length: size of the data in bytes
    :param data: the actual data

.. cpp:function:: void update_buffer(Id data_id, uint32_t offset, uint32_t stride, uint32_t data_length, Bytes *data)

    TODO: update column of existing buffer

.. cpp:function:: void delete_buffer(Id buffer_id)

    :param buffer_id: existing buffer identifier

    Remove resources associated with buffer identifier.

.. cpp:function:: void clear_buffers()

    Remove all existing buffers.

Matrices
~~~~~~~~

A matrix_id of zero is always the identity matrix.

.. cpp:function:: void set_projection_matrix(float matrix[16], const char *uniform_name)

    :param matrix: the matrix in OpenGL order
    :param uniform_name: 

    This provides compatibility between OpenGL 2 graphics drivers
    and newer graphics drivers,
    by setting the projection matrix stack if the uniform_name is
    gl_ProjectionMatrix.
    Otherwise, broadcast uniform change to all current programs.
    TODO: get uniform name from program annotation or just eliminate

.. cpp:function:: void set_modelview_matrix(float matrix[16], const char *modelview_name, const char *normal_name)

    :param matrix: the matrix in OpenGL order
    :param modelview_name: name of modelview matrix uniform
    :param normal_name: name of normal matrix uniform

    This provides compatibility between OpenGL 2 graphics drivers
    and newer graphics drivers,
    by setting the modelview matrix stack if the uniform_name is
    gl_ModelViewMatrix.
    The rotation part of the modelview matrix is assumed to be orthonormal,
    so the normal matrix is just the rotation part of the modelview matrix
    (i.e., the inverse transpose is an identity operation).
    Otherwise, broadcast uniform change to all current programs.
    TODO: get uniform name from program annotation or just eliminate

.. cpp:function:: void matrix(Id matrix_id, float mat[16])

    :param data_id: provided matrix id
    :param mat: the matrix in OpenGL order

.. cpp:function:: void delete_matrix(Id matrix_id)

    :param matrix_id: existing matrix identifier

    Remove resources associated with matrix identifier.

.. cpp:function:: void clear_matrices()

    Remove all existing matrices.

// flat scene graph

.. cpp:type:: AttributeInfo

.. cpp:member:: std::string name

.. cpp:member:: Id data_id

.. cpp:member:: uint32_t offset

.. cpp:member:: uint32_t stride

.. cpp:member:: uint32_t count

.. cpp:member:: DataType type

.. cpp:member:: bool normalized

.. cpp:type:: AttributeInfos

    std::vector<AttributeInfo>

// state-sorting scene graph of arrays and instances
	// glVertexAttributePointer per attribute (glVertexAttrib if singleton)
	// sort by program, then attribute
	// automatic instancing: if attributes identical except for singletons,
	//   then they can be combined (need private data_id for new array of
	//   singleton values)

.. cpp:function:: void add_object(Id obj_id, Id program_id, Id matrix_id, \
	const AttributeInfo& ai)

.. cpp:function:: void delete_object(Gluint obj_id)

// LOD primitives: ignore initially
.. cpp:function:: void add_sphere(Id obj_id, GLfloat radius, Id program_id, \
	Id matrix_id, const AttributeInfo& ai)

.. cpp:function:: void add_cylinder(Id obj_id, GLfloat radius, Id program_id, \
	Id matrix_id, const AttributeInfo& ai)

typedef std::vector<Id> ObjectList;

// selection highlights
.. cpp:function:: void selection(const ObjectList& ol)

// translucency
.. cpp:function:: void tranlucent(const ObjectList& ol)

// picking
.. cpp:function:: Id pick(int x, int y)

.. cpp:function:: ObjectList pickarea(int x, int y, int width, int height)
