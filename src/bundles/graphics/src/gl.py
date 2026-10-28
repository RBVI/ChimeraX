# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

'''
Small OpenGL binding for drawing with Qt OpenGL contexts
=========================================================

Importing PyOpenGL's OpenGL.GL loads about 190 modules (every GL version and
several extensions) and took ~45 ms of every ChimeraX start.  This module
defines the GL functions and constants ChimeraX uses, calling them through
ctypes with function pointers from QOpenGLContext.getProcAddress(), and
behaves like PyOpenGL for those uses: results have the same types, numpy
arrays are passed as pointers, and an OpenGL error after a call raises
GLError (PyOpenGL's default ERROR_CHECKING).

Any other name, or argument forms not handled here, go to PyOpenGL, which is
imported the first time that happens.
'''

import ctypes
from ctypes import c_uint, c_int, c_ubyte, c_float, c_double, c_void_p, c_ssize_t, c_char_p

# Constants used by ChimeraX, with values from the OpenGL headers.
GL_BACK                              = 0x405
GL_BACK_LEFT                         = 0x402
GL_BACK_RIGHT                        = 0x403
GL_BLEND                             = 0xBE2
GL_BYTE                              = 0x1400
GL_CLAMP_TO_BORDER                   = 0x812D
GL_CLAMP_TO_EDGE                     = 0x812F
GL_CLIP_DISTANCE0                    = 0x3000
GL_COLOR_ATTACHMENT0                 = 0x8CE0
GL_COLOR_BUFFER_BIT                  = 0x4000
GL_COLOR_LOGIC_OP                    = 0xBF2
GL_COMPARE_REF_TO_TEXTURE            = 0x884E
GL_COMPILE_STATUS                    = 0x8B81
GL_CONSTANT_COLOR                    = 0x8001
GL_CONTEXT_COMPATIBILITY_PROFILE_BIT = 0x2
GL_CONTEXT_CORE_PROFILE_BIT          = 0x1
GL_CONTEXT_PROFILE_MASK              = 0x9126
GL_CULL_FACE                         = 0xB44
GL_DEBUG_OUTPUT                      = 0x92E0
GL_DEPTH                             = 0x1801
GL_DEPTH24_STENCIL8                  = 0x88F0
GL_DEPTH_ATTACHMENT                  = 0x8D00
GL_DEPTH_BUFFER_BIT                  = 0x100
GL_DEPTH_COMPONENT                   = 0x1902
GL_DEPTH_COMPONENT24                 = 0x81A6
GL_DEPTH_TEST                        = 0xB71
GL_DRAW_FRAMEBUFFER                  = 0x8CA9
GL_DYNAMIC_DRAW                      = 0x88E8
GL_ELEMENT_ARRAY_BUFFER              = 0x8893
GL_EXTENSIONS                        = 0x1F03
GL_FALSE                             = 0x0
GL_FLOAT                             = 0x1406
GL_FRAGMENT_SHADER                   = 0x8B30
GL_FRAMEBUFFER                       = 0x8D40
GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE = 0x8215
GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE  = 0x8214
GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE = 0x8216
GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE = 0x8213
GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE   = 0x8212
GL_FRAMEBUFFER_COMPLETE              = 0x8CD5
GL_FRONT                             = 0x404
GL_FUNC_ADD                          = 0x8006
GL_GREATER                           = 0x204
GL_GREEN                             = 0x1904
GL_INFO_LOG_LENGTH                   = 0x8B84
GL_INVALID_ENUM                      = 0x500
GL_INVALID_FRAMEBUFFER_OPERATION     = 0x506
GL_INVALID_OPERATION                 = 0x502
GL_INVALID_VALUE                     = 0x501
GL_LEQUAL                            = 0x203
GL_LESS                              = 0x201
GL_LINEAR                            = 0x2601
GL_LINK_STATUS                       = 0x8B82
GL_MAX                               = 0x8008
GL_MAX_3D_TEXTURE_SIZE               = 0x8073
GL_MAX_RENDERBUFFER_SIZE             = 0x84E8
GL_MAX_TEXTURE_SIZE                  = 0xD33
GL_MAX_UNIFORM_BLOCK_SIZE            = 0x8A30
GL_NEAREST                           = 0x2600
GL_NONE                              = 0x0
GL_NO_ERROR                          = 0x0
GL_NUM_EXTENSIONS                    = 0x821D
GL_ONE                               = 0x1
GL_ONE_MINUS_SRC_ALPHA               = 0x303
GL_OUT_OF_MEMORY                     = 0x505
GL_POINT_SIZE_RANGE                  = 0xB12
GL_R11F_G11F_B10F                    = 0x8C3A
GL_R16                               = 0x822A
GL_R16_SNORM                         = 0x8F98
GL_R32F                              = 0x822E
GL_R8                                = 0x8229
GL_R8_SNORM                          = 0x8F94
GL_READ_FRAMEBUFFER                  = 0x8CA8
GL_RED                               = 0x1903
GL_RENDERBUFFER                      = 0x8D41
GL_RENDERER                          = 0x1F01
GL_RG                                = 0x8227
GL_RG16                              = 0x822C
GL_RG16_SNORM                        = 0x8F99
GL_RG32F                             = 0x8230
GL_RG8                               = 0x822B
GL_RG8_SNORM                         = 0x8F95
GL_RGB                               = 0x1907
GL_RGB16                             = 0x8054
GL_RGB16F                            = 0x881B
GL_RGB16_SNORM                       = 0x8F9A
GL_RGB32F                            = 0x8815
GL_RGB8                              = 0x8051
GL_RGB8_SNORM                        = 0x8F96
GL_RGBA                              = 0x1908
GL_RGBA16                            = 0x805B
GL_RGBA16F                           = 0x881A
GL_RGBA16_SNORM                      = 0x8F9B
GL_RGBA32F                           = 0x8814
GL_RGBA8                             = 0x8058
GL_RGBA8_SNORM                       = 0x8F97
GL_SHADING_LANGUAGE_VERSION          = 0x8B8C
GL_SHORT                             = 0x1402
GL_SRC_ALPHA                         = 0x302
GL_SRGB8_ALPHA8                      = 0x8C43
GL_STACK_OVERFLOW                    = 0x503
GL_STACK_UNDERFLOW                   = 0x504
GL_STATIC_DRAW                       = 0x88E4
GL_STEREO                            = 0xC33
GL_TEXTURE0                          = 0x84C0
GL_TEXTURE_1D                        = 0xDE0
GL_TEXTURE_2D                        = 0xDE1
GL_TEXTURE_3D                        = 0x806F
GL_TEXTURE_BORDER_COLOR              = 0x1004
GL_TEXTURE_COMPARE_FUNC              = 0x884D
GL_TEXTURE_COMPARE_MODE              = 0x884C
GL_TEXTURE_CUBE_MAP                  = 0x8513
GL_TEXTURE_CUBE_MAP_POSITIVE_X       = 0x8515
GL_TEXTURE_MAG_FILTER                = 0x2800
GL_TEXTURE_MIN_FILTER                = 0x2801
GL_TEXTURE_SWIZZLE_A                 = 0x8E45
GL_TEXTURE_SWIZZLE_B                 = 0x8E44
GL_TEXTURE_SWIZZLE_G                 = 0x8E43
GL_TEXTURE_WRAP_R                    = 0x8072
GL_TEXTURE_WRAP_S                    = 0x2802
GL_TEXTURE_WRAP_T                    = 0x2803
GL_TRUE                              = 0x1
GL_UNIFORM_BUFFER                    = 0x8A11
GL_UNPACK_ALIGNMENT                  = 0xCF5
GL_UNSIGNED_BYTE                     = 0x1401
GL_UNSIGNED_INT                      = 0x1405
GL_UNSIGNED_SHORT                    = 0x1403
GL_VALIDATE_STATUS                   = 0x8B83
GL_VENDOR                            = 0x1F00
GL_VERSION                           = 0x1F02
GL_VERTEX_SHADER                     = 0x8B31
GL_XOR                               = 0x1506

ERROR_CHECKING = True

_ERROR_NAMES = {
    GL_INVALID_ENUM: 'invalid enumerant',
    GL_INVALID_VALUE: 'invalid value',
    GL_INVALID_OPERATION: 'invalid operation',
    GL_STACK_OVERFLOW: 'stack overflow',
    GL_STACK_UNDERFLOW: 'stack underflow',
    GL_OUT_OF_MEMORY: 'out of memory',
    GL_INVALID_FRAMEBUFFER_OPERATION: 'invalid framebuffer operation',
}

class GLError(RuntimeError):
    '''An OpenGL call set the OpenGL error flag.'''
    def __init__(self, err, function, args):
        self.err = err
        self.function = function
        self.args_ = args
        msg = 'OpenGL error 0x%X (%s) from %s%r' % (err, _ERROR_NAMES.get(err, 'unknown'), function, tuple(args))
        RuntimeError.__init__(self, msg)

class NullFunctionError(GLError):
    def __init__(self, function):
        RuntimeError.__init__(self, 'OpenGL function %s is not available in the current context' % function)

# ctypes types for OpenGL argument types
_TYPES = {
    'e': c_uint,		# GLenum, GLbitfield
    'u': c_uint,		# GLuint
    'i': c_int,			# GLint, GLsizei
    'b': c_ubyte,		# GLboolean
    'f': c_float,
    'd': c_double,
    'p': c_void_p,		# pointer
    'z': c_ssize_t,		# GLintptr, GLsizeiptr
    's': c_char_p,
}

# name: (return type, argument types)
_SIGNATURES = {
    'glActiveTexture': (None, 'e'),
    'glAttachShader': (None, 'uu'),
    'glBindBuffer': (None, 'eu'),
    'glBindBufferBase': (None, 'euu'),
    'glBindFramebuffer': (None, 'eu'),
    'glBindRenderbuffer': (None, 'eu'),
    'glBindTexture': (None, 'eu'),
    'glBindVertexArray': (None, 'u'),
    'glBlendColor': (None, 'ffff'),
    'glBlendEquation': (None, 'e'),
    'glBlendFunc': (None, 'ee'),
    'glBlendFuncSeparate': (None, 'eeee'),
    'glBlitFramebuffer': (None, 'iiiiiiiiee'),
    'glBufferData': (None, 'ezpe'),
    'glBufferSubData': (None, 'ezzp'),
    'glCheckFramebufferStatus': ('e', 'e'),
    'glClear': (None, 'e'),
    'glClearColor': (None, 'ffff'),
    'glClearDepth': (None, 'd'),
    'glColorMask': (None, 'bbbb'),
    'glCompileShader': (None, 'u'),
    'glCopyTexImage2D': (None, 'eieiiiii'),
    'glCreateProgram': ('u', ''),
    'glCreateShader': ('u', 'e'),
    'glDeleteBuffers': (None, 'ip'),
    'glDeleteFramebuffers': (None, 'ip'),
    'glDeleteRenderbuffers': (None, 'ip'),
    'glDeleteShader': (None, 'u'),
    'glDeleteTextures': (None, 'ip'),
    'glDeleteVertexArrays': (None, 'ip'),
    'glDepthFunc': (None, 'e'),
    'glDepthMask': (None, 'b'),
    'glDepthRange': (None, 'dd'),
    'glDisable': (None, 'e'),
    'glDisableVertexAttribArray': (None, 'u'),
    'glDrawBuffer': (None, 'e'),
    'glDrawElements': (None, 'eiep'),
    'glDrawElementsInstanced': (None, 'eiepi'),
    'glEnable': (None, 'e'),
    'glEnableVertexAttribArray': (None, 'u'),
    'glFinish': (None, ''),
    'glFlush': (None, ''),
    'glFramebufferRenderbuffer': (None, 'eeeu'),
    'glFramebufferTexture': (None, 'eeui'),
    'glFramebufferTexture2D': (None, 'eeeui'),
    'glFramebufferTextureLayer': (None, 'eeuii'),
    'glGenBuffers': (None, 'ip'),
    'glGenFramebuffers': (None, 'ip'),
    'glGenRenderbuffers': (None, 'ip'),
    'glGenTextures': (None, 'ip'),
    'glGenVertexArrays': (None, 'ip'),
    'glGetBooleanv': (None, 'ep'),
    'glGetError': ('e', ''),
    'glGetFramebufferAttachmentParameteriv': (None, 'eeep'),
    'glGetIntegerv': (None, 'ep'),
    'glGetProgramInfoLog': (None, 'uipp'),
    'glGetProgramiv': (None, 'uep'),
    'glGetShaderInfoLog': (None, 'uipp'),
    'glGetShaderiv': (None, 'uep'),
    'glGetString': ('s', 'e'),
    'glGetStringi': ('s', 'eu'),
    'glGetTexImage': (None, 'eieep'),
    'glGetUniformBlockIndex': ('u', 'us'),
    'glGetUniformLocation': ('i', 'us'),
    'glIsEnabled': ('b', 'e'),
    'glLinkProgram': (None, 'u'),
    'glLogicOp': (None, 'e'),
    'glPixelStorei': (None, 'ei'),
    'glPointSize': (None, 'f'),
    'glReadBuffer': (None, 'e'),
    'glReadPixels': (None, 'iiiieep'),
    'glRenderbufferStorage': (None, 'eeii'),
    'glShaderSource': (None, 'uipp'),
    'glTexImage1D': (None, 'eiiiieep'),
    'glTexImage2D': (None, 'eiiiiieep'),
    'glTexImage3D': (None, 'eiiiiiieep'),
    'glTexParameterfv': (None, 'eep'),
    'glTexParameteri': (None, 'eei'),
    'glTexSubImage1D': (None, 'eiiieep'),
    'glTexSubImage2D': (None, 'eiiiiieep'),
    'glTexSubImage3D': (None, 'eiiiiiiieep'),
    'glUniform1f': (None, 'if'),
    'glUniform1i': (None, 'ii'),
    'glUniform2fv': (None, 'iip'),
    'glUniform3fv': (None, 'iip'),
    'glUniform4fv': (None, 'iip'),
    'glUniformBlockBinding': (None, 'uuu'),
    'glUniformMatrix4fv': (None, 'iibp'),
    'glUseProgram': (None, 'u'),
    'glValidateProgram': (None, 'u'),
    'glVertexAttribDivisor': (None, 'uu'),
    'glVertexAttribPointer': (None, 'uiebip'),
    'glViewport': (None, 'iiii'),
}

_cfuncs = {}

def _cfunc(name):
    '''ctypes function for an OpenGL entry point in the current context.'''
    f = _cfuncs.get(name)
    if f is None:
        from Qt.QtGui import QOpenGLContext
        c = QOpenGLContext.currentContext()
        if c is None:
            raise NullFunctionError(name)
        address = c.getProcAddress(name.encode('ascii'))
        if not address:
            raise NullFunctionError(name)
        rtype, atypes = _SIGNATURES[name]
        proto = ctypes.CFUNCTYPE(None if rtype is None else _TYPES[rtype],
                                 *[_TYPES[t] for t in atypes])
        _cfuncs[name] = f = proto(address)
    return f

def _check(name, args):
    e = _cfunc('glGetError')()
    if e != GL_NO_ERROR:
        raise GLError(e, name, args)

def _call(name, *args):
    r = _cfunc(name)(*args)
    if ERROR_CHECKING:
        _check(name, args)
    return r

def _simple(name):
    def call(*args):
        r = _cfunc(name)(*args)
        if ERROR_CHECKING:
            _check(name, args)
        return r
    call.__name__ = call.__qualname__ = name
    return call

for _name in ('glActiveTexture', 'glAttachShader', 'glBindBuffer', 'glBindBufferBase',
              'glBindFramebuffer', 'glBindRenderbuffer', 'glBindTexture', 'glBindVertexArray',
              'glBlendColor', 'glBlendEquation', 'glBlendFunc', 'glBlendFuncSeparate',
              'glBlitFramebuffer', 'glCheckFramebufferStatus', 'glClear', 'glClearColor',
              'glClearDepth', 'glColorMask', 'glCompileShader', 'glCopyTexImage2D',
              'glCreateProgram', 'glCreateShader', 'glDeleteShader', 'glDepthFunc', 'glDepthMask',
              'glDepthRange', 'glDisable', 'glDisableVertexAttribArray', 'glDrawBuffer', 'glEnable',
              'glEnableVertexAttribArray', 'glFinish', 'glFlush', 'glFramebufferRenderbuffer',
              'glFramebufferTexture', 'glFramebufferTexture2D', 'glFramebufferTextureLayer',
              'glIsEnabled', 'glLinkProgram', 'glLogicOp', 'glPixelStorei', 'glPointSize',
              'glReadBuffer', 'glRenderbufferStorage', 'glTexParameteri', 'glUniform1f',
              'glUniform1i', 'glUniformBlockBinding', 'glUseProgram', 'glValidateProgram',
              'glVertexAttribDivisor', 'glViewport'):
    globals()[_name] = _simple(_name)
del _name

def glGetError():
    return _cfunc('glGetError')()

def _pyopengl():
    import OpenGL.GL
    return OpenGL.GL

def __getattr__(name):
    '''Names not defined here come from PyOpenGL.'''
    if name.startswith('__'):
        raise AttributeError(name)
    value = getattr(_pyopengl(), name)
    globals()[name] = value
    return value

#
# Pointer arguments.
#
def _pointer(data):
    '''
    Return a value to pass as a void* argument and an object that must be
    kept alive during the call, or None if PyOpenGL should handle data.
    '''
    if data is None:
        return None, None
    if isinstance(data, (c_void_p, int)):
        return data, None
    from numpy import ndarray, ascontiguousarray
    if isinstance(data, ndarray):
        a = ascontiguousarray(data)
        return a.ctypes.data, a
    if isinstance(data, (bytes, ctypes.Array)):
        return data, data
    return _UNHANDLED, None

_UNHANDLED = object()

def _float_array(values):
    from numpy import ascontiguousarray, float32
    return ascontiguousarray(values, float32)

#
# Object names.
#
def _gen(name, n):
    from numpy import empty, uint32
    ids = empty((n,), uint32)
    _call(name, n, ids.ctypes.data)
    return ids[0] if n == 1 else ids

def glGenBuffers(n):
    return _gen('glGenBuffers', n)
def glGenFramebuffers(n):
    return _gen('glGenFramebuffers', n)
def glGenRenderbuffers(n):
    return _gen('glGenRenderbuffers', n)
def glGenTextures(n):
    return _gen('glGenTextures', n)
def glGenVertexArrays(n):
    return _gen('glGenVertexArrays', n)

def _delete(name, n, ids):
    from numpy import array, uint32
    a = array(ids, uint32).reshape(-1)
    _call(name, n, a.ctypes.data)

def glDeleteBuffers(n, ids):
    _delete('glDeleteBuffers', n, ids)
def glDeleteFramebuffers(n, ids):
    _delete('glDeleteFramebuffers', n, ids)
def glDeleteRenderbuffers(n, ids):
    _delete('glDeleteRenderbuffers', n, ids)
def glDeleteVertexArrays(n, ids):
    _delete('glDeleteVertexArrays', n, ids)
def glDeleteTextures(*args):
    # PyOpenGL takes glDeleteTextures(ids) or glDeleteTextures(n, ids).
    if len(args) == 1:
        ids = args[0]
        n = len(ids) if hasattr(ids, '__len__') else 1
    else:
        n, ids = args
    _delete('glDeleteTextures', n, ids)

#
# Queries.
#
def glGetString(name):
    return _call('glGetString', name)

def glGetStringi(name, index):
    return _call('glGetStringi', name, index)

# Queries that return one value, answered without PyOpenGL's size table.
_SCALAR_GETS = frozenset((GL_MAX_TEXTURE_SIZE, GL_MAX_3D_TEXTURE_SIZE, GL_MAX_RENDERBUFFER_SIZE,
                          GL_MAX_UNIFORM_BLOCK_SIZE, GL_NUM_EXTENSIONS, GL_CONTEXT_PROFILE_MASK,
                          GL_STEREO))

def glGetIntegerv(pname, *args):
    if args or pname not in _SCALAR_GETS:
        return _pyopengl().glGetIntegerv(pname, *args)
    from numpy import zeros, int32
    v = zeros((1,), int32)
    _call('glGetIntegerv', pname, v.ctypes.data)
    return v[0]
glGetInteger = glGetIntegerv

def glGetBooleanv(pname, *args):
    if args or pname not in _SCALAR_GETS:
        return _pyopengl().glGetBooleanv(pname, *args)
    from numpy import zeros, uint8
    v = zeros((1,), uint8)
    _call('glGetBooleanv', pname, v.ctypes.data)
    return v[0]
glGetBoolean = glGetBooleanv

def _get_iv(name, *args):
    from numpy import zeros, int32
    v = zeros((1,), int32)
    _call(name, *args, v.ctypes.data)
    return v[0]

def glGetProgramiv(program, pname, *args):
    if args:
        return _pyopengl().glGetProgramiv(program, pname, *args)
    return _get_iv('glGetProgramiv', program, pname)

def glGetShaderiv(shader, pname, *args):
    if args:
        return _pyopengl().glGetShaderiv(shader, pname, *args)
    return _get_iv('glGetShaderiv', shader, pname)

def glGetFramebufferAttachmentParameteriv(target, attachment, pname, *args):
    if args:
        return _pyopengl().glGetFramebufferAttachmentParameteriv(target, attachment, pname, *args)
    return _get_iv('glGetFramebufferAttachmentParameteriv', target, attachment, pname)

def _info_log(get_iv, get_log, obj):
    # Like PyOpenGL: an empty log is the str '', otherwise bytes.
    n = int(get_iv(obj, GL_INFO_LOG_LENGTH))
    if n <= 0:
        return ''
    buf = ctypes.create_string_buffer(n)
    _call(get_log, obj, n, None, buf)
    return buf.value.strip(b'\0')

def glGetProgramInfoLog(program):
    return _info_log(glGetProgramiv, 'glGetProgramInfoLog', program)

def glGetShaderInfoLog(shader):
    return _info_log(glGetShaderiv, 'glGetShaderInfoLog', shader)

def _name_arg(name):
    return name.encode('utf-8') if isinstance(name, str) else name

def glGetUniformLocation(program, name):
    return _call('glGetUniformLocation', program, _name_arg(name))

def glGetUniformBlockIndex(program, name):
    return _call('glGetUniformBlockIndex', program, _name_arg(name))

#
# Buffers.
#
def glBufferData(target, *args):
    # PyOpenGL takes (target, size, data, usage) or (target, data, usage).
    if len(args) == 3:
        size, data, usage = args
    else:
        data, usage = args
        size = getattr(data, 'nbytes', None)
        if size is None:
            return _pyopengl().glBufferData(target, *args)
    p, keep = _pointer(data)
    if p is _UNHANDLED:
        return _pyopengl().glBufferData(target, *args)
    _call('glBufferData', target, size, p, usage)

def glBufferSubData(target, offset, *args):
    # PyOpenGL takes (target, offset, size, data) or (target, offset, data).
    if len(args) == 2:
        size, data = args
    else:
        data, = args
        size = getattr(data, 'nbytes', None)
        if size is None:
            return _pyopengl().glBufferSubData(target, offset, *args)
    p, keep = _pointer(data)
    if p is _UNHANDLED:
        return _pyopengl().glBufferSubData(target, offset, *args)
    _call('glBufferSubData', target, offset, size, p)

def glVertexAttribPointer(index, size, type, normalized, stride, pointer):
    if pointer is not None and not isinstance(pointer, (c_void_p, int)):
        # Client-side arrays.
        return _pyopengl().glVertexAttribPointer(index, size, type, normalized, stride, pointer)
    _call('glVertexAttribPointer', index, size, type, normalized, stride, pointer)

def glDrawElements(mode, count, type, indices):
    if indices is not None and not isinstance(indices, (c_void_p, int)):
        return _pyopengl().glDrawElements(mode, count, type, indices)
    _call('glDrawElements', mode, count, type, indices)

def glDrawElementsInstanced(mode, count, type, indices, instancecount):
    if indices is not None and not isinstance(indices, (c_void_p, int)):
        return _pyopengl().glDrawElementsInstanced(mode, count, type, indices, instancecount)
    _call('glDrawElementsInstanced', mode, count, type, indices, instancecount)

#
# Textures and pixels.
#
def _with_data(name, args):
    *rest, data = args
    p, keep = _pointer(data)
    if p is _UNHANDLED:
        return getattr(_pyopengl(), name)(*args)
    _call(name, *rest, p)

def glTexImage1D(*args):
    return _with_data('glTexImage1D', args)
def glTexImage2D(*args):
    return _with_data('glTexImage2D', args)
def glTexImage3D(*args):
    return _with_data('glTexImage3D', args)
def glTexSubImage1D(*args):
    return _with_data('glTexSubImage1D', args)
def glTexSubImage2D(*args):
    return _with_data('glTexSubImage2D', args)
def glTexSubImage3D(*args):
    return _with_data('glTexSubImage3D', args)

def _read_into(name, args, nargs):
    # Fill a caller's contiguous numpy array; otherwise let PyOpenGL allocate.
    from numpy import ndarray
    if len(args) == nargs + 1 and isinstance(args[-1], ndarray) and args[-1].flags.c_contiguous:
        a = args[-1]
        _call(name, *args[:-1], a.ctypes.data)
        return a
    return getattr(_pyopengl(), name)(*args)

def glReadPixels(*args):
    return _read_into('glReadPixels', args, 6)

def glGetTexImage(*args):
    return _read_into('glGetTexImage', args, 4)

def glTexParameterfv(target, pname, params):
    a = _float_array(params)
    _call('glTexParameterfv', target, pname, a.ctypes.data)

#
# Uniforms.
#
def glUniform2fv(location, count, value):
    a = _float_array(value)
    _call('glUniform2fv', location, count, a.ctypes.data)

def glUniform3fv(location, count, value):
    a = _float_array(value)
    _call('glUniform3fv', location, count, a.ctypes.data)

def glUniform4fv(location, count, value):
    a = _float_array(value)
    _call('glUniform4fv', location, count, a.ctypes.data)

def glUniformMatrix4fv(location, count, transpose, value):
    a = _float_array(value)
    _call('glUniformMatrix4fv', location, count, transpose, a.ctypes.data)

#
# Shaders.
#
def glShaderSource(shader, source):
    if isinstance(source, (str, bytes)):
        source = [source]
    strings = [s.encode('utf-8') if isinstance(s, str) else s for s in source]
    n = len(strings)
    ptrs = (c_char_p * n)(*strings)
    lengths = (c_int * n)(*[len(s) for s in strings])
    _call('glShaderSource', shader, n, ptrs, lengths)

class ShaderCompilationError(RuntimeError):
    pass

class shaders:
    '''The part of PyOpenGL's OpenGL.GL.shaders that ChimeraX uses.'''
    @staticmethod
    def compileShader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        result = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if result == GL_FALSE:
            raise ShaderCompilationError(
                'Shader compile failure (%s): %s' % (result, glGetShaderInfoLog(shader)),
                source, shader_type)
        return shader
