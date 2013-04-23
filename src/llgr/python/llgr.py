"""
llgr: Provide access to low-level graphics library
--------------------------------------------------

Provide a Python interface to the C++ :doc:`/devel/llgr`.
The interface is primarily documented in the C++ interface,
and the corresponding Python function signatures are given below.
Since the interface is a wrapped C++ interface,
the argument types are important.

The interface is extended to optionally manage Ids
with the :py:func:`next_data_id`, :py:func:`next_matrix_id`,
:py:func:`next_object_id`, :py:func:`next_program_id`,
and :py:func:`next_texture_id` functions.

Python interface
^^^^^^^^^^^^^^^^
"""
__all__ = [
	## from this module
	"next_data_id",
	"next_matrix_id",
	"next_object_id",
	"next_program_id",
	#"next_texture_id",
	## from _llgr
	# functions
	"add_cylinder", "add_sphere",
	"clear_all", "clear_buffers", "clear_matrices",
	"clear_objects", "clear_primitives", "clear_programs",
	"create_buffer", "create_matrix", "create_object",
	"create_program", "create_singleton",
	"delete_buffer", "delete_matrix", "delete_object", "delete_program",
	"hide_objects", "opaque", "pick", "render",
	"selection_add", "selection_clear", "selection_remove",
	"set_clear_color", "set_uniform", "set_uniform_matrix",
	"show_objects", "transparent",
	"vsphere_drag", "vsphere_press", "vsphere_release", "vsphere_setup",
	# classes
	"AttributeInfo",
	# enums
	"Byte", "UByte", "Short", "UShort", "Int", "UInt", "Float",
	"IVec1", "IVec2", "IVec3", "IVec4",
	"UVec1", "UVec2", "UVec3", "UVec4",
	"FVec1", "FVec2", "FVec3", "FVec4",
	"Mat2x2", "Mat3x3", "Mat4x4",
	"Mat2x3", "Mat3x2", "Mat2x4", "Mat4x2", "Mat3x4", "Mat4x3",
	"ARRAY",
	"ELEMENT_ARRAY",
	"Points",
	"Lines",
	"Line_loop",
	"Line_strip",
	"Triangles",
	"Triangle_strip",
	"Triangle_fan",
]
from _llgr import (
	add_cylinder, add_sphere,
	clear_all, clear_buffers, clear_matrices,
	clear_objects, clear_primitives, clear_programs,
	create_buffer, create_matrix, create_object,
	create_program, create_singleton,
	delete_buffer, delete_matrix, delete_object, delete_program,
	hide_objects, opaque, pick, render,
	selection_add, selection_clear, selection_remove,
	set_clear_color, set_uniform, set_uniform_matrix,
	show_objects, transparent,
	vsphere_drag, vsphere_press, vsphere_release, vsphere_setup,
	AttributeInfo,
	Byte, UByte, Short, UShort, Int, UInt, Float,
	IVec1, IVec2, IVec3, IVec4,
	UVec1, UVec2, UVec3, UVec4,
	FVec1, FVec2, FVec3, FVec4,
	Mat2x2, Mat3x3, Mat4x4,
	Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3,
	ARRAY,
	ELEMENT_ARRAY,
	Points,
	Lines,
	Line_loop,
	Line_strip,
	Triangles,
	Triangle_strip,
	Triangle_fan,
)

import itertools

# next line is for pyflakes
_data_id, _matrix_id, _object_id, _program_id, _texture_id = 0, 0, 0, 0, 0
def _init():
	global _data_id, _matrix_id, _object_id, _program_id, _texture_id
	_data_id = itertools.count(start=1)
	_matrix_id = itertools.count(start=1)
	_object_id = itertools.count(start=1)
	_program_id = itertools.count(start=1)
	_texture_id = itertools.count(start=1)
_init()

_clear_programs = clear_programs
def clear_programs():
	global _program_id
	_clear_programs()
	_program_id = itertools.count(start=1)

_clear_buffers = clear_buffers
def clear_buffers():
	global _data_id
	_clear_buffers()
	_data_id = itertools.count(start=1)

"""TODO:
_clear_textures = clear_textures
def clear_textures():
	global _texture_id
	_clear_texture()
	_texture_id = itertools.count(start=1)
"""

_clear_matrices = clear_matrices
def clear_matrices():
	global _matrix_id
	_clear_matrices()
	_matrix_id = itertools.count(start=1)

_clear_objects = clear_objects
def clear_objects():
	global _object_id
	_clear_objects()
	_object_id = itertools.count(start=1)

_clear_all = clear_all
def clear_all():
	_clear_all()
	_init()

def next_data_id():
	"""Return next available integer data id"""
	return _data_id.next()

def next_matrix_id():
	"""Return next available integer matrix id"""
	return _matrix_id.next()

def next_object_id():
	"""Return next available integer object id"""
	return _object_id.next()

def next_program_id():
	"""Return next available integer program id"""
	return _program_id.next()

#def next_texture_id():
#	"""Return next available integer data id"""
#	return _texture_id.next()
