# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
llgr: Provide access to low-level graphics library
--------------------------------------------------

The llgr module provides an equivalent Python interface
to the C++ :doc:`/devel/llgr_c++` interface.
There are two alternative implementations, and one must be chosen at runtime.
Either an OpenGL interface that calls the C++ library,
or an implementation that outputs a textual representation
(see :py:func:`set_output` below).
The C++ interface documentation is considered to be the reference documentation.

In addition to the C++ interface,
the Python interface is extended to optionally manage :cpp:type:`Id`'s
with the :py:func:`next_data_id`, :py:func:`next_matrix_id`,
:py:func:`next_object_id`,
:py:func:`next_group_id`,
and :py:func:`next_program_id` functions.

Python interface
^^^^^^^^^^^^^^^^
"""

_llgr_syms = [
	## from _llgr
	# functions
	"add_cone", "add_cylinder", "add_disk", "add_sphere",
	"clear_all", "clear_buffers", "clear_matrices",
	"clear_objects", "clear_primitives", "clear_programs",
	"create_buffer", "create_matrix", "create_object",
	"create_program", "create_singleton",
	"delete_buffer", "delete_matrix", "delete_object", "delete_program",
	"hide_objects", "opaque", "render",
	"selection_add", "selection_clear", "selection_remove",
	"create_group", "delete_group", "clear_groups",
	"group_add", "group_remove", "hide_group", "show_group",
	"selection_add_group", "selection_remove_group",
	"set_clear_color", "set_uniform", "set_uniform_matrix",
	"show_objects", "transparent",
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
_llgr_ui_syms = [
	"pick",
	"vsphere_drag", "vsphere_press", "vsphere_release", "vsphere_setup",
]
_local_syms = [
	## from this module
	"set_output",
	"output_type",
	"next_data_id",
	"next_matrix_id",
	"next_object_id",
	"next_group_id",
	"next_program_id",
	#"next_texture_id",
]

# symbols that are wrapped with local versions
_wrapped_syms = set([
	"clear_all", "clear_buffers", "clear_matrices",
	"clear_objects", "clear_programs"
])

global _output_type
_output_type = None

def output_type():
	return _output_type

def set_output(type):
	"""
	Set the output type

	The type may be **opengl** for OpenGL output, or one of several
	textual types: **json**, **c++**, or **js**.
	Textual output goes to :py:data:`sys.stdout`.

	Note: this funtion must be called before calling the llgr functions,
	*i.e.*, there is no default.
	"""
	global _output_type
	import importlib
	if type == 'opengl':
		llgr = importlib.import_module('._llgr', __package__)
	elif type == 'pyopengl':
		llgr = importlib.import_module('.pyopengl', __package__)
		# since this is for prototyping, make all symbols available
		gsyms = globals()
		for sym in dir(llgr):
			if sym.startswith('__'):
				continue
			if sym in _wrapped_syms:
				gsyms['_%s' % sym] = getattr(llgr, sym)
			else:
				gsyms[sym] = getattr(llgr, sym)
		_output_type = type
		return
	else:
		llgr = importlib.import_module('.dump', __package__)
		if type not in llgr.FORMATS:
			raise ValueError('type should be one: %s, pyopengl, or opengl' % ', '.join(llgr.FORMATS))
		llgr.set_dump_format(type)
		_llgr_syms.append('Enum')
	gsyms = globals()
	for sym in _llgr_syms:
		if sym not in _wrapped_syms:
			gsyms[sym] = getattr(llgr, sym)
		else:
			gsyms['_%s' % sym] = getattr(llgr, sym)
	gsyms['__all__'] = _llgr_syms + list(_wrapped_syms) + _local_syms
	has_ui_syms = False
	for sym in _llgr_ui_syms:
		if hasattr(llgr, sym):
			gsyms[sym] = getattr(llgr, sym)
			has_ui_syms = True
		elif hasattr(gsyms, sym):
			del gsyms[sym]
	if has_ui_syms:
		gsyms['__all__'] += _llgr_ui_syms
	_output_type = type

import itertools

def _init():
	global _data_id, _matrix_id, _object_id, _group_id, _program_id, _texture_id
	_data_id = itertools.count(start=1)
	_matrix_id = itertools.count(start=1)
	_object_id = itertools.count(start=1)
	_group_id = itertools.count(start=1)
	_program_id = itertools.count(start=1)
	_texture_id = itertools.count(start=1)
_init()

def clear_programs():
	global _program_id
	_clear_programs()
	_program_id = itertools.count(start=1)

def clear_buffers():
	global _data_id
	_clear_buffers()
	_data_id = itertools.count(start=1)
	_matrix_id = itertools.count(start=1)

"""TODO:
def clear_textures():
	global _texture_id
	_clear_texture()
	_texture_id = itertools.count(start=1)
"""

def clear_matrices():
	global _matrix_id
	_clear_matrices()
	_matrix_id = itertools.count(start=1)

def clear_objects():
	global _object_id
	_clear_objects()
	_object_id = itertools.count(start=1)
	_group_id = itertools.count(start=1)

def clear_groups():
	global _group_id
	_clear_groups()
	_group_id = itertools.count(start=1)

def clear_all():
	_clear_all()
	_init()

def next_data_id():
	"""Return next available integer data id
	
	Reset when :py:func:`clear_all` or :py:func:`clear_buffers` is called.
	"""
	return next(_data_id)

def next_matrix_id():
	"""Return next available integer matrix id
	
	Reset when :py:func:`clear_all` or :py:func:`clear_matrices` is called.
	"""
	return next(_matrix_id)

def next_object_id():
	"""Return next available integer object id
	
	Reset when :py:func:`clear_all` and :py:func:`clear_objects` is called.
	"""
	return next(_object_id)

def next_group_id():
	"""Return next available integer group id
	
	Reset when :py:func:`clear_all` and :py:func:`clear_groups` is called.
	"""
	return next(_group_id)

def next_program_id():
	"""Return next available integer program id
	
	Reset when :py:func:`clear_all` or :py:func:`clear_programs` is called.
	"""
	return next(_program_id)

#def next_texture_id():
#	"""Return next available integer data id
#	return next(_texture_id)
