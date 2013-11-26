# convert webgl client data to llgr calls

import sys
import llgr
llgr.set_output('pyopengl')

DataType = llgr.DataType
PrimitiveType = llgr.PrimitiveType
ShaderType = llgr.ShaderType

class Symbolic(str):
	"""Used for C++ identifiers, where the string should not be quoted"""

	def __repr__(self):
		return self

def print_arg(arg):
	if isinstance(arg, bool):
		if arg:
			return print('true', end='')
		else:
			return print('false', end='')
	if isinstance(arg, (int, float)):
		return print(arg, end='')
	if isinstance(arg, str):
		arg = repr(arg)
		if arg[0] == "'":
			arg = arg.replace('"', '\\"')
			arg = '"%s"' % arg[1:-1]
		return print(arg, end='')
	if isinstance(arg, list) and len(arg) == 3 and isinstance(arg[0], bool) and isinstance(arg[1], int) and isinstance(arg[2], list) and ((arg[1] + 3) // 4) == len(arg[2]):
		# it's a buffer
		nbytes = arg[1]
		import struct
		fmt = 'I' * (nbytes // 4)
		fmt = '<' + fmt + ['', 'B', 'H', 'HB'][nbytes % 4]
		b = struct.pack(fmt, *arg[2])
		b = ''.join('\\x%02x' % x for x in b)
		return print('%d, "%s"' % (nbytes, b), end='')
	return print(arg, end='')

# Things to convert to symbolic ShaderType
shader_types = {
	# llgr function: nth argument
	'set_uniform': 2,
	'set_uniform_matrix': 3
}

# Things to convert to symbolic DataType
data_types = {
	'create_object': 8
}

# Things to convert to symbolic PrimitiveType
primitive_types = {
	'create_object': 4
}

def print_symbolic(value, type):
	index = type.values.index(value)
	print(type.labels[index], end='')

# Things to convert to AttributeInfos
attribute_infos = {
	'create_object': 3,
	'add_sphere': 4,
	'add_cylinder': 5,
}

# Things to convert to 4x4 matrix
matrices = {
	'create_matrix': 1
}

# Things to convert to Objects
objects = {
	'hide_objects': 0,
	'show_objects': 0,
	'transparent': 0,
	'opaque': 0,
	'selection_add': 0,
	'selection_remove': 0,
	'create_group': 1,
}

num_attrinfos = 0
num_matrices = 0
num_objects = 0

def print_call(call):
	global num_attrinfos, num_matrices, num_objects
	func = call[0]
	if len(call) > 1:
		args = call[1]
	else:
		args = []
	if func in attribute_infos:
		argnum = attribute_infos[func]
		num_attrinfos += 1
		argname = 'ais%s' % num_attrinfos
		print('AttributeInfos %s;' % argname)
		for ai in args[argnum]:
			print('%s.push_back(AttributeInfo(' % argname, end='')
			for i in range(len(ai)):
				if i > 0:
					print(', ', end='')
				if i == 5:
					print_symbolic(ai[i], DataType)
					continue
				print_arg(ai[i])
			print('));')
		args[argnum] = Symbolic(argname)
	if func in matrices:
		argnum = matrices[func]
		num_matrices += 1
		argnam = 'mat%d' % num_matrices
		print('const float %s[4][4] = { %s };' % (argnam,
				', '.join(str(x) for x in args[argnum])))
		args[argnum] = Symbolic('%s' % argnam)
	if func in objects:
		argnum = objects[func]
		num_objects += 1
		argnam = 'objs%d' % num_objects
		print('Objects %s;' % argnam)
		for i in args[argnum]:
			print("%s.push_back(%s);" % (argnam, i))
		args[argnum] = Symbolic(argnam)
	print('%s(' % func, end='')
	for i in range(len(args)):
		if i > 0:
			print(', ', end='')
		if func in data_types and data_types[func] == i:
			print_symbolic(args[i], DataType)
			continue
		if func in primitive_types and primitive_types[func] == i:
			print_symbolic(args[i], PrimitiveType)
			continue
		if func in shader_types and shader_types[func] == i:
			print_symbolic(args[i], ShaderType)
			continue
		print_arg(args[i])
	print(');')

def mk_cpp(json_filename, cpp_filename):
	import json
	f = open(json_filename)
	json_rep = json.load(f)
	save_stdout = sys.stdout
	sys.stdout = open(cpp_filename, 'w')

	print("""#include <llgr.h>

	using namespace llgr;
	""")

	width, height = 0, 0
	for line in json_rep:
		tag, data = line
		if tag == 'scene':
			wh = data['viewport']
			width = wh[0]
			height = wh[1]
	print('int width = %d;' % width)
	print('int height = %d;' % height)
	print("""
	void
	initialize()
	{""")

	for line in json_rep:
		tag, data = line
		if tag == 'llgr':
			for call in data:
				print_call(call)
			continue

	print("}")

	sys.stdout.close()
	sys.stdout = save_stdout

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("usage: %s json-filename cpp-filename" % sys.argv[0],
				file=sys.stderr)
		raise SystemExit(2)
	mk_cpp(sys.argv[1], sys.argv[2])
