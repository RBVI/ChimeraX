"""
bild: bild format support
=========================

Read a subset of Chimera's
`bild format <http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/bild.html>`_:
.comment, .color, .transparency, .sphere, .cylinder, .arrow, .box,
.pop, .rotate, .scale, .translate.

The plan is to suport all of the existing bild format.
"""

from chimera2 import generic3d
from chimera2.cli import UserError
from chimera2.math3d import Point, Xform, Identity
from math import radians

def is_int(i):
	try:
		int(i)
		return True
	except ValueError:
		return False

def interp(t, a, b):
	c = list(a)
	for i in range(3):
		c[i] += t * (float(b[i]) - float(a[i]))
	return c

def RGB_color(color_number):
	# backwards compatible Midas colors
	if color_number == 0:
		return (1, 1, 1)
	if color_number < 9:
		return interp((color_number - 1) / 8.0, (0, 1, 0), (0, 1, 1))
	if color_number < 17:
		return interp((color_number - 8) / 8.0, (0, 1, 1), (0, 0, 1))
	if color_number < 25:
		return interp((color_number - 16) / 8.0, (0, 0, 1), (1, 0, 1))
	if color_number < 33:
		return interp((color_number - 24) / 8.0, (1, 0, 1), (1, 0, 0))
	if color_number < 49:
		return interp((color_number - 32) / 16.0, (1, 0, 0), (1, 1, 0))
	if color_number < 65:
		return interp((color_number - 48) / 16.0, (1, 1, 0), (0, 0, 0))
	if color_number == 65:
		return (0.7, 0.7, 0.7)
	raise ValueError("color number must be from 0 to 65 inclusive")

_builtin_open = open

def open(stream, *args, **kw):
	"""Populate the scene with the geometry from a bild file
	

	:param stream: either a binary I/O stream or the name of a file

	Extra arguments are ignored.
	"""

	if hasattr(stream, 'read'):
		input = stream
	else:
		# it's really a filename
		input = _builtin_open(stream, 'rb')

	model = generic3d.Generic3D()
	model.make_graphics()

	# parse input
	warned = set()
	transforms = [Identity()]
	cur_color = [1.0, 1.0, 1.0, 1.0]
	lineno = 0
	num_objects = 0
	# unknown encoding, assume UTF-8
	for line in input.readlines():
		lineno += 1
		line = line.decode('utf-8', 'ignore').rstrip()
		tokens = line.split()
		if tokens[0] in ('.c', '.comment'):
			pass
		elif tokens[0] == '.color':
			try:
				if len(tokens) == 2:
					if is_int(tokens[1]):
						cur_color[0:3] = RGB_color(int(tokens[1]))
					else:
						from chimera2.color import Color_arg
						cur_color[0:3] = Color_arg.parse(tokens[1]).rgb
				elif len(tokens) != 4:
					raise UserError("expected R, G, B values after .color on line %d" % lineno)
				else:
					cur_color[0:3] = [float(x) for x in tokens[1:4]]
			except ValueError as e:
				raise UserError("%s on line %d" % (e, lineno))
		elif tokens[0] == '.transparency':
			if len(tokens) != 2:
				raise UserError("expected value after .transparency on line %d" % lineno)
			cur_color[3] = 1 - float(tokens[1])
		elif tokens[0] == '.sphere':
			if len(tokens) != 5:
				raise UserError("expected x y z r after .sphere on line %d" % lineno)
			data = [float(x) for x in tokens[1:5]]
			center = Point(data[0:3])
			radius = data[3]
			model.graphics.add_sphere(radius, center, cur_color, transforms[-1])
			num_objects += 1
		elif tokens[0] == '.cylinder':
			if len(tokens) not in (8, 9) or (len(tokens) == 9
						and tokens[8] != 'open'):
				raise UserError("expected x1 y1 z1 x2 y2 z2 r [open] after .cylinder on line %d" % lineno)
			data = [float(x) for x in tokens[1:8]]
			p0 = Point(data[0:3])
			p1 = Point(data[3:6])
			radius = data[6]
			if len(tokens) < 9:
				bottom = top = True
			else:
				bottom = top = False
			model.graphics.add_cylinder(radius, p0, p1, cur_color,
				bottom=bottom, top=top, xform=transforms[-1])
			num_objects += 1
		elif tokens[0] == '.box':
			if len(tokens) != 7:
				raise UserError("expected x1 y1 z1 x2 y2 z2 after .box on line %d" % lineno)
			data = [float(x) for x in tokens[1:7]]
			p0 = Point(data[0:3])
			p1 = Point(data[3:6])
			model.graphics.add_box(p0, p1, cur_color, transforms[-1])
			num_objects += 1
		elif tokens[0] == '.arrow':
			if len(tokens) not in (7, 8, 9, 10):
				raise UserError("expected x1 y1 z1 x2 y2 z2 [r1 [r2 [rho]]] after .arrow on line %d" % lineno)
			data = [float(x) for x in tokens[1:]]
			r1 = data[6] if len(tokens) > 7 else 0.1
			r2 = data[7] if len(tokens) > 8 else 4 * r1
			rho = data[8] if len(tokens) > 9 else 0.75
			p1 = Point(data[0:3])
			p2 = Point(data[3:6])
			junction = p1 + rho * (p2 - p1)
			model.graphics.add_cylinder(r1, p1, junction, cur_color,
				bottom=True, top=False, xform=transforms[-1])
			model.graphics.add_cone(r2, junction, p2, cur_color, bottom=True, xform=transforms[-1])
			num_objects += 1
		elif tokens[0] == '.pop':
			if len(transforms) == 1:
				raise UserError("empty transformation stack on line %d" % lineno)
			transforms.pop()
		elif tokens[0] in ('.tran', '.translate'):
			if len(tokens) != 4:
				raise UserError("expected x y z after %s on line %d" % (tokens[0], lineno))
			data = [float(x) for x in tokens[1:4]]
			xform = Xform(transforms[-1])
			xform.translate(data)
			transforms.append(xform)
		elif tokens[0] in ('.rot', '.rotate'):
			if len(tokens) not in (3, 5):
				raise UserError("expected angle axis after %s on line %d" % (tokens[0], lineno))
			if len(tokens) == 3:
				angle = float(tokens[1])
				if tokens[2] == 'x':
					axis = (1., 0., 0.)
				elif tokens[2] == 'y':
					axis = (1., 0., 0.)
				elif tokens[2] == 'z':
					axis = (1., 0., 0.)
				else:
					raise UserError("unknown axis on line %d" % lineno)
			else:
				data = [float(x) for x in tokens[1:5]]
				angle = data[0]
				axis = data[1:4]
			angle = radians(angle)
			xform = Xform(transforms[-1])
			xform.rotate(axis, angle)
			transforms.append(xform)
		elif tokens[0] == '.scale':
			if len(tokens) not in (2, 3, 4):
				raise UserError("expected x [y [z]] after .scale on line %d" % lineno)
			data = [float(x) for x in tokens[1:]]
			if len(data) == 1:
				data.extend([data[0], data[0]])
			elif len(data) == 2:
				data.append(data[0])
			xform = Xform(transforms[-1])
			xform.scale(data)
			transforms.append(xform)
		elif tokens[0] not in warned:
			import sys
			print(tokens[0], 'is not supported on line %d' % lineno,
					file=sys.stderr)
			warned.add(tokens[0])

	if input != stream:
		input.close()

	return [model], "Opened BILD data containing %d objects" % num_objects

def register():
	from chimera2 import io
	io.register_format("BILD", generic3d.CATEGORY, (".bild",),
		reference="http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/bild.html",
		open_func=open)
