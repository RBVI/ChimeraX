"""
bild: bild format support
=========================
"""

# implement subset of Chimera's bild format
#  .comment, .color, .transparency, .sphere, and .cylinder

from . import scene
from .cmds import UserError
from .math3d import Point

_builtin_open = open

def open(filename, *args, **kw):
	if hasattr(filename, 'read'):
		# it's really a file-like object
		input = filename
	else:
		input = _builtin_open(filename, 'rU')

	# parse input
	warned = set()
	cur_color = [1.0, 1.0, 1.0, 1.0]
	for line in input.readlines():
		tokens = line.split()
		if '.comment'.startswith(tokens[0]):
			pass
		elif '.color'.startswith(tokens[0]):
			if len(tokens) != 4:
				raise UserError("expected R, G, B values after .color")
			cur_color[0:3] = [float(x) for x in tokens[1:4]]
		elif '.transparency'.startswith(tokens[0]):
			if len(tokens) != 2:
				raise UserError("expected value after .transparency")
			cur_color[3] = 1 - float(tokens[1])
		elif '.sphere'.startswith(tokens[0]):
			if len(tokens) != 5:
				raise UserError("expected x y z r after .sphere")
			data = [float(x) for x in tokens[1:5]]
			center = Point(data[0:3])
			radius = data[3]
			scene.add_sphere(radius, center, cur_color)
		elif '.cylinder'.startswith(tokens[0]):
			if len(tokens) not in (8, 9):
				raise UserError("expected x1 y1 z1 x2 y2 z2 r [open] after .cylinder")
			data = [float(x) for x in tokens[1:8]]
			p0 = Point(data[0:3])
			p1 = Point(data[3:6])
			radius = data[6]
			scene.add_cylinder(radius, p0, p1, cur_color)
		elif tokens[0] not in warned:
			import sys
			print >> sys.stderr, tokens[0], 'is not supported'
			warned.add(tokens[0])

	if input != filename:
		input.close()
