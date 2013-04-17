import sys
sys.path.insert(0, '../chimera2')
sys.path.insert(0, '../tests')

from math3d import *

#import one as data
#import pdbmtx_atoms as data
import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
#import pdb3cc4_atoms as data

import re
trailing_digit = re.compile(r'(\d)\b([^. ])')
one = re.compile(r'([^.])\b1f')
def cpp_repr(o, as_float=True):
	r = repr(o).replace('[', '{').replace(']', '}')
	r = re.sub(trailing_digit, r'\1f\2', r)
	r = r.replace('-0f', '0.0f')
	r = re.sub(one, '\11.0f', r)
	if r.startswith('array('):
		r = r[len('array('):-1]
	return r

def cvt_spheres():
	spheres = []
	for item in data.data:
		if item[0] != 's':
			continue
		spheres.append(item[1:])

	print "\n// spheres"
	print "Sphere spheres[] = {"
	for s in spheres:
		line = cpp_repr(s)
		print "\t%s," % line
	if not spheres:
		print "\t0"
	print "};"
	print "unsigned sphere_count =",
	if not spheres:
		print "0;"
	else:
		print "sizeof spheres / sizeof (Sphere);"

def cvt_cylinders():
	cylinders = []
	for item in data.data:
		if item[0] != 'c':
			continue
		cylinders.append(item[1:])

	print "\n// cylinders"
	print "Cylinder cylinders[] = {"
	for c in cylinders:
		line = cpp_repr(c)
		print "\t%s," % line
	if not cylinders:
		print "\t0"
	print "};"
	print "unsigned cylinder_count =",
	if not cylinders:
		print "0;"
	else:
		print "sizeof cylinders / sizeof (Cylinder);"

def cvt_triangles():
	triangles = []
	for item in data.data:
		if item[0] != 't':
			continue
		triangles.append(item[1:])

	print "// triangles"
	print "Triangles triangles = {"
	for t in triangles:
		line = cpp_repr(t)
		print "\t%s," % line
	print "};"

def cvt_camera():
	# ["cofr", x, y, z]
	# ["eyepos", x, y, z]
	# ["up", x, y, z]
	# ["persp", fov]
	# ["ortho", minx, miny, maxx, maxy]
	# ["vp", width, height, hither, yon]
	cofr = None
	eyepos = None
	up = None
	sytle = None
	for item in data.data:
		if item[0] == 'cofr':
			cofr = Point(item[1:])
		elif item[0] == 'eyepos':
			eyepos = Point(item[1:])
		elif item[0] == 'up':
			up = Vector(item[1:])
		elif item[0] == 'persp':
			style = item
		elif item[0] == 'ortho':
			style = item
		elif item[0] == 'vp':
			width, height, hither, yon = item[1:]
			vp = item[1:]
	print "int width = %d;" % width
	print "int height = %d;" % height
	print "\n// Camera"
	if style[0] == "ortho":
		projection = ortho(style[1], style[2], style[3], style[4],
				hither, yon)
	else:
		from math import radians
		rad = radians(style[1])
		projection = perspective(rad, width / float(height),
				hither, yon)
	print "float projection_matrix[16] = %s;" % cpp_repr(projection.getOpenGLMatrix())
	la = look_at(eyepos, cofr, cofr + up)
	modelview = la.getOpenGLMatrix()
	print "float modelview_matrix[16] = %s;" % cpp_repr(modelview)
	normal = list((la._matrix[0:3, 0:3]).transpose().flat)
	print "float normal_matrix[9] = %s;" % cpp_repr(normal)

# TODO: other possible conversions:
#    background, lights, points, lines, indexed lines, triangle strips

print '#include "data.h"'

cvt_camera()
sys.stdout.flush()
cvt_spheres()
sys.stdout.flush()
cvt_cylinders()
sys.stdout.flush()
