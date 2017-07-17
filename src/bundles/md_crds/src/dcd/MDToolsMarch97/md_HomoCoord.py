"""HomoCoord Hierarchy for MDTools

RCS: $Id: md_HomoCoord.py 26655 2009-01-07 22:02:30Z gregc $

Class Hierarchy:
   HomoCoord -> Coord -> Atom
             -> Vector

Utilities:
   dist(a,b) - distances between Coords or groups
   distsq(a,b) - distance squared between Coords
   angle(a,b,c,[d],[units]) - angle (a,b,c) or dihedral (a,b,c,d)
   (dist() and angle() also accept tuples of Coords as single arguments.)
"""

_RCS = "$Id: md_HomoCoord.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 1.2  2004/10/26 00:35:58  pett
# add Unicode support
#
# Revision 1.1  2004/05/17 18:43:19  pett
# as distributed
#
# Revision 0.66  1996/08/28 21:11:46  jim
# Removed automatic definition and dummy values for mass and charge in Atom.
#
# Revision 0.65  1996/05/24 01:28:47  jim
# Split into sub-modules, improved version reporting.
#

#print "- HomoCoord "+"$Revision: 1.3 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2005-08-20 00:26:36 $"[7:-11]+")"

import math
import struct
import copy
import tempfile
import os
import sys
import time

from .md_Constants import angleunits, angledefault
from numbers import Number

#
# HomoCoord class hierarchy:
#                                        HomoCoord
#                                         |     |
#                                      Coord   Vector
#                                       |
#                                     Atom
#
#    distsq(a,b)  angle(a,b,c,[d],[u])  angleconvert(angle,old,[new])
#

def _HomoCoord_downcast(x,y,z,W):
	if W == 0:
		return Vector(x,y,z)
	elif abs(W-1) < 0.0001:
		return Coord(x,y,z)
	else:
		return HomoCoord(x,y,z,W)

class HomoCoord:
	"""Homogeneous coordinates distinguish vectors and positions.

As defined in many computer graphics texts, homogeneous coordinates consist of four values: x, y, z, and W.  W is 0 for vectors and 1 for coordinates.  Downcasting to Vector and Coord is done automatically for arithmetic operations on a HomoCoord.

Data: x, y, z, W

Methods:
   a = HomoCoord(x,y,z,W)
   b = HomoCoord(x,y,z,W)
   a + b 
   a - b
   -a
   10. * a
   a * 10.
   a / 10.
   len(a) - returns 4
   a[2] - returns z
"""
	def __init__(self,x,y,z,W):
		self.x = float(x); self.y = float(y); self.z = float(z); self.W = float(W)
	def __repr__(s):
		return 'HomoCoord('+str(s.x)+','+str(s.y)+','+str(s.z)+','+str(s.W)+')'
	def __add__(s,o):
		return _HomoCoord_downcast(s.x+o.x,s.y+o.y,s.z+o.z,s.W+o.W)
	def __sub__(s,o):
		return _HomoCoord_downcast(s.x-o.x,s.y-o.y,s.z-o.z,s.W-o.W)
	def __neg__(s):
		return _HomoCoord_downcast(-s.x,-s.y,-s.z,-s.W)
	def __mul__(s,a):
		if isinstance(a, Number):
			return _HomoCoord_downcast(s.x*a,s.y*a,s.z*a,s.W*a)
		else: raise TypeError('HomoCoord multiplication by non-numeric')
	def __rmul__(s,a):
		if isinstance(a, Number):
			return _HomoCoord_downcast(s.x*a,s.y*a,s.z*a,s.W*a)
		else: raise TypeError('HomoCoord multiplication by non-numeric')
	def __truediv__(s,a):
		if isinstance(a, Number):
			return _HomoCoord_downcast(s.x/a,s.y/a,s.z/a,s.W/a)
		else: raise TypeError('HomoCoord division by non-numeric')
	def __len__(self):
		return 4
	def __getitem__(s,i):
		return (s.x,s.y,s.z,s.W)[i]

class Vector(HomoCoord):
	"""A vector has a length, dot product, and cross product.

A vector is a homogeneous coordinate with W = 0 and additional operations.

Methods:
   a = Vector(x,y,z)
   b = Vector(x,y,z)
   abs(a) - returns |a|
   a * b - dot product
   a % b - cross product
   a.unit() - returns a / |a|
"""
	def __init__(self,x,y,z):
		HomoCoord.__init__(self,x,y,z,0)
	def __abs__(s):
		return math.sqrt(s.x*s.x+s.y*s.y+s.z*s.z)
	def __mul__(s,o):
		if isinstance(o, Number):
			return HomoCoord.__mul__(s,o)
		elif o.W == 0:
			return s.x*o.x+s.y*o.y+s.z*o.z
		else: raise TypeError('Vector multiplication by non-numeric and non-Vector')
	def __mod__(s,o):
		if o.W == 0:
			return Vector(s.y*o.z-s.z*o.y,s.z*o.x-s.x*o.z,s.x*o.y-s.y*o.x)
		else: raise TypeError('Vector cross-product with non-Vector')
	def unit(s):
		a = abs(s)
		if ( a ):
			return s / a
		else:
			raise ZeroDivisionError("can't create unit vector from zero vector")
	def __repr__(s):
		return 'Vector('+str(s.x)+','+str(s.y)+','+str(s.z)+')'

class Coord(HomoCoord):
	"""A coordinate cannot be scaled.

A coordinate is a homogeneous coordinate with W = 1.

Methods:
   a = Coord(x,y,z)
   b = Coord(x,y,z)
   a.set(b) - important for subclasses

See also: dist, distsq, angle
"""
	def __init__(self,x,y,z):
		HomoCoord.__init__(self,x,y,z,1)
	def set(s,o):
		if o.W == 1:
			s.x = o.x; s.y = o.y; s.z = o.z
		else: raise TypeError('Coord set to non-Coord')
	def __repr__(s):
		return 'Coord('+str(s.x)+','+str(s.y)+','+str(s.z)+')'

def dist(a,b=None):
	if b is None and isinstance(a, tuple) and len(a) == 2:
		return math.sqrt(math.pow(a[0][0]-a[1][0],2)+math.pow(a[0][1]-a[1][1],2)+math.pow(a[0][2]-a[1][2],2))
	elif not ( hasattr(a,'atoms') or hasattr(b,'atoms') ):
		return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2)+math.pow(a[2]-b[2],2))
	else:
		if hasattr(a,'atoms'):
			al = a.atoms
		else:
			al = (a)
		if hasattr(b,'atoms'): 
			bl = b.atoms
		else:
			bl = (b)
		if len(al) > len(bl): (al,bl) = (bl,al)
		ds = 1000000000.
		for aa in al:
			for ba in bl:
				ds = min(ds,math.pow(aa[0]-ba[0],2)+math.pow(aa[1]-ba[1],2)+math.pow(aa[2]-ba[2],2))
		return math.sqrt(ds)

def distsq(a,b):
	return math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2)+math.pow(a[2]-b[2],2)

def angleconvert(angle,old,new=angledefault):
	return angle * ( angleunits[old] / angleunits[new] )

def angle(a,b=angledefault,c=angledefault,x1=angledefault,x2=angledefault):
	if isinstance(a, tuple):
		return apply(angle,a+(b,))
	if isinstance(x1, str):
		d = None
		units = x1
	else:
		d = x1
		units = x2
	if d:
		e = ((c-b)%(b-a)).unit()
		f = ((d-c)%(c-b)).unit()
		return angleconvert(math.asin((e%f)*((c-b).unit())),'rad',units)
	else:
		e = (a-b).unit()
		f = (c-b).unit()
		return angleconvert(math.acos(e*f),'rad',units)

class Atom(Coord):
	"""Holds all atom-based information.

Data: mass, charge, type, name, id, q, b, residue
      optionally: bonds, angles, dihedrals, impropers, donors, acceptors

Methods:
   a = Atom()
"""
	def __init__(self):
		Coord.__init__(self,0,0,0)
		# self.mass = 1.
		# self.charge = 0.
		self.type = '???'
		self.name = '???'
		self.id = 0
		self.q = 0.
		self.b = 0.
		self.residue = None
	def __repr__(s):
		return '< Atom '+str(s.name)+' at ('+str(s.x)+','+str(s.y)+','+str(s.z)+') >'

