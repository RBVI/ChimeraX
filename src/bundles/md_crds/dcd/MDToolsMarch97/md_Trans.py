"""Trans Hierarchy for MDTools

RCS: $Id: md_Trans.py 26655 2009-01-07 22:02:30Z gregc $

Class Hierarchy:
   Trans
"""

_RCS = "$Id: md_Trans.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 1.1  2004/05/17 18:43:19  pett
# as distributed
#
# Revision 0.65  1996/05/24 01:29:53  jim
# Split into sub-modules, improved version reporting.
#

#print "- Trans "+"$Revision: 1.2 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2005-08-20 00:26:36 $"[7:-11]+")"

import math
import struct
import copy
import tempfile
import os
import sys
import time

from .md_Constants import angleunits, angledefault
from .md_HomoCoord import *

#
# Trans class hierarchy:
#                                        Trans
#

class Trans:
	"""Transformation matrix generator.

Data: matrix

Methods:
   t = Trans([shift],[center],[axis],[angle],[units])
      NOTE: (x,y,z) or (x,y,z,1) treated as Vector, (x,y,z,0) as Coord
      shift=Vector: translate by this vector (applied last)
      shift=Coord: translate this coordinate to the origin (applied last)
      center=Coord: rotate about this coordinate
      axis=Vector: rotate around line along this direction from center
      axis=Coord: rotate around line from center to this coordinate
      angle: amount to rotate in units
   t(atom) - modify coordinates of an atom
   t(group) - modify coordinates of a group of atoms
   t(trans2) - left-multiply another transformation

See also: HomoCoord, 'angles'
"""
	def __init__(self,shift=(0.,0.,0.),center=(0.,0.,0.),axis=(0.,0.,1.),angle=0.,units=angledefault):
		angle = angleconvert(angle,units,'rad')
		if len(axis) == 4 and axis[3] == 1:
			axis = (axis[0]-center[0],axis[1]-center[1],axis[2]-center[2])
		mc = [[1.,0.,0.,-center[0]],[0.,1.,0.,-center[1]],[0.,0.,1.,-center[2]],[0.,0.,0.,1.]]
		mci = [[1.,0.,0.,center[0]],[0.,1.,0.,center[1]],[0.,0.,1.,center[2]],[0.,0.,0.,1.]]
		if ( axis[0] or axis[1] ):
			theta = 0.5*math.pi - math.atan2(axis[1],axis[0])
		else:
			theta = 0.
		mx = [[math.cos(theta),-math.sin(theta),0.,0.],[math.sin(theta),math.cos(theta),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
		mxi = [[math.cos(theta),math.sin(theta),0.,0.],[-math.sin(theta),math.cos(theta),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
		axis = (mx[0][0]*axis[0]+mx[0][1]*axis[1],mx[1][0]*axis[0]+mx[1][1]*axis[1],axis[2])
		if ( axis[1] or axis[2] ):
			theta = 0.5*math.pi - math.atan2(axis[2],axis[1])
		else:
			theta = 0.
		my = [[1.,0.,0.,0.],[0.,math.cos(theta),-math.sin(theta),0.],[0.,math.sin(theta),math.cos(theta),0.],[0.,0.,0.,1.]]
		myi = [[1.,0.,0.,0.],[0.,math.cos(theta),math.sin(theta),0.],[0.,-math.sin(theta),math.cos(theta),0.],[0.,0.,0.,1.]]
		mz = [[math.cos(angle),-math.sin(angle),0.,0.],[math.sin(angle),math.cos(angle),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
		m0 = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + mc[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + mx[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + my[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + mz[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + myi[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + mxi[i][k]*m0[k][j]
		m0 = m1
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + mci[i][k]*m0[k][j]
		m0 = m1
		if len(shift) == 4 and shift[3] == 1:
			shift = (-shift[0],-shift[1],-shift[2])
		ms = [[1.,0.,0.,shift[0]],[0.,1.,0.,shift[1]],[0.,0.,1.,shift[2]],[0.,0.,0.,1.]]
		m1 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					m1[i][j] = m1[i][j] + ms[i][k]*m0[k][j]
		self.matrix = tuple(map(tuple,m1))
	def __call__(self,coord):
		if ( hasattr(coord,'x') and hasattr(coord,'y') and hasattr(coord,'z') ):
			xnew = ( self.matrix[0][0] * coord.x + self.matrix[0][1] * coord.y +
				self.matrix[0][2] * coord.z + self.matrix[0][3] )
			ynew = ( self.matrix[1][0] * coord.x + self.matrix[1][1] * coord.y +
				self.matrix[1][2] * coord.z + self.matrix[1][3] )
			znew = ( self.matrix[2][0] * coord.x + self.matrix[2][1] * coord.y +
				self.matrix[2][2] * coord.z + self.matrix[2][3] )
			coord.x = xnew
			coord.y = ynew
			coord.z = znew
		elif hasattr(coord,'atoms'):
			for a in coord.atoms:
				self(a)
		elif hasattr(coord,'residues'):
			for a in coord.residues:
				self(a)
		elif hasattr(coord,'matrix'):
			m2 = coord.matrix
			m1 = self.matrix
			m = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
			for i in range(0,4):
				for j in range(0,4):
					for k in range(0,4):
						m[i][j] = m[i][j] + m1[i][k]*m2[k][j]
			coord.matrix = tuple(map(tuple,m))
	def __repr__(self):
		str = "< Trans ("
		for i in self.matrix:
			str = str + "("
			for j in i:
				str = str + repr(j) + ","
			str = str[:-1] + "),"
		str = str[:-1] + ")"
		return str + " >"

