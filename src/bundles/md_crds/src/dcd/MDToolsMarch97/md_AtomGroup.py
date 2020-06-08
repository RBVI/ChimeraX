"""AtomGroup Hierarchy of MDTools

RCS: $Id: md_AtomGroup.py 40844 2015-11-21 00:24:04Z pett $

Class Hierarchy:
   AtomGroup -> ASel
             -> Residue
             -> ResidueGroup -> RSel
                             -> Segment
                             -> SegmentGroup -> Molecule
"""

_RCS = "$Id: md_AtomGroup.py 40844 2015-11-21 00:24:04Z pett $"

# $Log: not supported by cvs2svn $
# Revision 1.3  2005/08/20 00:26:36  gregc
# Update for wrappy2.
#
# Revision 1.2  2004/05/17 18:46:42  pett
# handle X-PLOR acceptor lists
#
# Revision 0.70  1996/11/16 21:30:21  jim
# changed Numeric.Core to Numeric
#
# Revision 0.69  1996/10/12 20:10:54  jim
# added id field to Segment
#
# Revision 0.68  1996/10/12 18:52:09  jim
# update for NumPy1.0a4
#
# Revision 0.67  1996/08/28 20:51:23  jim
# Added masses(), charges(), and coordinate() methods.
#
# Revision 0.66  1996/08/28 19:27:59  jim
# Switched frames from lists of tuples to arrays.
#
# Revision 0.65  1996/05/24 01:20:46  jim
# Split into sub-modules, improved version reporting.
#

#print "- AtomGroup "+"$Revision: 1.4 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2007-02-07 20:59:45 $"[7:-11]+")"

import math
import struct
import copy
import tempfile
import os
import sys
import time
from numpy import array

from .md_local import pdbview
from .md_Constants import *
from .md_HomoCoord import *
from .md_Trans import *

from functools import reduce

#
# AtomGroup class hierarchy:
#                                        AtomGroup -------------
#                                         |     |              |
#                                    Residue   ResidueGroup   ASel
#                                              |    |     | 
#                                        Segment Molecule RSel
#

class AtomGroup:
	"""A group of atoms.

Data: atoms, [frames]

Methods:
   g = AtomGroup()
   g.atoms.append(a)
   g.masses() - array of masses
   g.tmass() - total mass
   g.charges() - array of charges
   g.tcharge() - total charge
   g.cgeom() - center of geometry
   g.cmass() - center of mass
   g.rgyration() - radius of gyration
   g.saveframe([key]) - save coordinates to internal dictionary
   g.loadframe([key]) - get coordinates from internal dictionary
   g.delframe([key]) - remove coordinates from internal dictionary
   frame = g.coordinates() - return array of coordinates
   g.putframe(frame) - fill array with coordinates
   g.getframe(frame) - get coordinates from array
   frame = mol.putframe()
   g.getmolframe(frame) - get coordinates from list for all atoms in molecule
   g.asel(func) - return atom selection based on filter function

See also: Molecule, ASel
"""
	def __init__(self):
		self.atoms = []
	def masses(self):
		return array([a.mass for a in self.atoms])
	def tmass(self):
		return reduce(lambda t,a: t+a.mass, self.atoms, 0.)
	def charges(self):
		return array([a.charge for a in self.atoms])
	def tcharge(self):
		return reduce(lambda t,a: t+a.charge, self.atoms, 0.)
	def cgeom(self):
		t = reduce(lambda t,a: t+a, self.atoms, Vector(0,0,0))
		return t / t.W
	def cmass(self):
		t = reduce(lambda t,a: t + a.mass*a, self.atoms, Vector(0,0,0))
		return t / t.W
	def rgyration(self):
		t = reduce(lambda t,a,com = self.cmass():
			t + a.mass*distsq(a,com), self.atoms, 0.)
		return math.sqrt( t / self.tmass() )
	def saveframe(self,key=None):
		if not hasattr(self,'frames'): self.frames = {}
		self.frames[key] = self.coordinates()
	def loadframe(self,key=None):
		if not hasattr(self,'frames') or not len(self.frames):
			raise "no frames saved internally"
		self.getframe(self.frames[key])
	def delframe(self,key=None):
		if not hasattr(self,'frames') or not len(self.frames):
			raise "no frames saved internally"
		del(self.frames[key])
	def getframe(self,frame):
		i = 0
		for a in self.atoms:
			(a.x,a.y,a.z) = tuple(frame[i])
			i = i + 1
	def getmolframe(self,frame):
		for a in self.atoms:
			(a.x,a.y,a.z) = tuple(frame[a.id-1])
	def putframe(self,frame):
		frame[:,:] = list(map(lambda a: (a.x,a.y,a.z), self.atoms))
	def coordinates(self):
		return array([(a.x, a.y, a.z) for a in self.atoms])
	def asel(self,func):
		return ASel(self,func)
	def __repr__(self):
		return '< '+self.__class__.__name__+' with '\
			+ str(len(self.atoms))+' atoms >'

class Residue(AtomGroup):
	"""A group of atoms with extra information.

Data: type, name, id, segment, prev, next

Methods:
   r = Residue()
   r.buildrefs() - assigns residue for atoms (done by Molecule)
   r.delrefs() - removes references to allow deletion (done by Molecule)
   r[name] - returns atoms by name (like a dictionary)
   r.rotate(angle,[units]) - rotate side chain
   r.phipsi([units]) - returns (phi,psi)

See also: Atom, Molecule, 'angles'
"""
	def __init__(self):
		AtomGroup.__init__(self)
		self.type = '???'
		self.name = '???'
		self.id = 0
		self.segment = None
		self.prev = None
		self.next = None
	def buildrefs(self):
		for a in self.atoms: a.residue = self
	def delrefs(self):
		for a in self.atoms: a.residue = None
	def __getitem__(self,name):
		for a in self.atoms:
			if ( a.name == name ): return a
		raise "No such atom."
	def rotate(self,angle,units=angledefault):
		t = Trans(center=self['CA'],axis=self['CB'],angle=angle,units=units)
		for a in self.atoms:
			if a.name not in backbone: t(a)
	def phipsi(self,units=angledefault):
		try: phi = angle(self.prev['C'],self['N'],self['CA'],self['C'],units)
		except Exception: phi = None
		try: psi = angle(self['N'],self['CA'],self['C'],self.next['N'],units)
		except Exception: psi = None
		return (phi,psi)
	def __repr__(self):
		return '< Residue '+self.name+' with '\
			+str(len(self.atoms))+' atoms >'

class ASel(AtomGroup):
	"""A group of atoms generated from a filter function.

Methods:
   s = ASel(base,func)

See also: RSel
"""
	def __init__(self,base,func):
		AtomGroup.__init__(self)
		self.atoms = list(filter(func,base.atoms))

class ResidueGroup(AtomGroup):
	"""A group of residues.

Data: residues

Methods:
   g = ResidueGroup()
   g.buildlists() - generate atoms from residues
   g.phipsi([units]) - returns list of all (phi,psi)
   g.rsel(func) - returns residue selection based on filter function

See also: RSel
"""
	def __init__(self):
		AtomGroup.__init__(self)
		self.residues = []
	def buildlists(self):
		self.atoms[:] = []
		for r in self.residues:
			for a in r.atoms: self.atoms.append(a)
	def phipsi(self,units=angledefault):
		return list(map(lambda r,u=units: r.phipsi(u), self.residues))
	def rsel(self,func):
		return RSel(self,func)
	def __repr__(self):
		return '< '+self.__class__.__name__+' with '\
			+str(len(self.residues))+' residues, and '\
			+str(len(self.atoms))+' atoms >'

class RSel(ResidueGroup):
	"""A group of residues generated from a filter function.

Methods:
   s = RSel(base,func)

See also: ASel
"""
	def __init__(self,base,func):
		ResidueGroup.__init__(self)
		self.residues = list(filter(func,base.residues))
		self.buildlists()

class Segment(ResidueGroup):
	"""A group of residues with extra information.

Data: name, molecule

Methods:
   s = Segment()
   s.buildrefs() - assigns segment for residues (done by Molecule)
   s.delrefs() - removes references to allow deletion (done by Molecule)

See also: Residue, Molecule
"""
	def __init__(self):
		ResidueGroup.__init__(self)
		self.name = '???'
		self.id = 0
		molecule = None
	def buildrefs(self):
		for r in self.residues:
			r.segment = self
			r.buildrefs()
		for i in range(1,len(self.residues)):
			self.residues[i-1].next = self.residues[i]
			self.residues[i].prev = self.residues[i-1]
	def delrefs(self):
		for r in self.residues:
			r.segment = None
			r.delrefs()
		for i in range(1,len(self.residues)):
			self.residues[i-1].next = None
			self.residues[i].prev = None
	def __repr__(self):
		return '< Segment '+self.name+' with '\
			+str(len(self.residues))+' residues, and '\
			+str(len(self.atoms))+' atoms >'

class SegmentGroup(ResidueGroup):
	"""A group of segments.

Data: segments

Methods:
   g = SegmentGroup()
   g.buildlists() - generate residues from segments
"""
	def __init__(self):
		ResidueGroup.__init__(self)
		self.segments = []
	def buildlists(self):
		self.residues[:] = []
		for s in self.segments:
			s.buildlists()
			for r in s.residues: self.residues.append(r)
		ResidueGroup.buildlists(self)
	def __repr__(self):
		return '< '+self.__class__.__name__+' with '\
			+str(len(self.segments))+' segments, '\
			+str(len(self.residues))+' residues, and '\
			+str(len(self.atoms))+' atoms >'

def _sround(x,n):
	raw = str(round(x,n))
	if raw.find('.') == -1 :
		raw = raw + '.'
	while len(raw) - raw.find('.') <= n :
		raw = raw + '0'
	return raw

class Molecule(SegmentGroup):
	"""Complete interface for pdb/psf molecule files.

Data: pdbfile, psffile, pdbremarks, psfremarks
      _bonds, _angles, _dihedrals, _impropers, _donors, _acceptors
      optionally: bonds, angles, dihedrals, impropers, donors, acceptors

Methods:
   m = Molecule([pdb],[psf]) - read molecule from file(s)
   m.buildrefs() - assigns molecule for segments (done on creation)
   m.delrefs() - removes references to allow deletion (must be done by user)
   m.buildstructure() - adds structure lists to molecule and atoms
   m.writepdb([file]) - write pdb to file (pdbfile by default)
   m.view() - launch a pdb viewer with the current coordinates

See also: Segment, pdbdisplayfunction
"""
	def __init__(self,pdb=None,psf=None):
		SegmentGroup.__init__(self)
		self.pdbfile = pdb
		self.psffile = psf
		self.pdbremarks = []
		self.psfremarks = []
		self._bonds = []
		self._angles = []
		self._dihedrals = []
		self._impropers = []
		self._donors = []
		self._acceptors = []
		pdb = self.pdbfile
		psf = self.psffile
		if not ( pdb or psf ):
			raise "No data files specified."
		if pdb:
			pdbf = open(self.pdbfile,'r')
			pdbrec = pdbf.readline()
			while len(pdbrec) and pdbrec[0:6] == 'REMARK':
				self.pdbremarks.append(pdbrec.strip())
				print(self.pdbremarks[-1])
				pdbrec = pdbf.readline()
		if psf:
			psff = open(self.psffile,'r')
			psfline = psff.readline()
			psfrec = psfline.split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1] == '!NTITLE'):
				psfline = psff.readline()
				psfrec = psfline.split()
			nrecs = int(psfrec[0])
			for i in range(0,nrecs):
				psfrec = psff.readline()
				self.psfremarks.append(psfrec.strip())
				print(self.psfremarks[-1])
			psfline = psff.readline()
			psfrec = psfline.split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1] == '!NATOM'):
				psfline = psff.readline()
				psfrec = psfline.split()
			nrecs = int(psfrec[0])
		moretogo = 0
		if pdb:
			if len(pdbrec) and pdbrec[0:6] in ('ATOM  ','HETATM'): moretogo = 1
		if psf:
			psfrec = psff.readline().split()
			if len(psfrec) < 9:
				# empty segment name
				psfrec.insert(1, "")
			if nrecs > len(self.atoms): moretogo = 1
		curseg = None
		curres = None
		numread = 0
		while moretogo:
			moretogo = 0
			if psf:
				if (not curseg) or psfrec[1] != curseg.name:
					curseg = Segment()
					self.segments.append(curseg)
					curseg.name = psfrec[1]
					curseg.id = len(self.segments)
				if (not curres) or int(psfrec[2]) != curres.id:
					curres = Residue()
					curseg.residues.append(curres)
					curres.id = int(psfrec[2])
					curres.name = psfrec[3]
					curres.type = curres.name
			else:
				if (not curseg) or pdbrec[67:].strip() != curseg.name:
					curseg = Segment()
					self.segments.append(curseg)
					curseg.name = pdbrec[67:].strip()
					curseg.id = len(self.segments)
				if (not curres) or int(pdbrec[22:26]) != curres.id:
					curres = Residue()
					curseg.residues.append(curres)
					curres.id = int(pdbrec[22:26])
					curres.name = pdbrec[17:21].strip()
					curres.type = curres.name
			curatom = Atom()
			curres.atoms.append(curatom)
			numread = numread + 1
			if pdb:
				curatom.name = pdbrec[12:16].strip()
				curatom.type = curatom.name
				curatom.id = int(pdbrec[6:11])
				curatom.x = float(pdbrec[30:38])
				curatom.y = float(pdbrec[38:46])
				curatom.z = float(pdbrec[46:54])
				curatom.q = float(pdbrec[54:60])
				curatom.b = float(pdbrec[60:66])
				pdbrec = pdbf.readline()
				if len(pdbrec) and pdbrec[0:6] in ('ATOM  ','HETATM'): moretogo = 1
			if psf:
				curatom.name = psfrec[4]
				curatom.type = psfrec[5]
				curatom.id = int(psfrec[0])
				curatom.mass = float(psfrec[7])
				curatom.charge = float(psfrec[6])
				psfrec = psff.readline().split()
				if len(psfrec) < 9:
					# empty segment name
					psfrec.insert(1, "")
				if nrecs > numread: moretogo = 1
		if pdb: pdbf.close()
		if psf:
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:6] == '!NBOND'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._bonds.append((int(psfrec[0]),int(psfrec[1])))
					nrecs = nrecs - 1
					psfrec = psfrec[2:]
			# if there are problems/errors in the remainder of the PSF, we don't care...
			psff.close()
			self.buildlists()
			self.buildrefs()
			return

			# original code follows...
			psfrec = psff.readline().split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:7] == '!NTHETA'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._angles.append((int(psfrec[0]),
						int(psfrec[1]),int(psfrec[2])))
					nrecs = nrecs - 1
					psfrec = psfrec[3:]
			psfrec = psff.readline().split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:5] == '!NPHI'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._dihedrals.append((int(psfrec[0]),int(psfrec[1]),
						int(psfrec[2]),int(psfrec[3])))
					nrecs = nrecs - 1
					psfrec = psfrec[4:]
			psfrec = psff.readline().split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:7] == '!NIMPHI'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._impropers.append((int(psfrec[0]),int(psfrec[1]),
						int(psfrec[2]),int(psfrec[3])))
					nrecs = nrecs - 1
					psfrec = psfrec[4:]
			psfrec = psff.readline().split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:5] == '!NDON'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._donors.append((int(psfrec[0]),int(psfrec[1])))
					nrecs = nrecs - 1
					psfrec = psfrec[2:]
			psfrec = psff.readline().split()
			while len(psfline) and not (len(psfrec) > 1 and psfrec[1][0:5] == '!NACC'):
				psfrec = psff.readline().split()
			nrecs = int(psfrec[0])
			while ( nrecs ):
				psfrec = psff.readline().split()
				while ( len(psfrec) ):
					self._acceptors.append((int(psfrec[0]),int(psfrec[1])))
					nrecs = nrecs - 1
					psfrec = psfrec[2:]
			psff.close()
		self.buildlists()
		self.buildrefs()
	def buildrefs(self):
		for s in self.segments:
			s.molecule = self
			s.buildrefs()
	def delrefs(self):
		for s in self.segments:
			s.molecule = None
			s.delrefs()
	def buildstructure(self):
		self.bonds = []
		self.angles = []
		self.dihedrals = []
		self.impropers = []
		self.donors = []
		self.acceptors = []
		for a in self.atoms:
			a.bonds = []
			a.angles = []
			a.dihedrals = []
			a.impropers = []
			a.donors = []
			a.acceptors = []
		def mapfunc(id,list=self.atoms): 
			if id == 0:
				return None
			a = list[id-1]
			if ( a.id != id ):
				raise "Atom list indexes corrupted. ('%s' != '%s'" % (a.id, id)
			return a
		def mapatom(t,func=mapfunc):
			return tuple(map(func,t))
		for b in self._bonds:
			s = mapatom(b)
			self.bonds.append(s)
			for a in s: a.bonds.append(s)
		for b in self._angles:
			s = mapatom(b)
			self.angles.append(s)
			for a in s: a.angles.append(s)
		for b in self._dihedrals:
			s = mapatom(b)
			self.dihedrals.append(s)
			for a in s: a.dihedrals.append(s)
		for b in self._impropers:
			s = mapatom(b)
			self.impropers.append(s)
			for a in s: a.impropers.append(s)
		for b in self._donors:
			s = mapatom(b)
			self.donors.append(s)
			for a in s: a.donors.append(s)
		for b in self._acceptors:
			s = mapatom(b)
			self.acceptors.append(s)
			for a in s:
				if a is not None:
					a.acceptors.append(s)
	def writepdb(self,pdbfile=None):
		if not pdbfile:
			pdbfile = self.pdbfile
		if not pdbfile:
			raise "No pdb file specified."
		f = open(pdbfile,'w')
		for r in self.pdbremarks:
			if r[0:6] == 'REMARK' :
				f.write(r+'\n')
			else:
				f.write('REMARK '+r+'\n')
		for a in self.atoms:
			f.write('ATOM  ')
			f.write(str(a.id).rjust(5)+' ')
			if len(a.name) > 3:
				f.write(a.name.ljust(4)+' ')
			else:
				f.write(' '+a.name.ljust(3)+' ')
			f.write(a.residue.name.ljust(4))
			f.write(' '+str(a.residue.id).rjust(4)+'    ')
			f.write(_sround(a.x,3).rjust(8))
			f.write(_sround(a.y,3).rjust(8))
			f.write(_sround(a.z,3).rjust(8))
			f.write(_sround(a.q,2).rjust(6))
			f.write(_sround(a.b,2).rjust(6))
			f.write(a.residue.segment.name.rjust(10))
			f.write('\n')
		f.write('END\n')
		f.close()
	def view(self):
		d = pdbview()
		self.writepdb(d.load())
		d.show()
		d.free()
	def __repr__(self):
		return '< Molecule with '\
			+str(len(self.segments))+' segments, '\
			+str(len(self.residues))+' residues, and '\
			+str(len(self.atoms))+' atoms >'


