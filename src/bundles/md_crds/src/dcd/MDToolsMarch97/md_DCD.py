"""DCD Hierarchy for MDTools.

RCS: $Id: md_DCD.py 31986 2010-11-25 00:08:52Z pett $

Class Hierarchy:
   DCD -> DCDWrite
   Data -> NAMDOutput
"""

_RCS = "$Id: md_DCD.py 31986 2010-11-25 00:08:52Z pett $"

# $Log: not supported by cvs2svn $
# Revision 1.9  2007/02/07 21:52:55  pett
# missing part of numpy conversion
#
# Revision 1.8  2007/02/07 20:59:45  pett
# convert to Numpy
#
# Revision 1.7  2007/01/09 01:10:44  pett
# without unit cell info, the format is the same as X-PLOR
#
# Revision 1.6  2006/09/20 20:25:50  pett
# fixed issue with byte-padding of structures
#
# Revision 1.5  2006/09/20 00:27:28  pett
# NAMD/CHARMm embed periodic cell info with each frame
#
# Revision 1.4  2006/06/28 21:38:51  pett
# some fixes for fixed-position trajectories from Ben Webb
#
# Revision 1.3  2005/08/20 00:26:36  gregc
# Update for wrappy2.
#
# Revision 1.2  2004/05/17 18:47:53  pett
# handle endianness issues
#
# Revision 0.71  1996/11/16 21:30:21  jim
# changed Numeric.Core to Numeric
#
# Revision 0.70  1996/10/12 18:52:09  jim
# update for NumPy1.0a4
#
# Revision 0.69  1996/10/10 21:41:36  jim
# Changes for new Numeric Python version.
#
# Revision 0.68  1996/08/28 20:51:47  jim
# Changes for coordinates() method of AtomGroup (rev. 0.67).
#
# Revision 0.67  1996/08/28 19:35:01  jim
# Fixed docs to reflect use of arrays.
#
# Revision 0.66  1996/08/28 19:28:22  jim
# Switched frames from lists of tuples to arrays.
#
# Revision 0.65  1996/05/24 01:24:23  jim
# Split into sub-modules, improved version reporting.
#

#print "- DCD "+"$Revision: 1.10 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2007-02-07 22:18:50 $"[7:-11]+")"

import math
import struct
import copy
import tempfile
import os
import sys
import time
from numpy import array, zeros, float32, float64, fromstring, take

from .md_AtomGroup import *

class DCDFormatError(ValueError):
        pass

#
# DCD class hierarchy:
#                                        DCD
#                                         |
#                                      DCDWrite
#

class DCD:
        """Reads from a DCD file.

Data: dcdfile, file, posNSET, NSET, ISTART, NSAVC, NAMNF, DELTA, remarks, NTITLE, N, FREEINDEXES, fixed, pos1, pos2, rlen, numframes, numatoms

Methods:
   d = DCD(dcdfile) - open dcdfile and read header
   len(d) - return number of frames
   d[10] - return a frame as an array
   s = d.asel() - return a selection of dummy atoms for all atoms in d
   sf = d.aselfree(mol) - return a selection from mol of the free atoms in d
   sf2 = d.aselfree() - return a selection of dummy atoms for free atoms in d
   sf.getmolframe(d[10]) - load only free atoms from d into mol

See also: AtomGroup, ASel
"""
        def __init__(self,dcdfile):
                self.dcdfile = dcdfile
                self.file = open(dcdfile,'rb')
                if hasattr(self.file, 'fileno'):
                        from mmap import mmap, ACCESS_READ
                        self.file = mmap(self.file.fileno(), 0, access=ACCESS_READ)
                # Abbreviations
                cs = struct.calcsize
                f = self.file
                # Read header
                dat = struct.unpack('i4c',f.read(cs('i4c')))
                if dat[0] in [84, 164]:
                        def endianFunc(*args):
                                return struct.unpack(*args)
                        self.byteSwap = False
                        recLen = dat[0]
                elif dat[0] in [1409286144, 2751463424]:
                        # force byte reversal
                        if dat[0] == 1409286144:
                                recLen = 84
                        else:
                                recLen = 164
                        v = struct.unpack("<i", struct.pack("i", recLen))
                        if v[0] == recLen:
                                # little-endian
                                def endianFunc(fmt, *args):
                                        return struct.unpack(">"+fmt, *args)
                        else:
                                def endianFunc(fmt, *args):
                                        return struct.unpack("<"+fmt, *args)
                        self.byteSwap = True
                else:
                        raise DCDFormatError("1")
                # initial record is 4-character string + 20 ints,
                # so recLen of 84 implies 4-byte ints and recLen
                # of 164 8-byte ints
                if recLen == 84:
                        intFmt = 'i'
                        floatFmt = 'f'
                        bytesPerInt = 4
                        numpyFloat = float32
                else:
                        intFmt = 'q'
                        floatFmt = 'd'
                        bytesPerInt = 8
                        numpyFloat = float64
                self.intFmt = intFmt
                if dat[1:5] != (b'C',b'O',b'R',b'D') :
                        raise DCDFormatError("1")
                self.endianFunc = endianFunc
                up = endianFunc
                self.posNSET = f.tell()
                self.NSET = up(intFmt,f.read(cs(intFmt)))[0]
                self.ISTART = up(intFmt,f.read(cs(intFmt)))[0]
                self.NSAVC = up(intFmt,f.read(cs(intFmt)))[0]
                f.read(cs('4'+intFmt))
                f.read(cs(intFmt)) # Why?
                self.NAMNF = up(intFmt,f.read(cs(intFmt)))[0]
                self.DELTA = up(floatFmt,f.read(cs(floatFmt)))[0]
                NAMDvsXPLOR = up(intFmt,f.read(cs(intFmt)))[0] # NAMD/CHARMm
                if NAMDvsXPLOR in [0,1]:
                        self.NAMD = True
                        self.hasCellInfo = NAMDvsXPLOR
                else:
                        self.NAMD = self.hasCellInfo = False
                f.read(cs(intFmt)) # CHARMm "4 dims" flag
                f.read(cs('7'+intFmt))
                f.read(cs(intFmt)) # CHARMm version #
                dat = up('i',f.read(cs('i')))[0]
                if dat != recLen :
                        raise DCDFormatError("2")
                size = up('i',f.read(cs('i')))[0]
                if (size-4)%80 != 0 :
                        raise DCDFormatError("3")
                self.remarks = []
                self.NTITLE = up('i',f.read(cs('i')))[0]
                for i in range(0,self.NTITLE):
                        dat = []
                        for c in up('80c',f.read(cs('80c'))):
                            if c == b'\x00':
                                break
                            dat.append(c.decode('utf-8'))
                        self.remarks.append(''.join(dat).strip())
                        #print(self.remarks[-1])
                if up('i',f.read(cs('i')))[0] != size :
                        raise DCDFormatError("4")
                if up('i',f.read(cs('i')))[0] != bytesPerInt :
                        raise DCDFormatError("5")
                self.N = up(intFmt,f.read(cs(intFmt)))[0]
                if up('i',f.read(cs('i')))[0] != bytesPerInt :
                        raise DCDFormatError("6")
                if self.NAMNF:
                        size = up('i',f.read(cs('i')))[0]
                        if size != (self.N-self.NAMNF)*cs(intFmt) :
                                raise DCDFormatError("7")
                        fi = up(repr(self.N-self.NAMNF)+intFmt,
                                f.read(cs(repr(self.N-self.NAMNF)+intFmt)))
                        self.FREEINDEXES = array(fi) - 1
                        if up('i',f.read(cs('i')))[0] != size :
                                raise DCDFormatError("8")
                else:
                        self.FREEINDEXES = ()
                if self.NAMD and self.hasCellInfo:
                        fmt = "2i6d2i%df4i"
                else:
                        fmt = "%df6i"
                        
                self.fixed = self.NAMNF
                self.pos1 = f.tell()
                self.pos2 = self.pos1 + cs(fmt % (3*self.N))
                self.rlen = cs(fmt % (3*(self.N-self.NAMNF)))
                if self.fixed :
                        self.fixed_buff = zeros((self.N,3),'f')
                        size = up(intFmt,f.read(cs(intFmt)))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("9")
                        tmp = fromstring(f.read(cs(repr(self.N)+floatFmt)),numpyFloat)
                        if self.byteSwap:
                                tmp.byteswap(True)
                        self.fixed_buff[...,0] = tmp
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("10")
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("11")
                        tmp = fromstring(f.read(cs(repr(self.N)+floatFmt)),numpyFloat)
                        if self.byteSwap:
                                tmp.byteswap(True)
                        self.fixed_buff[...,1] = tmp
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("12")
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("13")
                        tmp = fromstring(f.read(cs(repr(self.N)+floatFmt)),numpyFloat)
                        if self.byteSwap:
                                tmp.byteswap(True)
                        self.fixed_buff[...,2] = tmp
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+floatFmt) :
                                raise DCDFormatError("14")
                f.seek(0,2)
                self.numframes = int((f.tell()-self.pos2)/self.rlen + 1.5)
                self.numatoms = self.N
        def __getitem__(self,fn):
                # Abbreviations
                up = self.endianFunc
                cs = struct.calcsize
                f = self.file
                # Find the right point in the file
                if fn < -1*self.numframes or fn >= self.numframes :
                        raise IndexError
                elif fn == 0 and self.fixed :
                        return copy.copy(self.fixed_buff)
                elif fn < 0 :
                        return self.__getitem__(self.numframes + fn)
                else :
                        f.seek(self.pos2 + (fn-1)*self.rlen)
                # Read data
                if self.hasCellInfo:
                        if up('i',f.read(cs('i')))[0] != 48 :
                                raise DCDFormatError("15")
                        f.read(cs('6d')) # periodic cell
                        if up('i',f.read(cs('i')))[0] != 48 :
                                raise DCDFormatError("16")
                size = up('i',f.read(cs('i')))[0]
                if self.fixed == 0 :
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("9")
                        x = fromstring(f.read(cs(repr(self.N)+'f')),float32)
                        if self.byteSwap:
                                x.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("10")
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("11")
                        y = fromstring(f.read(cs(repr(self.N)+'f')),float32)
                        if self.byteSwap:
                                y.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("12")
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("13")
                        z = fromstring(f.read(cs(repr(self.N)+'f')),float32)
                        if self.byteSwap:
                                z.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != cs(repr(self.N)+'f') :
                                raise DCDFormatError("14")
                        frame = array([x,y,z]).transpose()
                else:
                        free = len(self.FREEINDEXES)
                        fm = repr(free)+'f'
                        sz = cs(fm)
                        if size != sz :
                                raise DCDFormatError("9")
                        xfree = fromstring(f.read(sz),float32)
                        if self.byteSwap:
                                xfree.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != sz :
                                raise DCDFormatError("10")
                        size = up('i',f.read(cs('i')))[0]
                        if size != sz :
                                raise DCDFormatError("11")
                        yfree = fromstring(f.read(sz),float32)
                        if self.byteSwap:
                                yfree.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != sz :
                                raise DCDFormatError("12")
                        size = up('i',f.read(cs('i')))[0]
                        if size != sz :
                                raise DCDFormatError("13")
                        zfree = fromstring(f.read(sz),float32)
                        if self.byteSwap:
                                zfree.byteswap(True)
                        size = up('i',f.read(cs('i')))[0]
                        if size != sz :
                                raise DCDFormatError("14")
                        frame = copy.copy(self.fixed_buff)
                        ii = 0
                        for i in self.FREEINDEXES:
                                frame[i] = (xfree[ii],yfree[ii],zfree[ii])
                                ii = ii + 1
                return frame
        def asel(self):
                fakemol = AtomGroup()
                for i in range(0,self.numatoms):
                        a = Atom()
                        a.id = i + 1
                        fakemol.atoms.append(a)
                return fakemol
        def aselfree(self,mol=None):
                if ( mol is None ):
                        if ( self.fixed ):
                                fakemol = AtomGroup()
                                for i in self.FREEINDEXES:
                                        a = Atom()
                                        a.id = i + 1
                                        fakemol.atoms.append(a)
                                return fakemol
                        else:
                                return self.asel()
                else:
                        if ( self.fixed ):
                                return ASel(mol,lambda a,l=self.FREEINDEXES+1: a.id in l)
                        else:
                                return ASel(mol,lambda a,l=range(1,self.numatoms+1): a.id in l)
        def __len__(self):
                return self.numframes
        def __del__(self):
                self.file.close()
        def __repr__(self):
                return "< DCD " + self.dcdfile + " with " + repr(self.numframes) + " frames of " + repr(self.numatoms) + " atoms (" + repr(self.fixed) + " fixed) >"

class DCDWrite(DCD):
        """Writes a DCD file.  Can only append.

Data: allatoms, freeatoms.

Methods:
   d = DCD(dcdfile,atoms,[free],[ISTART],[NSAVC],[DELTA]) - open dcdfile
      dcdfile: must not exist
      atoms=AtomGroup: all atoms to save to the file
      free=AtomGroup: if there are fixed atoms, these are the free ones
      ISTART: first timestep
      NSAVC: saving frequency
      DELTA: timestep
   d.append() - append the current coordinates of atoms (free) to the file

Note: DO NOT modify atoms or free while DCDWrite is being used.

See also: AtomGroup
"""
        def __init__(self,dcdfile,atoms,free=None,**header):
                self.allatoms = atoms
                if free is None: self.freeatoms = self.allatoms
                else: self.freeatoms = free
                self.dcdfile = dcdfile
                self.file = open(dcdfile,'w+b')
                # Abbreviations
                p = struct.pack
                cs = struct.calcsize
                f = self.file
                # Write header
                f.write(p('i4c',84,b'C',b'O',b'R',b'D'))
                self.posNSET = f.tell()
                self.NSET = 0
                self.numframes = 0
                f.write(p('i',self.NSET))
                try:
                        self.ISTART = header['ISTART']
                except Exception:
                        self.ISTART = 0
                f.write(p('i',self.ISTART))
                try:
                        self.NSAVC = header['NSAVC']
                except Exception:
                        self.NSAVC = 0
                f.write(p('i',self.NSAVC))
                f.write(p('4i',0,0,0,0))
                f.write(p('i',0)) # Why?
                self.NAMNF = len(self.allatoms.atoms) - len(self.freeatoms.atoms)
                f.write(p('i',self.NAMNF))
                self.fixed = self.NAMNF
                try:
                        self.DELTA = header['DELTA']
                except Exception:
                        self.DELTA = 0.
                f.write(p('d',self.DELTA))
                f.write(p('i',0)) # Why?
                f.write(p('8i',0,0,0,0,0,0,0,0))
                f.write(p('i',84))
                rawremarks = [ \
                        'REMARKS FILENAME='+self.dcdfile+' CREATED BY MDTools for Python',
                        'REMARKS DATE: '+time.ctime(time.time())+' CREATED BY USER: '+os.environ['USER']]
                self.remarks = []
                for r in rawremarks: self.remarks.append(r.ljust(80)[0:80])
                size = cs('i')+80*len(self.remarks)
                f.write(p('i',size))
                self.NTITLE = len(self.remarks)
                f.write(p('i',self.NTITLE))
                for r in self.remarks:
                        f.write(p('80c', *tuple(c.encode('utf-8') for c in r)))
#                        print(r.strip())
                f.write(p('i',size))
                f.write(p('i',cs('i')))
                self.N = len(self.allatoms.atoms)
                self.numatoms = self.N
                f.write(p('i',self.N))
                f.write(p('i',cs('i')))
                if self.NAMNF:
                        size = (self.N-self.NAMNF)*cs('i')
                        f.write(p('i',size))
                        #fi = []
                        #for i in range(0,len(self.allatoms.atoms)):
                        #        if self.allatoms.atoms[i] in self.freeatoms.atoms: fi.append(i+1)
                        fi = map(lambda a,l=self.allatoms.atoms: l.index(a)+1,self.freeatoms.atoms)
                        self.FREEINDEXES = array(fi) - 1
                        if len(self.FREEINDEXES) != self.N-self.NAMNF:
                                raise "some free atoms were not in atoms list"
                        fi[:0] = [repr(self.N-self.NAMNF)+'i']
                        f.write(apply(p,tuple(fi)))
                        f.write(p('i',size))
                else:
                        self.FREEINDEXES = ()
                self.pos1 = f.tell()
                self.pos2 = self.pos1 + cs(repr(3*self.N)+'f6i')
                self.rlen = cs(repr(3*(self.N-self.NAMNF))+'f6i')
                f.flush()
        def append(self,frame=None):
                p = struct.pack
                cs = struct.calcsize
                f = self.file
                fn = self.NSET
                self.NSET = self.NSET + 1
                self.numframes = self.NSET
                f.seek(self.posNSET)
                f.write(p('i',self.NSET))
                f.seek(0,2)
                if frame is not None:
                        if len(frame) != self.N : raise "frame size mismatch"
                        if fn and self.fixed:
                                fr = take(frame,self.FREEINDEXES, 0)
                        else:
                                fr = frame
                else:
                        if fn:
                                fr = self.freeatoms.coordinates()
                        else:
                                fr = self.allatoms.coordinates()
                                if self.fixed :
                                        self.fixed_buff = fr
                x = array(fr[...,0],float32)
                y = array(fr[...,1],float32)
                z = array(fr[...,2],float32)
                fm = repr(len(x))+'f'
                size = cs(fm)
                f.write(p('i',size))
                f.write(x.tostring())
                f.write(p('i',size))
                f.write(p('i',size))
                f.write(y.tostring())
                f.write(p('i',size))
                f.write(p('i',size))
                f.write(z.tostring())
                f.write(p('i',size))
                f.flush()
        def __repr__(self):
                return "< DCDWrite " + self.dcdfile + " with " + repr(self.numframes) + " frames of " + repr(self.numatoms) + " atoms (" + repr(self.fixed) + " fixed) >"

