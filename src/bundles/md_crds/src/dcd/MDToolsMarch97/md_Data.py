"""Data Hierarchy for MDTools

RCS: $Id: md_Data.py 26655 2009-01-07 22:02:30Z gregc $

Class Hierarchy:
   Data -> NAMDOutput
"""

_RCS = "$Id: md_Data.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 1.1  2004/05/17 18:43:19  pett
# as distributed
#
# Revision 0.67  1997/03/14 21:52:25  jim
# Changes to NAMDOutput to support changes in NAMD 2.
#
# Revision 0.66  1997/03/08 19:22:17  jim
# Fixed old-style calls to Data.plot() in NAMDOutput.
#
# Revision 0.65  1996/05/24 01:27:26  jim
# Split into sub-modules, improved version reporting.
#

#print "- Data "+"$Revision: 1.2 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2005-08-20 00:26:36 $"[7:-11]+")"

import math
import struct
import copy
import tempfile
import os
import sys
import time

from .md_local import xyplot

#
# Data class hierarchy
#                                       Data
#                                         |
#                                     NAMDOutput
#

class Data:
	"""General structure for sequence of data points.

Data: fields, names, data

Methods:
   d = Data(fields,[data])
      fields: list or tuple of field names as in ('t','x','y','z')
   d.append(rec) - append a record such as (1,1.2,3.2,5.1)
   d[8:13] - return a list of records
   len(d) - return number of records
   addfield(name,args,func) - add a field based on other fields
      name: name of new field, as in 'x+y'
      args: tuple of arguments for func, as in ('x','y'), just one, as in 'x'
      func: function to create new field, as in lambda x,y:x+y
   addindex([offset],[name]) - add an index field
   filter(args,func) - return new Data of records that pass filter func
   average(args,[func]) - mean of function (default is x)
   deviation(args,[func]) - STD of function (default is x)
   Warning: deviation() divides by N - 1 to estimate the STD from a sample
   plot([args]) - launch a plotting program to plot args
   list([args],[file],[titles]) - print to screen (file) w/ or w/o titles

See also: xyplotfunction
"""
	def __init__(self,fields,data=[]):
		self.fields = {}
		self.names = tuple(fields)
		self.data = []
		for f in fields:
			if isinstance(f, int):
				raise "integer field names not allowed"
			if f in self.fields.keys():
				raise "duplicate field name "+str(f)
			self.fields[f] = len(self.fields)
		for d in data:
			self.data.append(tuple(d))
	def __getitem__(self,key):
		if isinstance(key, int):
			return self.data[key]
		if isinstance(key, tuple):
			fs = map(lambda k,d=self.fields: d[k],key)
			tfunc = lambda rec,f=fs: \
				tuple(map(lambda e,r=rec:r[e],f))
			return map(tfunc,self.data)
		else:
			return map(lambda r,f=self.fields[key]: r[f],self.data)
	def __getslice__(self,i,j):
		return self.data[i:j]
	def __len__(self):
		return len(self.data)
	def __repr__(self):
		return '< Data with '+str(len(self.data))+' frames of '+str(self.names)+' data >'
	def append(self,rec):
		if ( len(rec) != len(self.fields) ):
			raise 'appending wrong length record'
		self.data.append(tuple(rec))
	def addfield(self,name,args,func):
		if not isinstance(args, tuple): args = (args,)
		fs = map(lambda k,d=self.fields: d[k],args)
		tfunc = lambda rec,f=fs: \
			tuple(map(lambda e,r=rec:r[e],f))
		data = []
		for d in self.data:
			dl = map(None,d)
			dl.append(apply(func,tfunc(d)))
			data.append(tuple(dl))
		self.fields[name] = len(self.fields)
		nl = map(None,self.names)
		nl.append(name)
		self.names = tuple(nl)
		self.data = data
	def addindex(self,offset=0,name='index'):
		if name in self.names:
			raise 'field name '+str(name)+' already in use'
		for i in range(0,len(self.data)):
			dl = map(None,self.data[i])
			dl[:0] = [i+offset]
			self.data[i] = tuple(dl)
		nl = map(None,self.names)
		nl[:0] = [name]
		self.names = tuple(nl)
		for n in self.fields.keys():
			self.fields[n] = self.fields[n] + 1
		self.fields[name] = 0
	def filter(self,args,func):
		if not isinstance(args, tuple): args = (args,)
		fs = map(lambda k,d=self.fields: d[k],args)
		tfunc = lambda rec,f=fs: \
			tuple(map(lambda e,r=rec:r[e],f))
		ffunc = lambda r,t=tfunc,f=func: apply(f,t(r))
		return Data(self.names,filter(ffunc,self.data))
	def average(self,args,func=lambda x:x,zero=0.):
		if not isinstance(args, tuple): args = (args,)
		fs = map(lambda k,d=self.fields: d[k],args)
		tfunc = lambda rec,f=fs: \
			tuple(map(lambda e,r=rec:r[e],f))
		ffunc = lambda s,r,t=tfunc,f=func: s+apply(f,t(r))
		return reduce(ffunc,self.data,zero)/len(self.data)
	def deviation(self,args,func=lambda x:x,zero=0.):
		if not isinstance(args, tuple): args = (args,)
		fs = map(lambda k,d=self.fields: d[k],args)
		tfunc = lambda rec,f=fs: \
			tuple(map(lambda e,r=rec:r[e],f))
		ffunc = lambda s,r,t=tfunc,f=func: s+apply(f,t(r))
		avg = reduce(ffunc,self.data,zero)/len(self.data)
		ffunc = lambda s,r,t=tfunc,f=func,a=avg: s+pow(apply(f,t(r))-a,2)
		return math.sqrt(reduce(ffunc,self.data,zero)/(len(self.data)-1))
	def plot(self,args=None):
		if args is None: args = self.names
		if not isinstance(args, tuple): args = (args,)
		p = xyplot()
		file = p.load()
		f = open(file,'w')
		for e in args:
			f.write(e+' ')
		f.write('\n')
		for r in self.data:
			if None not in r:
				for e in args:
					f.write(str(r[self.fields[e]])+' ')
				f.write('\n')
		f.close()
		p.show()
		p.free()
	def list(self,args=None,file=None,titles=1):
		if args is None: args = self.names
		if not isinstance(args, tuple): args = (args,)
		if file is None: f = sys.stdout
		else: f = open(file,'w')
		if titles:
			for e in args:
				f.write(e.center(18))
		f.write('\n')
		for r in self.data:
			for e in args:
				f.write(str(r[self.fields[e]]).center(18))
			f.write('\n')
		if file is not None: f.close()

def _NAMD_infochop(s):
	return s[s.find('>')+1:]

class NAMDOutput(Data):
	"""Reads output files of the molecular dynamics program NAMD.

Data: timestep, namdfields

Methods:
   d = NAMDOutput(namdfile,[fields=('TS','TEMP')])
   d.append(namdfile) - append another output file
   d.addtime() - add a 'TIME' field based on timestep
   d.plot([args]) - same as Data but eliminates 'TS' if 'TIME' present

See also: http://www.ks.uiuc.edu/Research/namd/
"""
	def __init__(self,namdfile,fields=('TS','TEMP')):
		Data.__init__(self,fields)
		self.namdfile = namdfile
		self.namdfields = {}
		self.timestep = 0
		f = open(self.namdfile,'r')
		raw = f.readline()
		rec = _NAMD_infochop(raw).split()
		while len(raw) and raw[-1] == '\n' and not self.namdfields :
			if len(rec) and rec[0] == 'ETITLE:' :
				self.namdfields = {}
				for fn in self.names:
					self.namdfields[fn] = rec.index(fn)
			elif len(rec) == 2 and rec[0] == 'TIMESTEP' :
				self.timestep = float(rec[1]) / 1000.0
			elif len(rec) == 3 and rec[0] == 'Info:' and rec[1] == 'TIMESTEP' :
				self.timestep = float(rec[2]) / 1000.0
			raw = f.readline()
			rec = _NAMD_infochop(raw).split()
		fieldnums = []
		for fn in self.names:
			fieldnums.append(self.namdfields[fn])
		while len(raw) and raw[-1] == '\n':
			if len(rec) and rec[0] == 'ENERGY:' :
				dr = []
				for i in fieldnums:
					dr.append(float(rec[i]))
				self.data.append(tuple(dr))
			raw = f.readline()
			rec = _NAMD_infochop(raw).split()
		f.close()
	def append(self,namdfile):
		f = open(self.namdfile,'r')
		fieldnums = []
		for fn in self.names:
			fieldnums.append(self.namdfields[fn])
		raw = f.readline()
		rec = _NAMD_infochop(raw).split()
		while len(raw) :
			if len(rec) and rec[0] == 'ENERGY:' :
				dr = []
				for i in fieldnums:
					dr.append(float(rec[i]))
				self.data.append(tuple(dr))
			raw = f.readline()
			rec = _NAMD_infochop(raw).split()
		f.close()
	def addtime(self):
		self.addfield('TIME','TS',lambda ts,d=self.timestep:ts*d)
	def plot(self,args=None):
		if args is None and 'TIME' in self.names:
			l = map(None,(self.names))
			l.remove('TIME')
			try: l.remove('TS')
			except Exception: pass
			l[0:0] = ['TIME']
			Data.plot(self,tuple(l))
		else:
			Data.plot(self,args)

