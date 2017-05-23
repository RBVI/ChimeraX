"""Localizations for MDTools for Python by James Phillips.
Implements pdbview class using Rasmol.
RCS: $Id: md_local_rasmol.py 26655 2009-01-07 22:02:30Z gregc $
"""

_RCS = "$Id: md_local_rasmol.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 0.8  1996/05/24 01:34:19  jim
# Improved version reporting.
#
# Revision 0.7  1996/05/23 22:52:02  jim
# Split up md_local module to import from sub-modules.
#

#print "- local_rasmol "+"$Revision: 1.1 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2004-05-17 18:43:19 $"[7:-11]+")"

import os
import tempfile

class pdbview:
	def __init__(self):
		self.stdinname = tempfile.mktemp() + '.md.py.stdin'
		os.system('mkfifo ' + self.stdinname)
		os.system('rasmol < ' + self.stdinname + ' &')
		self.stdin = open(self.stdinname,'w')
	def send(self,command):
		self.stdin.write(command+'\n')
		self.stdin.flush()
	def load(self):
		self.pdbfile = tempfile.mktemp() + '.md.py.pdb'
		os.system('mkfifo ' + self.pdbfile)
		self.send('zap')
		self.send('load pdb '+self.pdbfile)
		return self.pdbfile
	def show(self):
		os.unlink(self.pdbfile)
		del(self.pdbfile)
	def free(self):
		self.stdin.close()
		os.unlink(self.stdinname)
	def kill(self):
		self.send('exit')
		self.free()

