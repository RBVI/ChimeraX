"""Localizations for MDTools for Python by James Phillips.
Implements xyplot and pdbview classes using open.
RCS: $Id: md_local_open.py 26655 2009-01-07 22:02:30Z gregc $
"""

_RCS = "$Id: md_local_open.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 0.8  1996/05/24 01:34:00  jim
# Improved version reporting.
#
# Revision 0.7  1996/05/23 22:50:29  jim
# Split up md_local module to import from sub-modules.
#

#print "- local_open "+"$Revision: 1.1 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2004-05-17 18:43:19 $"[7:-11]+")"

import os
import tempfile

class xyplot:
	def __init__(self):
		pass
	def send(self,command):
		print("Sorry, can't send commands.")
	def load(self):
		self.datfile = tempfile.mktemp() + '.md.py.dat'
		return self.datfile
	def show(self):
		os.system('( open ' + self.datfile +
			'; sleep 60; /bin/rm -f ' + self.datfile + ' )&')
		del(self.datfile)
	def free(self):
		pass
	def kill(self):
		print("Sorry, can't kill.")

class pdbview:
	def __init__(self):
		pass
	def send(self,command):
		print("Sorry, can't send commands.")
	def load(self):
		self.datfile = tempfile.mktemp() + '.md.py.pdb'
		return self.datfile
	def show(self):
		os.system('( open ' + self.datfile +
			'; sleep 60; /bin/rm -f ' + self.datfile + ' )&')
		del(self.datfile)
	def free(self):
		pass
	def kill(self):
		print("Sorry, can't kill.")

