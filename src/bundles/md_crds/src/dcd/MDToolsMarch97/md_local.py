"""Localizations for MDTools for Python by James Phillips.
RCS: $Id: md_local.py 26655 2009-01-07 22:02:30Z gregc $
"""

_RCS = "$Id: md_local.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 0.8  1996/05/24 01:33:33  jim
# Improved version reporting.
#
# Revision 0.7  1996/05/23 22:42:17  jim
# xyplotfunction() and pdbdisplayfunction() are now classes
# xyplot and pdbview so they keep data internally.
# Changed if statements to import from different sub-modules.
# xmgr is now used as the plotting program under X-windows.
# There is a matching revision to md for this update.
#
# Revision 0.6  1996/05/08 20:04:04  jim
# Fixed NEXTSTEP versions of xyplotfunction and pdbdisplayfunction.
#
# Revision 0.5  1996/05/07 22:22:46  jim
# Started using RCS.
#

#print "- local "+"$Revision: 1.1 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2004-05-17 18:43:19 $"[7:-11]+")"

import os

try:
	ostype = os.environ['OSTYPE']
except Exception:
	ostype = '???'

if ostype in ('IRIX','HP-UX','SunOS'):
	_RCS = _RCS + '\n'
	from . import md_local_rasmol
	_RCS = _RCS + md_local_rasmol._RCS
	from .md_local_rasmol import pdbview
	_RCS = _RCS + '\n'
	from . import md_local_xmgr
	_RCS = _RCS + md_local_xmgr._RCS
	from .md_local_xmgr import xyplot
elif ostype in ('NeXT',):
	_RCS = _RCS + '\n'
	from . import md_local_open
	_RCS = _RCS + md_local_open._RCS
	from .md_local_open import xyplot, pdbview
else:
	#print("Unknown OSTYPE " + ostype + " set.  Viewing and plotting are disabled.")
	class pdbview:
		def __init__(self):
			raise "Viewing is disabled."
	class xyplot:
		def __init__(self):
			raise "Plotting is disabled."


