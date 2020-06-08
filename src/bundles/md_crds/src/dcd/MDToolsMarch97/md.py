"""MDTools provides utilities for manipulating molecular data.

Written by James Phillips, University of Illinois.

WWW: http://www.ks.uiuc.edu/~jim/mdtools/

RCS: $Id: md.py 26655 2009-01-07 22:02:30Z gregc $

Class Hierarchy:
   HomoCoord -> Coord -> Atom
             -> Vector
   AtomGroup -> ASel
             -> Residue
             -> ResidueGroup -> RSel
                             -> Segment
                             -> SegmentGroup -> Molecule
   Trans
   DCD -> DCDWrite
   Data -> NAMDOutput

Utilities:
   help([topic]) - easy access to documentation
   dist(a,b) - distances between Coords or groups
   distsq(a,b) - distance squared between Coords
   angle(a,b,c,[d],[units]) - angle (a,b,c) or dihedral (a,b,c,d)
   (dist() and angle() also accept tuples of Coords as single arguments.)

Localizations:
   xyplot - internal use class for Data to launch a plotting program
   pdbview - internal use class for Molecule to launch a pdb viewer

Constants:
   backbone - names of atoms considered part of the backbone
   angleunits - definitions of 'rad', 'pi', and 'deg'
   angledefault - set to 'deg'
"""

# $Log: not supported by cvs2svn $
# Revision 1.1  2004/05/17 18:43:19  pett
# as distributed
#
# Revision 0.65  1996/05/24 01:19:11  jim
# Split into sub-modules, improved version reporting.
#
# Revision 0.64  1996/05/23 22:35:53  jim
# Added delrefs() to Molecule and contents to allow deletion.
# (Circular references prevent Python from deallocating objects.)
# Molecule.display() is now view().
# Molecule.display() and Data.plot() no longer take filenames.
# xyplotfunction() and pdbdisplayfunction() are now classes
# xyplot and pdbview so they keep data internally.
# There is a matching revision to md_local for this update.
#
# Revision 0.63  1996/05/22 15:26:37  jim
# Added structure information read from psf file to Molecule and
# buildstructure() to create useful structure on Molecule and Atoms.
# Modified dist() and angle() to accept tuples of Coords.
#
# Revision 0.62  1996/05/17 15:37:02  jim
# Small doc and error checking changes to Data.
#
# Revision 0.61  1996/05/17 15:00:30  jim
# Added flush() to end of DCDWrite.append() because the last
# frame appended to the file wasn't being recorded when the
# DCDWrite object was deleted with del().
#
# Revision 0.60  1996/05/08 16:31:20  jim
# Improved __repr__ for AtomGroup and subclasses.
# Changed DCD.dummymol() to DCD.asel().
# Fixed DCD.aselfree() to work without fixed atoms.
#
# Revision 0.59  1996/05/07 22:17:21  jim
# Started using RCS system.
#
# print "MDTools 0.58 beta (4/12/96) by James Phillips.",
# Added dummy atom creation methods to DCD.
# print "MDTools 0.57 beta (3/27/96) by James Phillips.",
# Enhanced molecule display to generate fewer leftover files.
# print "MDTools 0.56 beta (3/12/96) by James Phillips.",
# Added underscores to internal functions, split localization.
# print "MDTools 0.55 beta (3/11/96) by James Phillips.",
# Improved help system to reduce need for quotes.
# print "MDTools 0.54 beta (3/11/96) by James Phillips.",
# Added help system, fixed little bugs (unit vector from 0 and variance).
# print "MDTools for Python version 0.53 beta (2/12/96) by James Phillips"
# Fixed bug in Data.deviation().
# print "MDTools for Python version 0.52 beta (1/30/96) by James Phillips"
# Fixed old use of buildAtomList in RSel.__init__().
# print "MDTools for Python version 0.51 beta (1/18/96) by James Phillips"
# Allowed negative subscripts in DCD and fixed small bug in NAMDOutput.plot().
# print "MDTools for Python version 0.5 beta (12/18/95) by James Phillips"
# Original beta release.
# print " "+__name__+".help() for more info."

#print "MDTools "+"$Revision: 1.2 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2005-08-20 00:26:36 $"[7:-11]+") by James Phillips.  "+__name__+".help() for more info."

#
# Sub-Modules
#

_RCS = "$Id: md.py 26655 2009-01-07 22:02:30Z gregc $"
RCS = _RCS
RCS = RCS + '\n'

from .md_HomoCoord import *
from . import md_HomoCoord
RCS = RCS + md_HomoCoord._RCS
del(md_HomoCoord)

RCS = RCS + '\n'

from .md_AtomGroup import *
from . import md_AtomGroup
RCS = RCS + md_AtomGroup._RCS
del(md_AtomGroup)

RCS = RCS + '\n'

from .md_Trans import *
from . import md_Trans
RCS = RCS + md_Trans._RCS
del(md_Trans)

RCS = RCS + '\n'

from .md_DCD import *
from . import md_DCD
RCS = RCS + md_DCD._RCS
del(md_DCD)

RCS = RCS + '\n'

from .md_Data import *
from . import md_Data
RCS = RCS + md_Data._RCS
del(md_Data)

#
# Constants
#

RCS = RCS + '\n'

from .md_Constants import *
from . import md_Constants
RCS = RCS + md_Constants._RCS
del(md_Constants)

#
# Localizations
#

RCS = RCS + '\n'

from . import md_local
RCS = RCS + md_local._RCS
del(md_local)

#
# Help utility
#
_tips = {
'help':"""Function help([topic]): Prints documentation.

The topic may be any object, behavior varies.  Classes and Instances have __doc__ strings printed for that class and all superclasses.  Functions with names in the _tips dictionary have documentation strings printed.  Strings are searched for in the _tips dictionary and evaluated if not found.
""",
'angles':"""Units for angles are defined in angleunits.  Currently defined units are 'rad', 'pi', and 'deg'.  Angles have a default unit set in angledefault, which is normally 'deg'.

See also: angleunits, angledefault, angleconvert, angle
""",
'angleconvert':"""Function angleconvert(a,units,[newunits]): Converts angles.

See also: angleunits, angledefault, angle, 'angles'
""",
'dist':"""Function dist(a,b): Returns the distance between the objects a and b.  If a or b is a group then the closest distance is returned.
""",
'distsq':"""Function distsq(a,b): Returns the square of the distance between a and b more efficiently than squaring the result of dist(a,b).
""",
'angle':"""Function angle(a,b,c): Returns the angle between the coordinates a, b, and c.  angle(a,b,c,d) returns the dihedral angle.  Desired units may be appended to the parameters, as in angle(a,b,c,'rad').  Sorry, no angles between vectors.
"""
}
def help(name=None):
	if name is None:
		print(__doc__)
	elif isinstance(name, types.ClassType):
		if ( len(name.__bases__) ):
			classdesc = name.__name__ + "["
			for c in name.__bases__:
				help(c)
				classdesc = classdesc + c.__name__ + ","
			classdesc = classdesc[:-1] + "]"
		else:
			classdesc = name.__name__
		print("Class",classdesc+":",name.__doc__)
	elif isinstance(name, types.InstanceType):
		help(name.__class__)
	elif name in _tips.keys():
		print(_tips[name])
	elif hasattr(name,"__name__") and name.__name__ in _tips.keys():
		print(_tips[name.__name__])
	else:
		try:
			help(eval(name))
		except Exception:
			print('Evaluates to',repr(name)+'.')


