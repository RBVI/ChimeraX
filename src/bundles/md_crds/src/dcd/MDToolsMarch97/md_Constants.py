"""Constant Definitions of MDTools

RCS: $Id: md_Constants.py 26655 2009-01-07 22:02:30Z gregc $

Constants:
   backbone - names of atoms considered part of the backbone
   angleunits - definitions of 'rad', 'pi', and 'deg'
   angledefault - set to 'deg'
"""

_RCS = "$Id: md_Constants.py 26655 2009-01-07 22:02:30Z gregc $"

# $Log: not supported by cvs2svn $
# Revision 0.65  1996/05/24 01:23:33  jim
# Split into sub-modules, improved version reporting.
#

#print "- Constants "+"$Revision: 1.1 $"[11:-1]+"$State: Exp $"[8:-1]+"("+"$Date: 2004-05-17 18:43:19 $"[7:-11]+")"

#
# Constant definitions
#

import math

backbone = ('N','HN','CA','HA','H','C','O','HT1','HT2','HT3','OT1','OT2')

angleunits = {'rad':1.,'pi':math.pi,'deg':math.pi/180.}
angledefault = 'deg'

