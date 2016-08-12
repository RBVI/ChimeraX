# LineType values match X3D specification.  And those match the
# "Linetype Section of the International Register of Graphical Items" 
# <http://www.cgmopen.org/technical/registry/>.

from enum import Enum


class LineType(Enum):
    Solid = 1
    Dashed = 2
    Dotted = 3
    DashedDotted = 4
    DashDotDot = 5
    CustomLine = 16
