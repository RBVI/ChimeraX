# ChimeraX: installed as shiboken6/_chimerax_fast_enum.py by patch_pyside6.py.
"""
Create PySide's enum classes with enum._simple_enum.

libshiboken creates a Python enum class for every C++ enum of a Qt class
when the class is first used, with the functional API:
IntEnum(name, [(member name, value), ...], boundary=KEEP).  At ChimeraX
startup that is about 290 classes with 3,600 members, ~26 ms.  The
standard library's enum._simple_enum, which it uses for its own enums,
builds the same classes in about 60% of the time; enum._test_simple_enum
finds no difference for any of the 826 enums of the PySide modules
ChimeraX ships.

Only functional-API creation of Enum, IntEnum, Flag or IntFlag with
boundary=KEEP and a list of (name, value) pairs, as libshiboken does it,
takes this path.  Looking up members, e.g. Qt.AlignmentFlag(1), does not
go through EnumType._create_.
"""

import builtins
import enum

_BASES = frozenset((enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag))
_create = enum.EnumType._create_


def _create_(cls, class_name, names, *, module=None, qualname=None, type=None,
             start=1, boundary=None):
    if (boundary is enum.KEEP and cls in _BASES and names.__class__ is list
            and module is None and qualname is None and type is None):
        body = {}
        for item in names:
            if item.__class__ is not tuple or len(item) != 2 or item[0] in body:
                break
            body[item[0]] = item[1]
        else:
            plain = builtins.type(class_name, (), body)
            return enum._simple_enum(cls, boundary=enum.KEEP)(plain)
    return _create(cls, class_name, names, module=module, qualname=qualname,
                   type=type, start=start, boundary=boundary)


if not getattr(enum.EnumType._create_, "_chimerax_fast_enum", False):
    _create_._chimerax_fast_enum = True
    enum.EnumType._create_ = _create_
