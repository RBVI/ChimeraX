# ChimeraX: installed as shiboken6/_chimerax_fast_enum.py by patch_pyside6.py.
"""
Create PySide's enum classes without general-purpose class-body processing.

libshiboken creates a Python enum class for every C++ enum of a Qt class
when the class is first used, with the functional API:
IntEnum(name, [(member name, value), ...], boundary=KEEP).  At ChimeraX
startup that is about 290 classes with 3,600 members. On Python 3.14,
constant integer members can be installed directly using the standard
Enum/Flag implementations and member maps. Other values and Python
versions use enum._simple_enum. The compatibility tests compare every
enum exposed by the installed PySide modules with the functional API.

Only functional-API creation of Enum, IntEnum, Flag or IntFlag with
boundary=KEEP and a list of (name, value) pairs, as libshiboken does it,
takes this path.  Looking up members, e.g. Qt.AlignmentFlag(1), does not
go through EnumType._create_.
"""

import builtins
import enum
import sys

_BASES = frozenset((enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag))
_INTEGER_ENUM = sys.version_info[:2] == (3, 14)
_create = enum.EnumType._create_
_create = getattr(_create, '_chimerax_original', _create)


def _integer_enum(cls, class_name, values):
    """The constant integer subset used by generated PySide bindings.

    Keep the standard library's Enum/Flag implementations and metadata. Avoid
    the temporary plain class and repeated descriptor/member checks needed by
    _simple_enum for user-defined classes. The caller validates this subset.
    """
    names, members, by_value, hashable = [], {}, {}, []
    member_type = cls._member_type_
    body = dict(_new_member_=member_type.__new__, _use_args_=cls._use_args_,
        _generate_next_value_=cls._generate_next_value_, _member_names_=names,
        _member_map_=members, _value2member_map_=by_value, _hashable_values_=hashable,
        _unhashable_values_=[], _unhashable_values_map_={}, _member_type_=member_type,
        _value_repr_=cls._value_repr_, __module__=__name__, __doc__='An enumeration.')
    is_flag = issubclass(cls, enum.Flag)
    if is_flag:
        body.update(_boundary_=enum.KEEP, _flag_mask_=None, _all_bits_=None,
                    _singles_mask_=None, _inverted_=None)
        for method in ('__or__', '__xor__', '__and__', '__ror__', '__rxor__', '__rand__', '__invert__'):
            body[method] = getattr(enum.Flag, method)
    result = builtins.type(class_name, (cls,), body, boundary=enum.KEEP, _simple=True)
    # Use the same method resolution rule as enum._simple_enum.
    set_class_attr = builtins.type.__setattr__
    for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
        if getattr(result, name) in (getattr(member_type, name), getattr(object, name)):
            set_class_attr(result, name, getattr(cls, name))
    single_bits = multi_bits = 0
    for name, value in values.items():
        member = by_value.get(value)
        if member is None:
            member = (member_type.__new__(result, value) if cls._use_args_
                      else member_type.__new__(result))
            member.__dict__.update(_value_=value, _name_=name, __objclass__=result,
                                   _sort_order_=len(names))
            by_value[value] = member
            hashable.append(value)
            if not is_flag or (value > 0 and value & (value - 1) == 0):
                names.append(name)
                single_bits |= value
            else:
                multi_bits |= value
        members[name] = member
        set_class_attr(result, name, member)
    if is_flag:
        mask = single_bits | multi_bits
        set_class_attr(result, '_flag_mask_', mask)
        set_class_attr(result, '_singles_mask_', single_bits)
        set_class_attr(result, '_all_bits_', (1 << mask.bit_length()) - 1)
        member_values = [member._value_ for member in result]
        if member_values != sorted(member_values):
            set_class_attr(result, '_iter_member_', result._iter_member_by_def_)
    set_class_attr(result, '__new__', enum.Enum.__new__)
    return result


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
            if _INTEGER_ENUM and all(name.__class__ is str and name and name[0].isupper()
                   and value.__class__ is int for name, value in body.items()):
                return _integer_enum(cls, class_name, body)
            plain = builtins.type(class_name, (), body)
            return enum._simple_enum(cls, boundary=enum.KEEP)(plain)
    return _create(cls, class_name, names, module=module, qualname=qualname,
                   type=type, start=start, boundary=boundary)


_create_._chimerax_fast_enum = True
_create_._chimerax_original = _create
if not getattr(enum.EnumType._create_, "_chimerax_fast_enum", False):
    enum.EnumType._create_ = _create_
