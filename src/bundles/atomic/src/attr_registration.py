# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
register_attr: infrastructure for molecular classes to register custom attributes
===============================

TODO
"""

from chimerax.core.state import State, StateManager

# something that can't be a default value, yet can be saved in sessions...
class _NoDefault(State):

    def take_snapshot(self, session, flags):
        return {'version':1}

    @staticmethod
    def restore_snapshot(session, data):
        return NO_DEFAULT
NO_DEFAULT = _NoDefault()

# methods that the manager will insert into the managed classes (inheritance won't work)
def __getattr__(self, attr_name, look_in_class=None):
    if look_in_class is None:
        look_in_class = self.__class__
    try:
        return look_in_class._attr_registration.get_attr(attr_name)
    except AttributeError:
        for base in look_in_class.__bases__:
            if hasattr(base, "__getattr__"):
                return base.__getattr__(self, attr_name, look_in_class=base)
        else:
            raise

@classmethod
def register_attr(cls, session, attr_name, registerer, *, default_value=NO_DEFAULT, attr_type=None,
        can_return_none=False):
    cls._attr_registration.register(session, attr_name, registerer, default_value, (attr_type, can_return_none))

# used within the class to hold the registration info
class AttrRegistration:
    def __init__(self, class_):
        self.reg_attr_info = {}
        self.class_ = class_
        self._session_attrs = {}
        self._ses_attr_counts = {}

    def get_attr(self, attr_name):
        try:
            registrant, default_value, type_info = self.reg_attr_info[attr_name]
        except KeyError:
            if attr_name in dir(self.class_):
                raise AttributeError("Execution of '%s' object's '%s' property raised AttributeError" % (self.class_.__name__, attr_name)) from None
            else:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.class_.__name__, attr_name)) from None
        if default_value == NO_DEFAULT:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.class_.__name__, attr_name))
        return default_value

    def register(self, session, attr_name, registrant, default_value, type_info):
        if attr_name in self.reg_attr_info:
            prev_registrant, prev_default, prev_type_info = self.reg_attr_info[attr_name]
            if prev_default == default_value and prev_type_info == type_info:
                return
            raise ValueError("Registration of attr '%s' with %s by %s conflicts with previous"
                " registration by %s" % (attr_name, self.class_.__name__, registrant, prev_registrant))
        self.reg_attr_info[attr_name] = (registrant, default_value, type_info)
        session_attrs = self._session_attrs.setdefault(session, set())
        if attr_name not in session_attrs:
            session_attrs.add(attr_name)
            self._ses_attr_counts[attr_name] = self._ses_attr_counts.get(attr_name, 0) + 1

    # session functions; called from manager, not directly from session-saving mechanism,
    # so API varies from that for State class
    def reset_state(self, session):
        if session not in self._session_attrs:
            return
        for attr_name in self._session_attrs[session]:
            if self._ses_attr_counts[attr_name] == 1:
                del self._ses_attr_counts[attr_name]
                del self.reg_attr_info[attr_name]
            else:
                self._ses_attr_counts[attr_name] -= 1
        del self._session_attrs[session]

    def take_snapshot(self, session, flags):
        ses_reg_attr_info = {}
        attr_names = self._session_attrs.get(session, [])
        for attr_name in attr_names:
            ses_reg_attr_info[attr_name] = self.reg_attr_info[attr_name]
        data = {'reg_attr_info': ses_reg_attr_info, 'version': 2}
        return data

    def restore_session_data(self, session, data):
        if data.get('version', 1) == 1:
            reg_attr_info = {}
            for attr_name, reg_info in data['reg_attr_info'].items():
                registrant, default_value, attr_type = reg_info
                reg_attr_info[attr_name] = (registrant, default_value, (attr_type, False))
        else:
            reg_attr_info = data['reg_attr_info']
        for attr_name, reg_info in reg_attr_info.items():
            self.register(session, attr_name, *reg_info)

# used in session so that registered attributes get saved/restored
from . import Atom, AtomicStructure, Bond, CoordSet, Pseudobond, \
    PseudobondGroup, PseudobondManager, Residue, Sequence, Structure, StructureSeq

# custom Chain attrs should be registered in the StructureSeq base class
registerable_classes = [ Atom, AtomicStructure, Bond, CoordSet, Pseudobond,
    PseudobondGroup, PseudobondManager, Residue, Sequence, Structure, StructureSeq ]

class RegAttrManager(StateManager):

    def __init__(self):
        for reg_class in registerable_classes:
            if not hasattr(reg_class, '_attr_registration'):
                reg_class._attr_registration = AttrRegistration(reg_class)
                reg_class.__getattr__ = __getattr__
                reg_class.register_attr = register_attr
        self._class_builtin_types = {}

    def attributes_returning(self, klass, return_types, *, none_okay=False):
        """Return list of attribute names for class 'klass' whose return types
           are the same or a subset of 'return_types'.  If 'none_okay' is True,
           then it is okay for the attribute to also possibly be None.
        """
        # builtin properties
        if klass not in self._class_builtin_types:
            self._class_builtin_types[klass] = builtin_types = {}
            for attr_name, attr_return_types in getattr(klass, '_cython_property_return_info', []):
                builtin_types[attr_name] = attr_return_types
        matching_attr_names = []
        for attr_name, types in self._class_builtin_types[klass].items():
            matched_one = False
            for t in types:
                if t in return_types:
                    matched_one = True
                else:
                    if not (t is None and none_okay):
                        break
            else:
                if matched_one:
                    matching_attr_names.append(attr_name)

        # registered attributes
        if hasattr(klass, '_attr_registration'):
            for attr_name, attr_info in klass._attr_registration.reg_attr_info.items():
                registrant, default_value, type_info = attr_info
                attr_type, can_return_none = type_info
                if attr_type not in return_types:
                    # this will also catch attr_type of None
                    continue
                if can_return_none and not none_okay:
                    continue
                matching_attr_names.append(attr_name)
        return matching_attr_names

    # session functions; there is one manager per session, and is only in charge of
    # remembering registrations from its session (atomic instances save their own
    # attrs)
    def reset_state(self, session):
        for reg_class in registerable_classes:
            reg_class._attr_registration.reset_state(session)

    def take_snapshot(self, session, flags):
        # force save of registration instance session info
        return {
            'version': 1,
            'registrations': {rc.__name__: rc._attr_registration.take_snapshot(session, flags)
                for rc in registerable_classes}
        }

    @staticmethod
    def restore_snapshot(session, data):
        inst = RegAttrManager()
        from . import bundle_api
        for class_name, registration in data['registrations'].items():
            bundle_api.get_class(class_name)._attr_registration.restore_session_data(
                session, registration)
        return inst

class CustomizedInstanceManager(StateManager):
    # This manager's only job is to remember instances that have
    # custom attibutes, so that they get saved/restored in sessions
    # even if there are no Python-layer references to them

    def reset_state(self, session):
        pass

    def take_snapshot(self, session, flags):
        from .molobject import all_python_instances
        # pure Sequence instances don't have 'session' attrs since they shouldn't
        # be saved if nothing else in the Python layer wants them saved
        return { 'instances': [inst for inst in all_python_instances()
            if inst.has_custom_attrs and getattr(inst, 'session', None) == session] }

    @staticmethod
    def restore_snapshot(session, data):
        # simply having the Python instances in 'data' restored was the whole point,
        # so mission accomplished already
        return CustomizedInstanceManager()
