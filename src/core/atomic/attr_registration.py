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

from ..state import State

class _NoDefault:
    pass

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
def register_attr(cls, registerer, attr_name, default_value=_NoDefault, attr_type=None):
    cls._attr_registration.register(registerer, attr_name, default_value, attr_type)

# used within the class to hold the registration info
class AttrRegistration(State):
    def __init__(self, class_name):
        self.reg_attr_info = {}
        self.class_name = class_name

    def get_attr(self, attr_name):
        try:
            registrant, default_value, attr_type = self.reg_attr_info[attr_name]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.class_name, attr_name))
        if default_value == _NoDefault:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.class_name, attr_name))
        return default_value

    def register(self, registrant, attr_name, default_value, attr_type):
        if attr_name in self.reg_attr_info:
            prev_registrant, prev_default, prev_type = self.reg_attr_info[attr_name]
            if prev_default == default_value and prev_type == attr_type:
                return
            raise ValueError("Registration of attr '%s' with %s by %s conflicts with previous"
                " registration by %s" % (attr_name, self.class_name, registrant, prev_registrant))
        self.reg_attr_info[attr_name] = (registrant, default_value, attr_type)

    # session functions
    def reset_state(self, session):
        self.reg_attr_info = {}

    def take_snapshot(self, session, flags):
        data = {'reg_attr_info': self.reg_attr_info, 'class_name': self.class_name}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        inst = AttrRegistration(data["class_name"])
        inst.reg_attr_info = data["reg_attr_info"]
        return inst

# used in session so that registered attributes get saved/restored
from . import Atom, AtomicStructure, Bond, Chain, CoordSet, Pseudobond, \
    PseudobondGroup, PseudobondManager, Residue, Sequence, Structure, StructureSeq

registerable_classes = [ Atom, AtomicStructure, Bond, Chain, CoordSet, Pseudobond,
    PseudobondGroup, PseudobondManager, Residue, Sequence, Structure, StructureSeq ]

class RegAttrManager(State):

    def __init__(self):
        for reg_class in registerable_classes:
            if not hasattr(reg_class, '_attr_registration'):
                reg_class._attr_registration = AttrRegistration(reg_class.__name__)
                reg_class.__getattr__ = __getattr__
                reg_class.register_attr = register_attr

    # session functions
    def reset_state(self, session):
        for reg_class in registerable_classes:
            reg_class._attr_registration.reset_state(session)

    def take_snapshot(self, session, flags):
        # force save of registration instance session info
        return { 'registrations':
            {rc.__name__: rc._attr_registration for rc in registerable_classes} }

    @staticmethod
    def restore_snapshot(session, data):
        class_map = { reg_class.__name__: reg_class for reg_class in registerable_classes }
        inst = RegAttrManager()
        for class_name, registration in data['registrations'].items():
            if class_name in class_map:
                class_map[class_name]._attr_registration = registration
        return inst
