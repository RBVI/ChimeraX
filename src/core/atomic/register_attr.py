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

from .. import State

class _NoDefault:
    pass

#inherited by the class to offer attr-registration functionality
class AttrRegistrant:
    def __getattr__(self, attr_name):
        return self._attr_registration.get_attr(attr_name, self.__class__.__name__)

    def register_attr(self, registerer, attr_name, default_value=_NoDefault, attr_type=None):
        self._attr_registration.register(registerer, attr_name, default_value, attr_type)

# used within the class to hold the registration info
class AttrRegistration(State):
    def __init__(self):
        self._reg_attr_info = {}

    def get_attr(self, attr_name, class_name):
        try:
            registrant, default_value, attr_type = self._reg_attr_info[attr_name]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" % (class_name, attr_name)
        if default_value == _NoDefault:
            raise AttributeError("'%s' object has no attribute '%s'" % (class_name, attr_name)
        return default_value

    def register(self, registrant, attr_name, default_value, attr_type):
        if attr_name in self._reg_attr_info:
            prev_registrant, prev_default, prev_type = self._reg_attr_info[attr_name]
            if prev_default == default_value and prev_type == attr_type:
                return
            raise ValueError("Registration of attr '%s' by %s conflicts with previous registration"
                " by %s" % (attr_name, registrant, prev_registrant))
        self._reg_attr_info[attr_name] = (registrant, default_value, attr_type)

    # session functions
    def reset_state(self, session):
        self._reg_attr_info = {}

    def take_snapshot(self, session, flags):
        data = {'reg_attr_info': self._reg_attr_info}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        inst = AttrRegistrant()
        inst._reg_attr_info = data["reg_attr_info"]
