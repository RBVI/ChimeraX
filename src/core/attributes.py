# vim: set expandtab ts=4 sw=4:

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
Attributes: support for attribute discovery, and custom attibute session saving
===============================================================================

This module provides two main functionalities:

1) Attribute discovery -- making the attributes of classes available for inspection in user interfaces
(e.g. selection inspector; show attribute tool).

2) Custom attribute preservation -- allow classes that can have arbitrary new attributes defined to preserve
those attributes in session files.

Both capabilities are "on request": a bundle can submit information for its classes that it wants to be
inspectable, and/or ask that custom attributes be preservable for some or all of its classes.
"""

# Attribute discovery:
# provider is per-class
# bundle provides a list of (attr_name, attr_type, none_okay, option) for provided class
# The attr_type/option can be None, in which case some interfaces will skip the attr

# Custom attrs:
# At class definition, need to call a method of this module to add in the registration machinery,
# which will also have to possibly add to any attribute-discovery info and fire triggers.
