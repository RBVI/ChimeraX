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

_version = (2, 0, 0, "")

def get_version_string():
	return "%d.%d.%d%s" % _version

def get_version():
	return _version

def get_major_version():
	return _version[0]

def get_minor_version():
	return _version[1]

def get_micro_version():
	return _version[2]

def get_nano_version():
	return _version[3]
