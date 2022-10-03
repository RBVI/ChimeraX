# -*- mode: make -*- vim: set syntax=make:
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

# Detect Operating System and architecture

OS	:= $(shell uname -s)
SHORT_OS := $(shell uname -o)

# Linux
ifeq ($(OS),Linux)
	UNIX	= linux
	MACHINE	:= $(shell uname -m)
	ifeq ($(MACHINE),x86_64)
		OSARCH = Linux64
	else ifeq ($(MACHINE),ia64)
		OSARCH = LinuxIA64
	else
		# MACHINE is i686
		OSARCH	= Linux
	endif
endif

# Mac OS 10, Aqua or X11 windowing systems
ifeq ($(OS),Darwin)
	UNIX	= macos
	OSXVER = $(shell sw_vers -productVersion | awk -F. '{ print $$1"."$$2 }')
	MACOSV1 = $(shell sw_vers -productVersion | awk -F. '{ print $$1 }')
	ifeq ($(shell uname -m),x86_64)
		OSARCH = DarwinIntel64
	else ifeq ($(shell uname -m),arm64)
		OSARCH = DarwinArm64
	endif
endif

# Cygwin environment on Windows, don't care if NT versus 2000 ...
ifneq ($(filter $(SHORT_OS),Cygwin Msys),)
	# Windows is what Python's platform.system() returns.
	OS = Windows
	#MACHINE	:= $(shell uname -m)
	MACHINE := $(shell test -d '$(shell cygpath -u '$(SYSTEMROOT)')/SysWow64' && echo x86_64)
	ifeq ($(MACHINE),x86_64)
		OSARCH = Windows64
	else
		OSARCH = Windows
	endif
	WIN32	= msvc
endif

ifndef UNIX
ifndef WIN32
$(error Unknown operating system $(OS))
endif
endif

ifndef OSARCH
$(error Unknown OS architecture)
endif
