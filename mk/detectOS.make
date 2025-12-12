# -*- mode: make -*- vim: set syntax=make:
# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Detect Operating System and architecture

OS=$(shell uname -s)
# We're on Windows
ifeq ($(filter $(OS),Linux Darwin),)
OS=$(shell uname -o)
ifneq ($(filter $(OS),Cygwin Msys),)
OS=Windows
endif
endif

# Linux
ifeq ($(OS),Linux)
	UNIX	= linux
	MACHINE	:= $(shell uname -m)
	ifeq ($(MACHINE),x86_64)
		OSARCH = Linux64
	else ifeq ($(MACHINE),ia64)
		OSARCH = LinuxIA64
	else ifeq ($(MACHINE),aarch64)
		OSARCH = LinuxArm64
	else
		# MACHINE is i686
		OSARCH = Linux
	endif
	DISTRO_NAME ?= $(shell . /etc/os-release && echo $$ID)
	DISTRO_VER ?= $(shell . /etc/os-release && echo $$VERSION_ID)
	# ID is "centos", "debian", "fedora", "rocky", or "ubuntu"
	ifeq ($(DISTRO_NAME),org.freedesktop.platform)
		FLATPAK_DIST = $(DISTRO_VER)
	else ifneq ($(filter $(DISTRO_NAME),centos rocky),)
		CENTOS_DIST = $(DISTRO_VER)
	else ifeq ($(DISTRO_NAME),debian)
		DEBIAN_DIST = $(DISTRO_VER)
	else ifeq ($(DISTRO_NAME),fedora)
		FEDORA_DIST = $(DISTRO_VER)
	else ifeq ($(DISTRO_NAME),ubuntu)
		UBUNTU_DIST = $(DISTRO_VER)
	endif
endif

# Mac OS 10, Aqua or X11 windowing systems
ifeq ($(OS),Darwin)
	UNIX	= macos
	OSXVER = $(shell sw_vers -productVersion | awk -F. '{ print $$1"."$$2 }')
	MACOSV1 = $(shell sw_vers -productVersion | awk -F. '{ print $$1 }')
	MACHINE	:= $(shell uname -m)
	ifeq ($(MACHINE),x86_64)
		OSARCH = DarwinIntel64
	else ifeq ($(MACHINE),arm64)
		OSARCH = DarwinArm64
	endif
endif

# Cygwin environment on Windows, don't care if NT versus 2000 ...
ifeq ($(OS),Windows)
	# Windows is what Python's platform.system() returns.
	OS = Windows
	MACHINE	:= $(shell uname -m)
	ifeq ($(MACHINE),x86_64)
		OSARCH = Windows64
	else ifeq ($(MACHINE),aarch64)
		OSARCH = WindowsArm64
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
