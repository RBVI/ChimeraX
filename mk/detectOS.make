# Detect Operating System and architecture

OS	:= $(shell uname -s)

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
	ifeq ($(shell uname -m),i386)
		ifeq (,$(shell echo $(OSXVER) | sed -e 's/^10\.[0-5]$$//'))
			OSARCH = DarwinIntel
		else
			# 10.6 and above can run 64-bit binaries even though
			# kernel is 32-bit.
			OSARCH = DarwinIntel64
		endif
	else ifeq ($(shell uname -m),x86_64)
		OSARCH = DarwinIntel64
	else
		OSARCH = DarwinPPC
	endif
endif

# Cygwin environment on Windows, don't care if NT versus 2000 ...
OS	:= $(patsubst CYGWIN_NT%,CYGWIN_NT,$(OS))
ifeq ($(OS),CYGWIN_NT)
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
