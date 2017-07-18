# The "make" targets are:
# 	wheel: build a Python wheel in "dist" directory.
# 	install: build wheel (if needed) and install in ChimeraX.
# 	test: run ChimeraX
# 	debug: run ChimeraX with debugging flag set
# 	clean: remove files used in building wheel

CHIMERAX_APP = ../../../ChimeraX.app

# Platform-dependent settings.  Should not need fixing.
# For Windows, we assume Cygwin is being used.
OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
ifeq ($(OS),CYGWIN_NT)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif

PYSRCS = $(wildcard src/*.py)
CSRCS = $(wildcard _mmtf/*.cpp)
SRCS = $(PYSRCS) $(CSRCS)
HEADERS = $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE)

MMTF_CPP_VERSION = master
MMTF_CPP_DISTRIBUTION = mmtf-cpp-$(MMTF_CPP_VERSION).zip
MMTF_CPP_SOURCE	= mmtf-cpp-$(MMTF_CPP_VERSION)
MSGPACK_VERSION = master
MSGPACK_DISTRIBUTION = msgpack-c-$(MSGPACK_VERSION).zip
MSGPACK_SOURCE	= msgpack-c-$(MSGPACK_VERSION)

wheel: bundle_info.xml $(SRCS) $(HEADERS)
	$(CHIMERAX_EXE) --nogui --cmd "devel build . exit true"

install app-install: bundle_info.xml $(SRCS) $(HEADERS)
	$(CHIMERAX_EXE) --nogui --cmd "devel install . user false exit true"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(CHIMERAX_EXE) --nogui --cmd "devel clean . exit true"
	rm -rf $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE)

pylint:
	$(CHIMERAX_EXE) -m flake8 $(filter %.py, $(SRCS))

$(MMTF_CPP_SOURCE): $(MMTF_CPP_DISTRIBUTION)
	# only need the include files
	unzip -u $(MMTF_CPP_DISTRIBUTION) $(MMTF_CPP_SOURCE)/include/*

$(MSGPACK_SOURCE): $(MSGPACK_DISTRIBUTION)
	# only need the include files
	unzip -u $(MSGPACK_DISTRIBUTION) $(MSGPACK_SOURCE)/include/*
