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
RUN = $(CHIMERAX_EXE) --nogui --exit --cmd

PYSRCS = $(wildcard src/*.py)
CSRCS = search_cpp/imex.h search_cpp/search.cpp search_cpp/search.h \
	search_cpp/ast.pyx search_cpp/ast.pxd
SRCS = $(PYSRCS) $(CSRCS)

wheel:	bundle_info.xml $(SRCS)
	$(RUN) "devel build . exit true test false"

search_cpp/imex.h:
	sed -e 's/LIBNAME/ATOM_SEARCH/' < $(includedir)/imex.i > $@

install app-install:	bundle_info.xml $(SRCS)
	$(RUN) "devel install . user false exit true test false"

test:
	for t in $(wildcard test*.cxc) $(wildcard test*.py);\
		do $(CHIMERAX_EXE) --exit --nogui $$t;\
	done

debug:
	$(CHIMERAX_EXE) --debug

clean:
	if [ -x $(CHIMERAX_EXE) ]; then \
		$(RUN) "devel clean . exit true" ; \
	else \
		rm -rf build dist *.egg-info src/__pycache__ ; \
	fi
	rm -f search_cpp/search.o* search_cpp/imex.h search_cpp/ast.cpp \
		src/lib/libatom_*

pylint:
	$(CHIMERAX_EXE) -m flake8 $(PYSRCS)
