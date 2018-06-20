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

TOP = ../../../..
include $(TOP)/mk/config.make

SUBDIRS	+= atomstruct_cpp element_cpp pyinstance_cpp

include $(TOP)/mk/subdir.make

LIBNAME = molc

SRCS	= molc.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= 
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -Iinclude
ifeq ($(WIN32),msvc)
LIBS	+= /LIBPATH:lib libatomstruct.lib libelement.lib libpyinstance.lib libarrays.lib liblogger.lib libappdirs.lib $(PYTHON_LIB)
else
LIBS	+= -Llib -latomstruct -lelement -lpyinstance -larrays -llogger -lappdirs $(PYTHON_LIB)
endif

all: lib include data
	make install

lib:
	mkdir lib

include:
	mkdir include

data:
	mkdir data

cymol.$(PYMOD_EXT): cymol.pyx include/atomstruct/Atom.h
	LDFLAGS="-L." $(PYTHON_EXE) setup.py build_ext -i
	mv chimerax/atomic/cymol.cp*.$(PYMOD_EXT) cymol.$(PYMOD_EXT)
	rmdir -p chimerax/atomic

install: $(SHLIB) cymol.$(PYMOD_EXT)
	$(RSYNC) $(SHLIB) cymol.$(PYMOD_EXT) include lib ../src

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)
ifeq ($(OS),Darwin)
	install_name_tool -change @rpath/libatomstruct.dylib @loader_path/lib/libatomstruct.dylib $(SHLIB)
	install_name_tool -change @rpath/libelement.dylib @loader_path/lib/libelement.dylib $(SHLIB)
	install_name_tool -change @rpath/libpyinstance.dylib @loader_path/lib/libpyinstance.dylib $(SHLIB)
endif

clean:
	rm -rf __pycache__ $(OBJS) $(SHLIB) cymol.$(PYMOD_EXT) lib include data
ifdef WIN32
	rm -f lib$(LIBNAME).{exp,pdb} vc*.pdb
endif

pylint:
	$(PYLINT) $(PYSRCS)

molc.$(OBJ_EXT): include/atomstruct/Atom.h include/atomstruct/AtomicStructure.h \
	include/atomstruct/Bond.h include/atomstruct/Chain.h \
	include/atomstruct/ChangeTracker.h include/atomstruct/destruct.h \
	include/atomstruct/PBGroup.h include/atomstruct/PBManager.h \
	include/atomstruct/Pseudobond.h include/atomstruct/Residue.h \
	include/atomstruct/RibbonXSection.h include/atomstruct/Sequence.h \
	include/element/Element.h include/pyinstance/PythonInstance.declare.h \
	include/pyinstance/PythonInstance.instantiate.h

include/atomstruct/Atom.h: atomstruct_cpp.install
include/atomstruct/AtomicStructure.h: atomstruct_cpp.install
include/atomstruct/Bond.h: atomstruct_cpp.install
include/atomstruct/Chain.h: atomstruct_cpp.install
include/atomstruct/ChangeTracker.h: atomstruct_cpp.install
include/atomstruct/destruct.h: atomstruct_cpp.install
include/atomstruct/PBGroup.h: atomstruct_cpp.install
include/atomstruct/PBManager.h: atomstruct_cpp.install
include/atomstruct/Pseudobond.h: atomstruct_cpp.install
include/atomstruct/Residue.h: atomstruct_cpp.install
include/atomstruct/RibbonXSection.h: atomstruct_cpp.install
include/atomstruct/Sequence.h: atomstruct_cpp.install
include/element/Element.h: element_cpp.install
include/pyinstance/PythonInstance.declare.h: pyinstance_cpp.install
include/pyinstance/PythonInstance.instantiate.h: pyinstance_cpp.install

molobject.py: cymol.$(PYMOD_EXT)

atomstruct_cpp.install: element_cpp.install pyinstance_cpp.install
dssp_cpp.install: atomstruct_cpp.install
element_cpp.install: pyinstance_cpp.install
mmcif_cpp.install: readcif_cpp.install atomstruct_cpp.install
pdbio.install: atomstruct_cpp.install pdb_cpp.install
