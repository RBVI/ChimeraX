TOP = ../../..
include $(TOP)/mk/config.make

SUBDIRS	+= atomstruct_cpp element_cpp mmcif_cpp pdb_cpp pdbio readcif_cpp

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/atomic

PYSRCS = __init__.py changes.py mmcif.py molc.py molarray.py molobject.py molsurf.py \
	 path.py pbgroup.py pdb.py pdbmatrices.py readpbonds.py ribbon.py structure.py
LIBNAME = molc

SRCS	= molc.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= 
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifeq ($(WIN32),msvc)
LIBS	+= libatomstruct.lib libelement.lib libarrays.lib $(PYTHON_LIB)
else
LIBS	+= -latomstruct -lelement -larrays $(PYTHON_LIB)
endif

all: $(PYOBJS) $(SHLIB)

install: all dir
	$(RSYNC) $(PYSRCS) $(SHLIB) $(PKG_DIR)
ifdef WIN32
	#$(RSYNC) $(LIBRARY) $(libdir)
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb $(PKG_DIR)
endif
endif

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__ $(OBJS) $(SHLIB)

lint:
	$(PYLINT) $(PYSRCS)

molc.$(OBJ_EXT): $(includedir)/atomstruct/Atom.h $(includedir)/atomstruct/AtomicStructure.h \
	$(includedir)/atomstruct/Bond.h $(includedir)/atomstruct/Chain.h \
	$(includedir)/atomstruct/ChangeTracker.h $(includedir)/atomstruct/destruct.h \
	$(includedir)/atomstruct/PBGroup.h $(includedir)/atomstruct/Pseudobond.h \
	$(includedir)/atomstruct/Residue.h $(includedir)/atomstruct/RibbonXSection.h \
	$(includedir)/atomstruct/Sequence.h

$(includedir)/atomstruct/Atom.h: atomstruct_cpp.install
$(includedir)/atomstruct/AtomicStructure.h: atomstruct_cpp.install
$(includedir)/atomstruct/Bond.h: atomstruct_cpp.install
$(includedir)/atomstruct/Chain.h: atomstruct_cpp.install
$(includedir)/atomstruct/ChangeTracker.h: atomstruct_cpp.install
$(includedir)/atomstruct/destruct.h: atomstruct_cpp.install
$(includedir)/atomstruct/PBGroup.h: atomstruct_cpp.install
$(includedir)/atomstruct/Pseudobond.h: atomstruct_cpp.install
$(includedir)/atomstruct/Residue.h: atomstruct_cpp.install
$(includedir)/atomstruct/RibbonXSection.h: atomstruct_cpp.install
$(includedir)/atomstruct/Sequence.h: atomstruct_cpp.install

atomstruct_cpp.install: element_cpp.install
mmcif_cpp.install: readcif_cpp.install atomstruct_cpp.install
pdbio.install: atomstruct_cpp.install pdb_cpp.install
