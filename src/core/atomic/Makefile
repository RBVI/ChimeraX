TOP = ../../..
include $(TOP)/mk/config.make

SUBDIRS	+= atomstruct_cpp basegeom_cpp element_cpp mmcif_cpp \
	pdb_cpp pdbio readcif_cpp

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/atomic

PYSRCS = __init__.py changes.py mmcif.py molc.py molarray.py molobject.py molsurf.py \
	 pbgroup.py pdb.py readpbonds.py ribbon.py structure.py
LIBNAME = molc

SRCS	= molc.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= 
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= -L$(libdir) -latomstruct -lelement -larrays $(PYTHON_LIB)

all: $(PYOBJS) $(SHLIB)

install: all dir
	$(RSYNC) $(PYSRCS) $(SHLIB) $(PKG_DIR)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__ $(OBJS) $(SHLIB)

lint:
	$(PYLINT) $(PYSRCS)

molc.o: $(includedir)/atomstruct/Atom.h $(includedir)/atomstruct/Bond.h \
	$(includedir)/atomstruct/Chain.h $(includedir)/atomstruct/Pseudobond.h \
	$(includedir)/atomstruct/Residue.h $(includedir)/basegeom/destruct.h

$(includedir)/atomstruct/Atom.h: atomstruct_cpp.install
$(includedir)/atomstruct/Bond.h: atomstruct_cpp.install
$(includedir)/atomstruct/Chain.h: atomstruct_cpp.install
$(includedir)/atomstruct/Pseudobond.h: atomstruct_cpp.install
$(includedir)/atomstruct/Residue.h: atomstruct_cpp.install
$(includedir)/basegeom/destruct.h: basegeom_cpp.install

atomstruct_cpp.install: basegeom_cpp.install element_cpp.install pseudobond_cpp.install
mmcif_cpp.install: readcif_cpp.install atomstruct_cpp.install
pdbio.install: atomstruct_cpp.install pdb_cpp.install basegeom_cpp.install
pseudobond_cpp.install: basegeom_cpp.install
