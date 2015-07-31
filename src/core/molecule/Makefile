TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/molecule
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/molecule

PYSRCS = __init__.py molc.py molarray.py molobject.py
LIBNAME = molc

SRCS	= molc.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= 
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= -L$(libdir) -latomstruct -larrays $(PYTHON_LIB)

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
