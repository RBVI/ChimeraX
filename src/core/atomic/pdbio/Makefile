TOP= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/atomic

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = pdbio
DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS)
ifdef WIN32
LIBS += libatomstruct.lib libelement.lib liblogger.lib libpdb.lib libarrays.lib $(PYTHON_LIB)
else
LIBS += -latomstruct -lelement -llogger -lpdb -larrays $(PYTHON_LIB)
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	#if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)

PDBio.o: $(includedir)/pdb/PDB.h $(includedir)/atomstruct/Atom.h \
	$(includedir)/atomstruct/AtomicStructure.h $(includedir)/atomstruct/Bond.h \
	$(includedir)/atomstruct/connect.h $(includedir)/atomstruct/CoordSet.h \
	$(includedir)/atomstruct/destruct.h $(includedir)/atomstruct/Residue.h \
	$(includedir)/atomstruct/Sequence.h $(includedir)/logger/logger.h
