TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR		= $(PYSITEDIR)/chimerax/seqalign/align_algs

PYMOD_NAME	= _sw
SRCS		= sw.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS)
LIBS		+= $(PYTHON_LIB)

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PKG_DIR)
endif
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
