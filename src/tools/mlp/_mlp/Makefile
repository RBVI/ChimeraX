TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR		= $(PYSITEDIR)/chimerax/mlp

PYMOD_NAME	= _mlp
SRCS		= mlp.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS)
LIBS		+= $(PYTHON_LIB)

ifdef WIN32
LIBS	+= libarrays.lib
else
LIBS	+= -larrays
endif

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
