TOP= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core

PYMOD_NAME = _appdirs
SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS)
ifeq ($(WIN32),msvc)
LIBS += libappdirs.lib
else
LIBS += -lappdirs
endif


$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PKG_DIR)
endif
endif

clean:
	rm -f $(OBJS) $(PYMOD)
ifdef WIN32
	rm -f $(PYMOD_NAME).pdb vc*.pdb
endif
