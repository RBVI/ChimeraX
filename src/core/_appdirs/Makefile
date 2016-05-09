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
	#if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
