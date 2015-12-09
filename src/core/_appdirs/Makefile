TOP= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core

PYMOD_NAME = _appdirs
SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS)
LIBS += -L$(libdir) -lappdirs


$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	#if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
