TOP= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
PYMOD_NAME = structaccess
INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS)
LIBS += -L$(libdir) -lblob

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

clean:
	/bin/rm -f $(PYMOD) $(OBJS)
