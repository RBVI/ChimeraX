TOP= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = pdbio
PYMOD= $(LIBNAME).so
INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS)
LIBS += -L$(libdir) -latomstruct -ltmpl -llogger -lblob -lpdb -lbasegeom $(PYTHON_LIB)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
