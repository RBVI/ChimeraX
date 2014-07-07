TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = mac_cpp
SRCS	= mac_cpp.cpp setfileicon.cpp touchevents.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
OBJS += setfileicon_mac.o
LIBS += -framework AppKit
OBJS += touchevents_mac.o
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS = -I$(includedir) $(PYTHON_INCLUDE_DIRS)

#OPT = -g -Wall
OPT = -O3

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix mac_cpp.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
