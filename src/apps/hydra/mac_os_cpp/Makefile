TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = mac_os_cpp
SRCS	= memory.cpp module.cpp repaint.cpp setfileicon.cpp touchevents.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
OBJS += repaint_mac.o setfileicon_mac.o touchevents_mac.o
LIBS += -framework AppKit
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
# Use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
