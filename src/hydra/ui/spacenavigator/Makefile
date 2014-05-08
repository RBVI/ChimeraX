TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui/spacenavigator

PYMOD_NAME = _spacenavigator
SRCS	= spacenav.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
PYSRCS = __init__.py snav.py snavmac.py

SNFW = /Library/Frameworks/3DconnexionClient.framework
SNINC = -I$(SNFW)/Versions/Current/Headers
INCS += $(PYTHON_INCLUDE_DIRS) $(SNINC)

LIBS	+= $(PYTHON_LIB) -F/Library/Frameworks -framework 3DconnexionClient

PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python

ifeq ($(shell test -e $(SNFW) && echo found),found)

all: $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix _spacenavigator.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PYMOD)
endif

install: all $(PKG_DIR)
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf $(PYMOD) $(OBJS) __pycache__

else

all:
install:
clean:

endif
