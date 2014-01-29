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

ifeq ($(shell test -e $(SNFW) && echo found),found)

all: $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: all pkg_dir
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf $(PYMOD) $(OBJS) __pycache__

else

all:
install:
clean:

endif
