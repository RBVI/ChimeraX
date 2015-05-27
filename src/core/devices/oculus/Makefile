TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/devices/oculus

PYMOD_NAME = _oculus
SRCS	= oculus.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
PYSRCS = __init__.py oculuscmd.py track.py

OCSDK = $(libdir)/LibOVR
OCINC = -I$(OCSDK)/Include
INCS += $(PYTHON_INCLUDE_DIRS) $(OCINC)

#LIBS	+= $(PYTHON_LIB) $(OCSDK)/Lib/Mac/Release/libovr.a
LIBS	+= $(PYTHON_LIB) $(OCSDK)/Lib/Mac/Debug/libovr.a
LIBS	+= -framework Cocoa -framework IOKit -framework OpenGL

PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python

# libovr is compiled without rtti and without exceptions
OPT = -fno-rtti -fno-exceptions -g -Wall

ifeq ($(shell test -e $(OCSDK) && echo found),found)
	HAVE_OCULUS_SDK = 1
else
	PYMOD = 
	PYOBJS = 
endif

all: $(PYMOD) $(PYOBJS)

ifdef HAVE_OCULUS_SDK

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix _oculus.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PYMOD)
endif

endif

install: all $(PKG_DIR)
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

clean:
	rm -f $(PYMOD) $(OBJS)
	rm -rf __pycache__
