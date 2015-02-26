TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/devices/oculus

PYMOD_NAME = _oculus
SRCS	= oculus.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
PYSRCS = __init__.py track.py

OCSDK = $(libdir)
OCINC = -I$(OCSDK)/LibOVR/Include
INCS += $(PYTHON_INCLUDE_DIRS) $(OCINC)

#LIBS	+= $(PYTHON_LIB) $(OCSDK)/LibOVR/Lib/Mac/Release/libovr.a
LIBS	+= $(PYTHON_LIB) $(OCSDK)/LibOVR/Lib/Mac/Debug/libovr.a
LIBS	+= -framework Cocoa -framework IOKit -framework OpenGL

PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python

# libovr is compiled without rtti and without exceptions
OPT = -fno-rtti -fno-exceptions -g -Wall

ifeq ($(shell test -e $(OCSDK) && echo found),found)

all: $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix _oculus.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PYMOD)
endif

install: all $(PKG_DIR)
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -f $(PYMOD) $(OBJS)
	rm -rf __pycache__

else

all:
install:
clean:

endif
