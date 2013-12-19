TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui/oculus

PYMOD_NAME = _oculus
SRCS	= oculus.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
PYSRCS = __init__.py track.py

OCINC = -I/Users/goddard/Desktop/OculusSDK/LibOVR/Include
INCS += $(PYTHON_INCLUDE_DIRS) $(OCINC)

#LIBS	+= $(PYTHON_LIB) /Users/goddard/Desktop/OculusSDK/LibOVR/Lib/MacOS/Release/libovr.a
LIBS	+= $(PYTHON_LIB) /Users/goddard/Desktop/OculusSDK/LibOVR/Lib/MacOS/Debug/libovr.a
LIBS	+= -framework Carbon -framework IOKit

# libovr is compiled without rtti and without exceptions
OPT = -fno-rtti -fno-exceptions -g

all: $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: all pkg_dir
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

clean:
	rm -f $(PYMOD) $(OBJS)
	rm -rf __pycache__
