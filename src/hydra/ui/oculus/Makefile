TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui/oculus

PYMOD_NAME = _oculus
SRCS	= oculus.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
PYSRCS = __init__.py track.py

OCSDK = /Users/goddard/Desktop/OculusSDK
OCINC = -I$(OCSDK)/LibOVR/Include
INCS += $(PYTHON_INCLUDE_DIRS) $(OCINC)

#LIBS	+= $(PYTHON_LIB) $(OCSDK)/LibOVR/Lib/MacOS/Release/libovr.a
LIBS	+= $(PYTHON_LIB) $(OCSDK)/LibOVR/Lib/MacOS/Debug/libovr.a
LIBS	+= -framework Carbon -framework IOKit

# libovr is compiled without rtti and without exceptions
OPT = -fno-rtti -fno-exceptions -g

ifeq ($(shell test -e $(OCSDK) && echo found),found)

all: $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: all pkg_dir
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)

clean:
	rm -f $(PYMOD) $(OBJS)
	rm -rf __pycache__

else

all:
install:
clean:

endif
