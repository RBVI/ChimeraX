TOP	= ../..
include $(TOP)/mk/config.make

ifeq ($(WIN32),msvc)
CXXFLAGS = -I../llgr/c++ -I../win32 -DWIN32 $(OPT)
LIBS = ../llgr/libllgr.$(LIB_EXT) freeglut.lib glew32.lib msvcrt.lib
else
CXXFLAGS = -I../llgr/c++ $(OPT)
LIBS = -L../llgr/c++ -lllgr -lglut -lGLU -lGLEW -lGL
LDFLAGS = -Wl,-rpath=../llgr/c++
endif

SRCS = demo.cpp data.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
PROG_NAME = demo

$(PROG)$(PROG_EXT): $(OBJS)
	$(PROG_LINK)

data.cpp: webgl2c++.py one_webgl.json
	$(PYTHON_EXE) webgl2c++.py one_webgl.json data.cpp

clean:
	rm -f $(PROG)$(PROG_EXT) $(OBJS) data.cpp
ifeq ($(WIN32),msvc)
	rm -f $(PROG).pdb $(PROG).ilk vc90.pdb
endif

run:
	# by running within make, the right environment variables are setup
	./$(PROG_NAME)
