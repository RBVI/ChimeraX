TOP	= ../../..
include $(TOP)/mk/config.make

INCS = -I'$(includedir)'

ifeq ($(WIN32),msvc)
CXXFLAGS += -DLLGR_DLL -DLLGR_EXPORT
LIBS = /link /LIBPATH:'$(libdir)' glew32.lib opengl32.lib msvcrt.lib
else
LIBS = $(OPENGL_LIBS)
endif

SRCS = ShaderProgram.cpp programs.cpp buffers.cpp matrices.cpp objects.cpp \
       optimize.cpp render.cpp spiral.cpp primitives.cpp vsphere.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
LIBNAME = llgr

#all: $(LIBRARY)
all: $(SHLIB)

install: all
	cp llgr.h llgr_ui.h $(includedir)
	cp $(SHLIB) $(shlibdir)

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(LIBRARY) $(SHLIB) $(OBJS)
ifeq ($(WIN32),msvc)
	rm -f $(SHLIB).manifest lib$(LIBNAME).{exp,pdb,ilk} vc90.pdb
endif
