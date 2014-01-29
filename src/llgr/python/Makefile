TOP	= ../../..
include $(TOP)/mk/config.make

PATCHES = $(wildcard *.patch)

WRAPPY = $(bindir)/wrappy3
ifeq (64,$(patsubst %64,64,$(OSARCH)))
WRAPPY	+= -L
ifeq ($(OSARCH),Windows64)
# size_t is unsigned long long on Windows64 
WRAPPY	+= -L
endif
endif

ifeq ($(WIN32),msvc)
	LIBS = /link /LIBPATH:'$(shell cygpath -m '$(libdir)')' libllgr.lib libwrappy3.lib $(PYTHON_LIB) glew32.lib msvcrt.lib
else
LIBS = -L$(libdir) -lllgr -lwrappy3 $(OPENGL_LIBS)
endif

ifdef WIN32
CXXFLAGS += -I'$(shell cygpath -m '$(includedir)')'
else
CXXFLAGS += -I$(includedir)
endif
CXXFLAGS += $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) $(PYDEF)

SRCS = MemoryMap.cpp
GENSRCS += _llgr.cpp AttributeInfo_Object.cpp
GENHDRS += $(GENSRCS:.cpp=.h)
OBJS = $(SRCS:.cpp=.$(OBJ_EXT)) $(GENSRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _llgr

PKG_DIR = $(PYSITEDIR)/llgr
PYSRCS = __init__.py dump.py pyopengl.py spiral.py shader.py

HDRS = ../c++/llgr.h ../c++/llgr_ui.h MemoryMap.h

all: $(PYMOD) $(PYOBJS)

install: all
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	cp $(PYMOD) $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

$(GENSRCS) $(OBJS): wrappy

wrappy: $(HDRS)
	$(WRAPPY) -d -N llgr -m _llgr -e LLGR_IMEX $(HDRS)
ifneq (,$(PATCHES))
	for i in $(PATCHES) ; do \
		patch -b < $$i ; \
	done
endif
	touch wrappy

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf $(PYMOD) $(OBJS) $(GENSRCS) $(GENHDRS) wrappy *.orig *.rej __pycache__
ifeq ($(WIN32),msvc)
	rm -f $(PYMOD).manifest $(PYMOD_NAME).{lib,exp,ilk,pdb} vc90.pdb

endif
