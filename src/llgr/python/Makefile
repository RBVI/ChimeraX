TOP	= ../../..
include $(TOP)/mk/config.make

PATCHES = $(wildcard *.patch)

WRAPPY = $(bindir)/wrappy3
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

GENSRCS += _llgr.cpp AttributeInfo_Object.cpp
GENHDRS += $(GENSRCS:.cpp=.h)
OBJS = $(GENSRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _llgr

PKG_DIR = $(PYSITEDIR)/llgr
PYSRCS = __init__.py dump.py pyopengl.py spiral.py shader.py

HDRS = ../c++/llgr.h ../c++/llgr_ui.h

all: $(PYMOD) $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	cp $(PYMOD) $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

$(GENSRCS): wrappy

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
