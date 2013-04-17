TOP	= ../../..
include $(TOP)/mk/config.make

PATCHES = $(wildcard *.patch)

WRAPPY = $(bindir)/wrappy3
ifeq ($(WIN32),msvc)
	LIBS = /link /LIBPATH:"$(libdir)" libllgr.lib libwrappy3.lib $(PYTHON_LIB) glew32.lib msvcrt.lib
else
LIBS = -L$(libdir) -lllgr -lwrappy3 $(OPENGL_LIBS)
endif

CXXFLAGS += -I$(includedir) -I$(PYTHON_INCLUDE_DIR) $(NUMPY_INC) $(PYDEF)

GENSRCS += _llgr.cpp AttributeInfo_Object.cpp
GENHDRS += $(GENSRCS:.cpp=.h)
OBJS = $(GENSRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _llgr

PKG_DIR = $(PYTHON_SITE_PACKAGES)
PYSRCS = llgr.py llgr_dump.py
PYOBJS = $(PYSRCS:.py=.pyc)

HDRS = ../c++/llgr.h ../c++/llgr_ui.h

all: $(PYMOD) $(PYOBJS)

install: all
	cp $(PYMOD) $(pymoddir)
	$(RSYNC) $(PYSRCS) typecheck $(PKG_DIR)

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

clean:
	rm -f $(PYMOD) $(OBJS) $(GENSRCS) $(GENHDRS) wrappy *.orig *.rej
ifeq ($(WIN32),msvc)
	rm -f $(PYMOD).manifest $(PYMOD_NAME).{lib,exp,ilk,pdb} vc90.pdb
endif
