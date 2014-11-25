TOP= ../../..
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

PATCHES = $(wildcard *.patch)

HDRS = mmcif.h
SRCS = mmcif.cpp template.cpp
GENSRCS += _mmcif.cpp
GENHDRS += $(GENSRCS:.cpp=.h)
OBJS = $(SRCS:.cpp=.$(OBJ_EXT)) $(GENSRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _mmcif

CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS) -I$(includedir)
LIBS += -L$(shlibdir) -latomstruct -ltmpl -lblob ../basegeom/libbasegeom.a -lreadcif -lwrappy3

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)

clean:
	/bin/rm -f $(PYMOD) $(OBJS) $(GENSRCS) $(GENHDRS) wrappy *.orig *.rej
ifeq ($(WIN32),msvc)
	rm -f $(PYMOD).manifest $(PYMOD_NAME).{lib,exp,ilk,pdb} vc90.pdb
endif

$(GENSRCS) $(OBJS): wrappy

wrappy: $(HDRS)
	$(WRAPPY) -d -N mmcif -m _mmcif $(HDRS)
ifneq (,$(PATCHES))
	for i in $(PATCHES) ; do \
		patch -b < $$i ; \
	done
endif
	touch wrappy
