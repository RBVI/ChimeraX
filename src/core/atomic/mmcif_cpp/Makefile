TOP= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/atomic

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

CXXFLAGS += -I$(includedir) $(PYTHON_INCLUDE_DIRS)
LIBS += -L$(shlibdir) -latomstruct -lelement -ltmpl -lbasegeom -lreadcif -lwrappy3 -llogger -larrays

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	#if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

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

mmcif.o: $(includedir)/atomstruct/Atom.h \
	$(includedir)/atomstruct/AtomicStructure.h $(includedir)/atomstruct/Bond.h \
	$(includedir)/atomstruct/CoordSet.h $(includedir)/atomstruct/Residue.h \
	$(includedir)/atomstruct/connect.h $(includedir)/atomstruct/tmpl/Atom.h \
	$(includedir)/atomstruct/tmpl/Residue.h \
	$(includedir)/atomstruct/string_types.h $(includedir)/logger/logger.h
