TOP	= ../../../..
include $(TOP)/mk/config.make

NO_SUBDIR_INSTALL = 1
SUBDIRS = cifgrep

include $(TOP)/mk/subdir.make

LIBNAME = readcif

SRCS = readcif.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
ifdef WIN32
INCS = -I'$(shell cygpath -m '$(includedir)')'
else
# TODO: remove this when everywhere
CXX += -fvisibility-ms-compat
endif

all: $(LIBRARY)
#all: $(SHLIB)

install: all
	$(RSYNC) readcif.h $(includedir)
	$(RSYNC) $(LIBRARY) $(libdir)
	#cp $(SHLIB) $(shlibdir)
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) readcif.h $(app_includedir); \
	fi
ifdef WIN32
	if [ -d "$(app_libdir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
	fi
endif
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(LIBRARY) $(OBJS)
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
endif

cifgrep.install: $(LIBRARY)
	$(MAKE) -C cifgrep install
