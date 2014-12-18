TOP	= ../../..
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
INCS = -I$(includedir)
endif

all: $(LIBRARY)
#all: $(SHLIB)

install: all
	cp readcif.h $(includedir)
	cp $(LIBRARY) $(libdir)
	#cp $(SHLIB) $(shlibdir)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(LIBRARY) $(OBJS)

cifgrep.install: $(LIBRARY)
	$(MAKE) -C cifgrep install
