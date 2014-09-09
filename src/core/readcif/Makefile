TOP	= ../../..
include $(TOP)/mk/config.make

NO_SUBDIR_INSTALL = 1
SUBDIRS = cifgrep

include $(TOP)/mk/subdir.make

# TODO: remove this when everywhere
CXX += -fvisibility-ms-compat

ifdef WIN32
INCS = -I'$(shell cygpath -m '$(includedir)')'
else
INCS = -I$(includedir)
endif

SRCS = readcif.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
LIBNAME = readcif

all: $(LIBRARY)
#all: $(SHLIB)

install: all
	cp readcif.h $(includedir)
	cp $(LIBRARY) $(libdir)
	#cp $(SHLIB) $(shlibdir)

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(LIBRARY) $(OBJS)

cifgrep.install: $(LIBRARY)
	$(MAKE) -C cifgrep install
