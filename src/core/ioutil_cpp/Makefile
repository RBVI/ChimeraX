TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = ioutil
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))

ifdef WIN32
CSRCS	= win32_dirent.c
OBJS	+= $(CSRCS:.c=.$(OBJ_EXT))
endif

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	rm -f $(OBJS) $(LIBRARY)
ifdef WIN32
	rm -f lib$(LIBNAME).{exp,pdb} vc*.pdb
endif
