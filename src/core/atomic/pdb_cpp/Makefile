TOP= ../../../..
include $(TOP)/mk/config.make

LIBNAME = pdb
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	rm -f $(LIBRARY) $(OBJS)
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
endif
