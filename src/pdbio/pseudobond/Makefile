SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.(OBJ_EXT))
TOP= ../../..
include $(TOP)/mk/config.make

#LIBNAME = pseudobond
CXXFLAGS += -I..
#LIBS += ../basegeom/libbasegeom.$(LIB_EXT)
DIRNAME = $(shell basename `pwd`)

#$(SHLIB): imex.h $(OBJS)
#	$(SHLIB_LINK)

#install: $(SHLIB)

install:
	#$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
