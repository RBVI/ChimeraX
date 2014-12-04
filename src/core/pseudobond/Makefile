SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

#LIBNAME = pseudobond
CXXFLAGS += -I..
#LIBS += ../basegeom/libbasegeom.a
DIRNAME = $(shell basename `pwd`)

#$(SHLIB): imex.h $(OBJS)
#	$(SHLIB_LINK)

#install: $(SHLIB)

install:
	#$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
