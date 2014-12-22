TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = appdirs
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)
	if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h
