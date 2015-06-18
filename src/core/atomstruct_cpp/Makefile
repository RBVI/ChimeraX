TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS	= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)
INCS	+= -I$(includedir)
ifneq ($(OS),Windows)
CXXFLAGS	+= -Wno-logical-op-parentheses
seq_assoc.o : CXXFLAGS += -Wno-sign-compare
endif
LIBS	+= -L$(libdir) -ltmpl -llogger -lbasegeom

$(SHLIB): imex.h $(OBJS) $(SUBDIR_INSTALL)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h

$(SUBDIR_INSTALL): imex.h
