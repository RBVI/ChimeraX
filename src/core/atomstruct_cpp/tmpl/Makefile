TOP= ../../../..
include $(TOP)/mk/config.make

LIBNAME = tmpl
DIRNAME = $(shell basename `pwd`)
DIRNAME := atomstruct/$(DIRNAME:_cpp=)

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
INCS += -I$(includedir)
LIBS += -L$(libdir) -lioutil -lutil -lappdirs

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	-mkdir -p $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)
	$(RSYNC) $(SHLIB) $(shlibdir)
	if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
