TOP= ../../../../..
include $(TOP)/mk/config.make

SUBDIRS	= idatm
include $(TOP)/mk/subdir.make

LIBNAME = tmpl
DIRNAME = $(shell basename `pwd`)
DIRNAME := atomstruct/$(DIRNAME:_cpp=)

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
INCS += -I$(includedir)
LIBS += -L$(libdir) -lelement -lioutil -lchutil -lappdirs

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	-mkdir -p $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)
	$(RSYNC) $(SHLIB) $(shlibdir)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

templates:
	cd leap-templates ; \
	for i in amino namino camino nucleic general ions; do \
		echo extracting $$i ; \
		./extract.csh $$i ; \
		mv $$i.cpp .. ; \
	done

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
