TOP= ../../../../..
include $(TOP)/mk/config.make

SUBDIRS	= idatm
include $(TOP)/mk/subdir.make

#LIBNAME = tmpl
DIRNAME = $(shell basename `pwd`)
DIRNAME := atomstruct/$(DIRNAME:_cpp=)

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
INCS += -I$(includedir)
#ifeq ($(WIN32),msvc)
#LIBS += libelement.lib libioutil.lib libchutil.lib libappdirs.lib
#else
#LIBS += -lelement -lioutil -lchutil -lappdirs
#endif

all: $(OBJS)

install: all
	-mkdir -p $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

templates:
	cd leap-templates ; \
	for i in amino namino camino nucleic general ions; do \
		echo extracting $$i ; \
		./extract.csh $$i ; \
		mv $$i.cpp .. ; \
	done

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
