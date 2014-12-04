TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = pdbio
PYMOD= $(LIBNAME).so
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS) -I$(includedir)
LIBS += -L$(shlibdir) -latomstruct -ltmpl -llogger -lblob ../pdb/libpdb.a ../basegeom/libbasegeom.a

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PYSITEDIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
