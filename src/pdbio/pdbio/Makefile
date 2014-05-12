TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = pdbio
PYMOD= $(LIBNAME).so
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
LIBS += -L$(shlibdir) -latomstruct -ltmpl -lblob -lconnectivity ../pdb/libpdb.a ../base-geom/libbase-geom.a

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
