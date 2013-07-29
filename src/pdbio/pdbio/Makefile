TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = pdbio
PYMOD= $(LIBNAME).so
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
LIBS= ../pdb/libpdb.a ../ioutil/libioutil.a ../molecule/libmolecule.a \
	../molecule/templates/libtmpl.a ../util/libutil.a

$(PYMOD): $(OBJS) $(LIBS)
	$(PYMOD_LINK)

install: $(PYMOD)

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
