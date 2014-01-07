TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = access
PYMOD= $(LIBNAME).so
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS) -std=c++11 -stdlib=libc++ -Wno-invalid-offsetof -Wno-deprecated-writable-strings
LIBS += -L$(shlibdir) -lmolecule
INCS += $(NUMPY_INC)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)

clean:
	/bin/rm -f $(PYMOD) $(OBJS)
