TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = blob
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
ifneq ($(OS),Windows)
CXXFLAGS += -std=c++11 -stdlib=libc++ -Wno-invalid-offsetof -Wno-deprecated-writable-strings
endif
LIBS += $(PYTHON_LIB)
INCS += $(NUMPY_INC)
LIBS += -L$(shlibdir) -lmolecule

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(SHLIB) $(OBJS) imex.h
