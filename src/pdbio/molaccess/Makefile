TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
PYMOD_NAME = molaccess
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
ifneq ($(OS),Windows)
CXXFLAGS += -Wno-invalid-offsetof -Wno-deprecated-writable-strings
endif
LIBS += -L$(shlibdir) -lblob

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)

clean:
	/bin/rm -f $(PYMOD) $(OBJS)
