TOP= ../../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
PYMOD_NAME = appdirs_cpp
LIBS+= -L$(shlibdir) -lappdirs

CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
ifneq ($(OS),Windows)
CXXFLAGS += -Wno-invalid-offsetof -Wno-deprecated-writable-strings
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PYSITEDIR); fi

clean:
	/bin/rm -f $(OBJS) $(PYMOD)
