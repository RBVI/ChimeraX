TOP	= ../..
include $(TOP)/mk/config.make

ifdef WIN32
INCS = -I'$(shell cygpath -m '$(includedir)')'
else
# TODO: remove this when everywhere
CXX += -fvisibility-ms-compat
INCS = -I$(includedir)
endif

SRCS = readcif.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
LIBNAME = readcif

all: $(LIBRARY)
#all: $(SHLIB)

install: all
	cp readcif.h $(includedir)
	cp $(LIBRARY) $(shlibdir)

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

clean:
	rm -f $(LIBRARY) $(OBJS)
