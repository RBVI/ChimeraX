TOP= ../../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = tmpl
SHLIB= lib$(LIBNAME).$(SHLIB_EXT)
CXXFLAGS += -I../..
LIBS += ../../ioutil/libioutil.a ../../util/libutil.a

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
