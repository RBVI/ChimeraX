TOP= ../../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = tmpl
CXXFLAGS += -I../..
LIBS += -L$(shlibdir) ../../ioutil/libioutil.a ../../util/libutil.a -lappdirs

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
