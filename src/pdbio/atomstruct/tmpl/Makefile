TOP= ../../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
LIBNAME = tmpl
CXXFLAGS += -I../..
LIBS += ../../ioutil/libioutil.$(LIB_EXT) ../../util/libutil.$(LIB_EXT)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
