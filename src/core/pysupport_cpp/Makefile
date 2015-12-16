TOP= ../../..
include $(TOP)/mk/config.make

DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)
INCS	+= -I$(includedir)

install:
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS)
