TOP	= ../../../../../..
include $(TOP)/mk/config.make

INCS = -I..

SRCS = cifgrep.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
ifdef WIN32
LIBS = ../libreadcif.lib
else
LIBS = -L.. -lreadcif
endif
PROG_NAME = cifgrep

all: $(PROG) 

$(PROG): $(OBJS)
	$(PROG_LINK)

install: all

clean:
	rm -f $(PROG) $(OBJS)
