TOP	= ../../../../..
include $(TOP)/mk/config.make

INCS = -I..

SRCS = cifgrep.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
ifdef WIN32
LIBS = ../readcif.lib
else
LIBS = -L.. -lreadcif
endif
PROG_NAME = cifgrep

all: $(PROG) $(PROG_NAME).man

$(PROG): $(OBJS)
	$(PROG_LINK)

$(PROG_NAME).man: $(PROG_NAME).rst
	rst2man $< $@

install: all
	#cp $(PROG) $(bindir)
	#cp $(PROG_NAME).man $(man1dir)/$(PROG_NAME).1

clean:
	rm -f $(PROG) $(OBJS) $(PROG_NAME).man
