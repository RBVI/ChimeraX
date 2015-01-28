# Use this make file to make a frozen verson of chimera for debugging

TOP	= ../../..
include $(TOP)/mk/config.make

PROG	= $(APP_NAME).valgrind
SUPPRESS= valgrind.suppress

all: $(SUPPRESS)

install: all
	$(RSYNC) app.sh $(bindir)/$(PROG)
	$(RSYNC) $(SUPPRESS) $(datadir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) app.sh $(app_bindir)/$(PROG); \
		$(RSYNC) $(SUPPRESS) $(app_datadir); \
	fi

clean:
	rm -f $(SUPPRESS)

$(SUPPRESS): $(wildcard *.supp)
	cat $^ > $@
