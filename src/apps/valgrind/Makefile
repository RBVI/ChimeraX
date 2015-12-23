TOP	= ../../..
include $(TOP)/mk/config.make

VPROG	= $(APP_NAME).valgrind
CPROG	= $(APP_NAME).callgrind
SUPPRESS= valgrind.suppress

all: $(SUPPRESS)

install: all
	$(RSYNC) valgrind.sh $(bindir)/$(VPROG)
	$(RSYNC) $(SUPPRESS) $(datadir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) valgrind.sh $(app_bindir)/$(VPROG); \
		$(RSYNC) $(SUPPRESS) $(app_datadir); \
	fi
	$(RSYNC) callgrind.sh $(bindir)/$(CPROG)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) callgrind.sh $(app_bindir)/$(CPROG); \
	fi

clean:
	rm -f $(SUPPRESS)

$(SUPPRESS): $(wildcard *.supp)
	cat $^ > $@
