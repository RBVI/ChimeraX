TOP= ../../../../..
include $(TOP)/mk/config.make

RESDIR = $(datadir)/idatm

install:
	-mkdir -p $(RESDIR)/templates
	$(RSYNC) templates/FORMAT templates/*.idatmres $(RESDIR)/templates

clean:
