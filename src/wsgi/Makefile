TOP = ../..
include $(TOP)/mk/config.make

install:
	$(RSYNC) chimera2.wsgi $(webdir)
	$(RSYNC) webapp_server.py $(PYSITEDIR)

clean:
