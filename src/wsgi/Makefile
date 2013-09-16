TOP = ../..
include $(TOP)/mk/config.make

install:
	$(RSYNC) webapp_server.py $(PYSITEDIR)

install-www:
	./install-wsgi

clean:

lint:
	pyflakes *.py
