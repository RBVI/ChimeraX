TOP = ../..
include $(TOP)/mk/config.make

install:
	$(RSYNC) webapp_server.py $(PYSITEDIR)
	./install-wsgi

clean:

lint:
	pyflakes *.py
	#for i in webapp_client/js/*.js webapp_client/js/c2/*.js; do
	for i in webapp_client/js/c2/*.js; do \
		echo $$i ; \
		rhino -w -strict -debug -version 150 $$i ; \
	done
