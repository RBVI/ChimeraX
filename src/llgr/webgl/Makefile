TOP = ../../..
include $(TOP)/mk/config.make

JS = llgr_webgl.js webgl-debug.js webgl-utils.js

all:

install:
	$(RSYNC) $(JS) $(webdir)

# do syntax checking so debugging in browser is just runtime errors
lint:
	for i in *.js; do \
		echo $$i ; \
		rhino -w -strict -debug -version 150 $$i ; \
	done
