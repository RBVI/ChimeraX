TOP = ../../..
include $(TOP)/mk/config.make

JS = llgr_webgl.js webgl-debug.js webgl-utils.js

all:

install:
	$(RSYNC) $(JS) $(webdir)

# do syntax checking so debugging in browser is just runtime errors
syntax:
	rhino -w -strict llgr_webgl.js
