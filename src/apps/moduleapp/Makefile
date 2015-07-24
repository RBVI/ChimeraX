# module application
#
# Include this Makefile and set MOD_APP to the name of the
# chimera module for the application.  bin/MOD_APP will be
# the application and it invoke the chimera.MOD_APP module.
# APP_NAME is set in config.make.

TOP = ../../..
include $(TOP)/mk/config.make

#MOD_APP = 
PKG_DIR = $(PYSITEDIR)/chimera
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera
ifeq ($(OS),Darwin)
	OS_BIN = MacOS
else
	OS_BIN = bin
endif

all: $(MOD_APP)

install: all
	$(RSYNC) $(MOD_APP).py $(PKG_DIR)
	$(RSYNC) $(MOD_APP) $(bindir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(MOD_APP).py $(APP_PKG_DIR); \
		$(RSYNC) $(MOD_APP) $(app_bindir); \
	fi

clean:
	rm -f $(MOD_APP)

lint:
	$(PYLINT) $(MOD_APP).py

$(MOD_APP): ../moduleapp/mod_app.sh.in
	sed -e 's,APP_NAME,$(APP_NAME),' \
		-e 's,OS_BIN,$(OS_BIN),' \
		-e 's,MOD_APP,$(MOD_APP),' \
		< ../moduleapp/mod_app.sh.in > $(MOD_APP)
	chmod +x $(MOD_APP)
