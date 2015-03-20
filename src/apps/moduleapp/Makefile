# module application
#
# Include this Makefile and set APP to the name for the
# application.  bin/APP will be the application and it
# invoke the APP.py module in the chimera namespace.

TOP = ../../..
include $(TOP)/mk/config.make

#APP = 
PKG_DIR = $(PYSITEDIR)/chimera
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera

all: $(APP)

install: all
	$(RSYNC) $(APP).py $(PKG_DIR)
	$(RSYNC) $(APP) $(bindir)

app-install: all
	$(RSYNC) $(APP).py $(APP_PKG_DIR)
	$(RSYNC) $(APP) $(app_bindir)

clean:
	rm -f $(APP)

lint:
	$(PYLINT) $(APP).py

$(APP): ../moduleapp/app.sh.in
	sed -e 's,APP_NAME,$(APP_NAME),' \
		< ../moduleapp/app.sh.in > $(APP)
	chmod +x $(APP)
