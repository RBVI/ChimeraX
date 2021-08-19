PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Install Qt shim that handle PyQt5 and PySide2

install:
	$(RSYNC) Qt $(PYSITEDIR)

app-install:
	$(RSYNC) Qt $(APP_PYSITEDIR)
