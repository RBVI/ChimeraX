PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Install Qt shim that handle PyQt5 and PySide2

install:
	$(RSYNC) Qt $(PYSITEDIR)
ifdef INSTALL_TO_VENV
ifndef VIRTUAL_ENV
	@echo "No virtual environment detected, skipping venv installation"
else
	$(RSYNC) Qt $(VIRTUAL_ENV)/lib/python3.9/site-packages/
endif # VIRTUAL_ENV
endif # INSTALL_TO_VENV


app-install:
	$(RSYNC) Qt $(APP_PYSITEDIR)
