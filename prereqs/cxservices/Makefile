PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

CXSERVICES_VER = 1.1

# Binary wheels not available for Mac and Windows for PyTables 3.6.1.
# So use versions we built with Makefile.macos and Makefile.win
DISTRIBUTION = cxservices-$(CXSERVICES_VER)-py3-none-any.whl

app-install: $(DISTRIBUTION)
	$(APP_PYTHON_EXE) -m pip install $(DISTRIBUTION)

app-reinstall: $(DISTRIBUTION)
	$(APP_PYTHON_EXE) -m pip install $(DISTRIBUTION) --force-reinstall

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/$(DISTRIBUTION)

clean:
	rm -rf $(DISTRIBUTION)
