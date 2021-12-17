PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

CXSERVICES_VER = 1.1
CXSERVICES_DEPLOYMENT_VER = test

# Binary wheels not available for Mac and Windows for PyTables 3.6.1.
# So use versions we built with Makefile.macos and Makefile.win
DISTRIBUTION = cxservices-$(CXSERVICES_VER)-$(CXSERVICES_DEPLOYMENT_VER)-py3-none-any.whl
REAL_DISTRIBUTION = cxservices-$(CXSERVICES_VER)-py3-none-any.whl

app-install: $(DISTRIBUTION)
	cp $(DISTRIBUTION) $(REAL_DISTRIBUTION)
	$(APP_PYTHON_EXE) -m pip install $(REAL_DISTRIBUTION)

app-reinstall: $(DISTRIBUTION)
	cp $(DISTRIBUTION) $(REAL_DISTRIBUTION)
	$(APP_PYTHON_EXE) -m pip install $(REAL_DISTRIBUTION) --force-reinstall

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/$(DISTRIBUTION)

clean:
	rm -rf $(DISTRIBUTION) $(REAL_DISTRIBUTION)
