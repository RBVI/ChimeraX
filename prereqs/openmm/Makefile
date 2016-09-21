PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 7.0.1
DISTRIBUTION = OpenMM-$(VERSION)-Mac.tar.bz2
SOURCE = $(tmpdir)/OpenMM-$(VERSION)-Mac
OPENMM_LIB_INSTALL = $(libdir)/openmm

ifeq ($(OS),Darwin)

install: $(SOURCE)
	cd $(SOURCE) ; printf "$(OPENMM_LIB_INSTALL)\n$(PYTHON_EXE)\n" | ./install.sh

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/lib/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

else

install:

app-install:

endif

clean:
	rm -rf $(SOURCE)
