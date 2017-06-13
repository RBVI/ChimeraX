PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Got 2013/05/02 PyOpenGL source by running:
#
#     bzr export PyOpenGL-20130502.tar.bz2 lp:pyopengl
#

ifdef WIN32
VERSION = 3.1.1
else
VERSION = 3.1.0
endif
PYOPENGL_DISTRIBUTION = PyOpenGL-$(VERSION).tar.gz
PYOPENGL_SOURCE = $(tmpdir)/PyOpenGL-$(VERSION)
ACCELERATE_DISTRIBUTION = PyOpenGL-accelerate-$(VERSION).tar.gz
ACCELERATE_SOURCE = $(tmpdir)/PyOpenGL-accelerate-$(VERSION)

ifdef WIN32
include ../pips/Makefile.pip
WHEEL = PyOpenGL-$(VERSION)-cp36-cp36m-win_amd64.whl \
	PyOpenGL_accelerate-$(VERSION)-cp36-cp36m-win_amd64.whl
endif

ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

PATCHES = $(wildcard PyOpenGL-$(VERSION)-*.patch) \
	$(wildcard PyOpenGL-$(VERSION)_$(OS).patch) \
	$(wildcard PyOpenGL-$(VERSION)_$(OS)_$(OPENGL_PLATFORM).patch)

all:

install: $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
ifdef WIN32
	$(PIP_INSTALL) $(WHEEL)
else
	cd $(PYOPENGL_SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install
endif

app-install: install
ifdef WIN32
	$(APP_PIP_INSTALL) $(WHEEL)
else
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/PyOpenGL-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)
	#$(RSYNC) $(PYSITEDIR)/PyOpenGL_accelerate-$(VERSION)-py$(PYTHON_VERSION)-*.egg $(APP_PYSITEDIR)
	cd $(PYOPENGL_SOURCE) && $(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && $(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install
endif

$(PYOPENGL_SOURCE):
ifndef WIN32
	tar zxf $(PYOPENGL_DISTRIBUTION) -C $(tmpdir)

ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(PYOPENGL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif
endif

$(ACCELERATE_SOURCE):
ifndef WIN32
	tar zxf $(ACCELERATE_DISTRIBUTION) -C $(tmpdir)
endif

clean:
	rm -rf $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
