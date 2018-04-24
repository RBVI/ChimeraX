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
	$(wildcard PyOpenGL-$(VERSION)_$(OS)_$(OPENGL_PLATFORM).patch) \
	PyOpenGL-wgl.patch PyOpenGL-glbuffersubdata.patch

all:

ifdef WIN32

install:
	$(PIP_INSTALL) $(WHEEL) | grep "already satisfied" || \
		((cd $(PYSITEDIR) && patch -f -p0) < PyOpenGL-wgl.patch && \
		 (cd $(PYSITEDIR) && patch -f -p0) < PyOpenGL-glbuffersubdata.patch)

app-install:
	$(APP_PIP_INSTALL) $(WHEEL) | grep "already satisfied" || \
		((cd $(APP_PYSITEDIR) && patch -f -p0) < PyOpenGL-wgl.patch && \
		 (cd $(APP_PYSITEDIR) && patch -f -p0) < PyOpenGL-glbuffersubdata.patch)

else

install: $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
	cd $(PYOPENGL_SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install: $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
	cd $(PYOPENGL_SOURCE) && $(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && $(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install

endif

$(PYOPENGL_SOURCE):
	tar zxf $(PYOPENGL_DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(PYOPENGL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

$(ACCELERATE_SOURCE):
	tar zxf $(ACCELERATE_DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
