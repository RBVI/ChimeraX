PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Got 2013/05/02 PyOpenGL source by running:
#
#     bzr export PyOpenGL-20130502.tar.bz2 lp:pyopengl
#

ifdef WIN32
# Downloaded wheels from http://www.lfd.uci.edu/~gohlke/pythonlibs/
# Because the PyOpenGL source wants to use numpy's float128, but the
# numpy in numpy+mkl compiled with Microsoft Visual C++ doesn't support it.
USE_WHEEL = 1
endif

ifdef USE_WHEEL
VERSION = 3.1.3b2
ACCEL_VERSION = 3.1.3b2
else
VERSION = 3.1.3b2
ACCEL_VERSION = 3.1.3b1
endif
PYOPENGL_DISTRIBUTION = PyOpenGL-$(VERSION).tar.gz
PYOPENGL_SOURCE = $(tmpdir)/PyOpenGL-$(VERSION)
ACCELERATE_DISTRIBUTION = PyOpenGL-accelerate-$(ACCEL_VERSION).tar.gz
ACCELERATE_SOURCE = $(tmpdir)/PyOpenGL-accelerate-$(ACCEL_VERSION)

ALREADY_INSTALLED = grep -E "(already satisfied|not upgraded)"

ifdef USE_WHEEL
include ../pips/Makefile.pip
WHEEL = PyOpenGL-$(VERSION)-cp$(PYVER_NODOT)-cp$(PYVER_NODOT)m-win_amd64.whl \
	PyOpenGL_accelerate-$(ACCEL_VERSION)-cp$(PYVER_NODOT)-cp$(PYVER_NODOT)m-win_amd64.whl
endif

ifdef DEBUG
ifndef USE_WHEEL
SETUP_BUILD += --debug
endif
endif

PYOPENGL_PATCHES = $(wildcard PyOpenGL-$(VERSION)-*.patch) \
	$(wildcard PyOpenGL-$(VERSION)_$(OS).patch) \
	$(wildcard PyOpenGL-$(VERSION)_$(OS)_$(OPENGL_PLATFORM).patch)

ACCELERATE_PATCHES = $(wildcard PyOpenGL-accelerate-$(ACCEL_VERSION)-*.patch)

all:

ifdef USE_WHEEL

install:
	$(PIP_INSTALL) $(WHEEL) | $(ALREADY_INSTALLED) || \
		(for p in $(PYOPENGL_PATCHES); do (cd $(PYSITEDIR) && patch -f -p0) < $$p; done)

app-install:
	$(APP_PIP_INSTALL) $(WHEEL) | $(ALREADY_INSTALLED) || \
		(for p in $(PYOPENGL_PATCHES); do (cd $(APP_PYSITEDIR) && patch -f -p0) < $$p; done)

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
ifneq (,$(PYOPENGL_PATCHES))
	for p in $(PYOPENGL_PATCHES); do \
		(cd $(PYOPENGL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

$(ACCELERATE_SOURCE):
	tar zxf $(ACCELERATE_DISTRIBUTION) -C $(tmpdir)
ifneq (,$(ACCELERATE_PATCHES))
	for p in $(ACCELERATE_PATCHES); do \
		(cd $(ACCELERATE_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
