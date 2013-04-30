FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.0.2
PYOPENGL_DISTRIBUTION = PyOpenGL-$(VERSION).tar.gz
PYOPENGL_SOURCE = $(tmpdir)/PyOpenGL-$(VERSION)
ACCELERATE_DISTRIBUTION = PyOpenGL-accelerate-$(VERSION).tar.gz
ACCELERATE_SOURCE = $(tmpdir)/PyOpenGL-accelerate-$(VERSION)

SETUP_BUILD = build_clib build_ext --force
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
	cd $(PYOPENGL_SOURCE) && python setup.py $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && python setup.py $(SETUP_BUILD) install

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
