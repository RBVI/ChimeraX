FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Got 2013/05/02 PyOpenGL source by running:
#
#     bzr export PyOpenGL-20130502.tar.bz2 lp:pyopengl
#

VERSION = 20140109
PYOPENGL_DISTRIBUTION = PyOpenGL-$(VERSION).tar.bz2
PYOPENGL_SOURCE = $(tmpdir)/PyOpenGL-$(VERSION)
#ACCELERATE_DISTRIBUTION = PyOpenGL-accelerate-$(VERSION).tar.gz
#ACCELERATE_SOURCE = $(tmpdir)/PyOpenGL-accelerate-$(VERSION)
ACCELERATE_SOURCE = $(tmpdir)/PyOpenGL-$(VERSION)/OpenGL_accelerate

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
	cd $(PYOPENGL_SOURCE) && python3 setup.py $(SETUP_BUILD) install
	cd $(ACCELERATE_SOURCE) && python3 setup.py $(SETUP_BUILD) install

$(PYOPENGL_SOURCE):
	tar jxf $(PYOPENGL_DISTRIBUTION) -C $(tmpdir)

ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(PYOPENGL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

#$(ACCELERATE_SOURCE):
#	tar zxf $(ACCELERATE_DISTRIBUTION) -C $(tmpdir)
$(ACCELERATE_SOURCE): $(PYOPENGL_SOURCE)

clean:
	rm -rf $(PYOPENGL_SOURCE) $(ACCELERATE_SOURCE)
