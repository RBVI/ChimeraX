PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 4.1.1
DISTRIBUTION = Pillow-$(VERSION).tar.gz
SOURCE = $(tmpdir)/Pillow-$(VERSION)
PATCHES	= pillow_ome_tiff.patch pillow_i16s_mode.patch

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build install

app-install: install
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py build install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
