TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/docs

HTML_SRCS = fitmap.html index.html molmap.html volume.html
ICONS = icons

all: 

install: pkg_dir
	$(RSYNC) $(HTML_SRCS) $(ICONS) $(PKG_DIR)

clean:
	rm -rf __pycache__
