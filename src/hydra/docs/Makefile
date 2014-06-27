TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/docs

HTML_SRCS = fitmap.html index.html molmap.html volume.html
ICONS = icons

all: 

install: $(PKG_DIR)
	$(RSYNC) $(HTML_SRCS) $(ICONS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
