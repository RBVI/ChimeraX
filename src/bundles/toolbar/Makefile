include ../Makefile.bundle

LIBS = src/lib/ribbon src/lib/jquery-1.6.1.min.js
CLEAN_ALWAYS += src/lib/ribbon

wheel install app-install:	$(LIBS)

# From https://github.com/OkGoDoIt/Office-Ribbon-2010
RIBBON_SRC = Office-Ribbon-2010-master
src/lib/ribbon:	$(RIBBON_SRC).zip
	unzip $? && mv $(RIBBON_SRC)/ribbon src/lib \
		&& mv $(RIBBON_SRC)/icons src/lib \
		&& rm -rf $(RIBBON_SRC) && touch $@
