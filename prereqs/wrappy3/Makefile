FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ifndef NO_SUBDIR
SUBDIRS	= lib
include $(TOP)/mk/subdir.make
endif

.SUFFIXES: .txt .html

.txt.html:
	docutils-html -tg --no-footnote-backlinks $*.txt $*.html

PROGNAME= wrappy3
ifdef WIN32
PROG	= $(PROGNAME).exe
else
PROG	= $(PROGNAME)
endif
DOCDIR	= $(OTFDOC)/bin/wrappy
SRCS	= wrappy.cpp common.cpp compute.cpp method.cpp module.cpp \
	type.cpp Lex.cpp ParseHeader.cpp CvtType.cpp \
	GetOpt.cpp Symbol.cpp Reg.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
ifeq ($(WIN32),msvc)
LIBS	= /link /LIBPATH:'$(shell cygpath -m '$(libdir)')' pcreposix.lib pcre.lib
else
LIBS	= -L$(libdir) -lpcreposix -lpcre -lm
ifeq ($(OS),Linux)
LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
endif
ifeq ($(OS),Darwin)
LDFLAGS += -Wl,-rpath,'@loader_path/../lib'
endif
endif
DOCS	= minidoc.txt
ifeq ($(WIN32),msvc)
INCS	+= -I$(shell cygpath -m '$(includedir)')
else
INCS	+= -I$(includedir)
endif

ifeq ($(OS),Linux)
## for compiling on Red Hat 7.1 with gcc 3.3.2
#DEFS	+= -DOTF_NO_LOCALE
endif

all: $(PROG)

$(PROG): $(OBJS)
	$(PROG_LINK)

install: $(PROG)
	cp -fp $(PROG) $(bindir)/

app-install: lib.install
	$(RSYNC) $(shlibdir)/lib$(PROGNAME).*$(SHLIB_EXT)* $(app_shlibdir)

docinstall:
	-mkdir $(DOCDIR)
	cp -fp $(DOCS) $(DOCDIR)

depend:
	mv Makefile Makefile.save
	sed -n '1,/DE\PEND/p' < Makefile.save > Makefile
	g++ -MM $(CFLAGS) ${SRCS} >> Makefile

clean:
	rm -rf $(OBJS) errs $(EXTRA_CLEAN)
	rm -f $(PROG) tags core errs

### DEPEND
