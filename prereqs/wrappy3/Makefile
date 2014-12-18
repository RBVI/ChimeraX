TOP = ../..
include $(TOP)/mk/config.make

ifndef NO_SUBDIR
SUBDIRS	= lib
include $(TOP)/mk/subdir.make
endif

.SUFFIXES: .txt .html

.txt.html:
	docutils-html -tg --no-footnote-backlinks $*.txt $*.html

ifdef WIN32
PROG	= wrappy3.exe
else
PROG	= wrappy3
endif
DOCDIR	= $(OTFDOC)/bin/wrappy
SRCS	= wrappy.cpp common.cpp compute.cpp method.cpp module.cpp \
	type.cpp Lex.cpp ParseHeader.cpp CvtType.cpp \
	GetOpt.cpp Symbol.cpp Reg.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
ifeq ($(WIN32),msvc)
LIBS	= /LINK LIBPATH:$(libdir) -lpcreposix -lpcre
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
INCS	+= -I$(includedir)

ifeq ($(OS),Linux)
## for compiling on Red Hat 7.1 with gcc 3.3.2
#DEFS	+= -DOTF_NO_LOCALE
endif

all: $(PROG)

$(PROG): $(OBJS)
	$(PROG_LINK)

install: $(PROG)
	cp -fp $(PROG) $(bindir)/

app-install:;

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
