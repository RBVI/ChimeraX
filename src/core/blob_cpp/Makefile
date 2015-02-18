TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = blob
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
INCS += $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -I$(includedir)
ifneq ($(OS),Windows)
CXXFLAGS += -Wno-invalid-offsetof -Wno-deprecated-writable-strings
endif
LIBS += -L$(libdir) -latomstruct $(PYTHON_LIB)

AtomBlob.o: AtomBlob.cpp AtomBlob.h $(includedir)/atomstruct/Atom.h set_blob.h filter_blob.h
BondBlob.o: BondBlob.cpp BondBlob.h $(includedir)/atomstruct/Bond.h set_blob.h filter_blob.h
ResBlob.o: ResBlob.cpp ResBlob.h $(includedir)/atomstruct/Residue.h set_blob.h filter_blob.h
StructBlob.o: StructBlob.cpp StructBlob.h $(includedir)/atomstruct/AtomicStructure.h set_blob.h filter_blob.h

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(SHLIB) $(OBJS) imex.h
