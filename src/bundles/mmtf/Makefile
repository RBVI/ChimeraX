TOP	= ../../..
include $(TOP)/mk/config.make
include ../Makefile.bundle

# mmcif-cpp-master.zip is from https://github.com/rcsb/mmtf-cpp
# msgpack-c-cpp_master.zip is from https://github.com/msgpack/msgpack-c/tree/cpp_master

CSRCS = $(wildcard _mmtf/*.cpp)
HEADERS = $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE)

MMTF_CPP_VERSION = 1.1.0
MMTF_CPP_SOURCE	= mmtf-cpp-$(MMTF_CPP_VERSION)
MMTF_CPP_DISTRIBUTION = $(MMTF_CPP_SOURCE).zip
MSGPACK_VERSION = 4.1.2
MSGPACK_SOURCE	= msgpack-c-cpp-$(MSGPACK_VERSION)
MSGPACK_DISTRIBUTION = $(MSGPACK_SOURCE).zip

CLEAN_ALWAYS = $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE)

build wheel install install-editable app-install:	$(CSRCS) $(HEADERS)

$(MMTF_CPP_SOURCE): $(MMTF_CPP_DISTRIBUTION)
	# only need the include files
	unzip -u $(MMTF_CPP_DISTRIBUTION) '$(MMTF_CPP_SOURCE)/include/**'

$(MMTF_CPP_DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/mmtf/$(MMTF_CPP_DISTRIBUTION)

$(MSGPACK_SOURCE): $(MSGPACK_DISTRIBUTION)
	# only need the include files
	unzip -u $(MSGPACK_DISTRIBUTION) '$(MSGPACK_SOURCE)/include/**'

$(MSGPACK_DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/mmtf/$(MSGPACK_DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(MMTF_CPP_DISTRIBUTION) $(MSGPACK_DISTRIBUTION) $(PREREQS_UPLOAD)/mmtf
