TOP	= ../../..
include $(TOP)/mk/config.make
include ../Makefile.bundle

# mmtf-cpp-$(MMTF_CPP_VERSION).zip is the "Source code (zip)" archive from
# https://github.com/rcsb/mmtf-cpp/releases/tag/v$(MMTF_CPP_VERSION)
#
# msgpack-c-cpp-$(MSGPACK_VERSION).zip is the "Source code (zip)" archive from
# https://github.com/msgpack/msgpack-c/releases/tag/cpp-$(MSGPACK_VERSION)

CSRCS = $(wildcard _mmtf/*.cpp)
HEADERS = $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE)

MMTF_CPP_VERSION = 1.1.0
MMTF_CPP_SOURCE	= mmtf-cpp-$(MMTF_CPP_VERSION)
MMTF_CPP_DISTRIBUTION = $(MMTF_CPP_SOURCE).zip
MMTF_CPP_URL = https://github.com/rcsb/mmtf-cpp/archive/refs/tags/v$(MMTF_CPP_VERSION).zip
MSGPACK_VERSION = 4.1.2
MSGPACK_SOURCE	= msgpack-c-cpp-$(MSGPACK_VERSION)
MSGPACK_DISTRIBUTION = $(MSGPACK_SOURCE).zip
MSGPACK_URL = https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-$(MSGPACK_VERSION).zip
CLEAN = $(MMTF_CPP_SOURCE) $(MSGPACK_SOURCE) build dist *.egg-info src/__pycache__ src/*/__pycache__
DISTCLEAN = $(MMTF_CPP_DISTRIBUTION) $(MSGPACK_DISTRIBUTION)

build wheel install uv-wheel uv-install install-editable app-install:	$(CSRCS) $(HEADERS)

$(MMTF_CPP_SOURCE): $(MMTF_CPP_DISTRIBUTION)
	# only need the include files
	unzip -u $(MMTF_CPP_DISTRIBUTION) '$(MMTF_CPP_SOURCE)/include/**'

$(MMTF_CPP_DISTRIBUTION):
	curl --silent --show-error --fail --location -o $@ $(MMTF_CPP_URL)

$(MSGPACK_SOURCE): $(MSGPACK_DISTRIBUTION)
	# only need the include files
	unzip -u $(MSGPACK_DISTRIBUTION) '$(MSGPACK_SOURCE)/include/**'

$(MSGPACK_DISTRIBUTION):
	curl --silent --show-error --fail --location -o $@ $(MSGPACK_URL)
