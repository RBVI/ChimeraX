Some of our prerequisites e.g. libtheora, libxdrfile, are very old, and their source tarballs
include versions of config.sub and config.guess that do not account for modern arm64 linux
systems or mingw. For those builds, config.guess and config.sub can be copied from this
directory to wherever they are in those repos before they're built.
