#ifndef VOLUMEARRAY_CONFIG_HEADER_INCLUDED
#define VOLUMEARRAY_CONFIG_HEADER_INCLUDED

# ifndef VOLUMEARRAY_DLL
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define VOLUMEARRAY_IMEX __attribute__((__visibility__("default")))
#  else
#   define VOLUMEARRAY_IMEX
#  endif
# elif defined(VOLUMEARRAY_EXPORT)
#  define VOLUMEARRAY_IMEX __declspec(dllexport)
# else
#  define VOLUMEARRAY_IMEX __declspec(dllimport)
# endif

#endif
