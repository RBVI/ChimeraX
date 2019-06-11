#ifndef SEGMENT_CONFIG_HEADER_INCLUDED
#define SEGMENT_CONFIG_HEADER_INCLUDED

# ifndef SEGMENT_DLL
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define SEGMENT_IMEX __attribute__((__visibility__("default")))
#  else
#   define SEGMENT_IMEX
#  endif
# elif defined(SEGMENT_EXPORT)
#  define SEGMENT_IMEX __declspec(dllexport)
# else
#  define SEGMENT_IMEX __declspec(dllimport)
# endif

#endif
