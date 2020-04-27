# ifdef _WIN32
#  ifdef ARRAYS_EXPORT
#   define ARRAYS_IMEX __declspec(dllexport)
#  else
#   define ARRAYS_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define ARRAYS_IMEX __attribute__((__visibility__("default")))
#  else
#   define ARRAYS_IMEX
#  endif
# endif
