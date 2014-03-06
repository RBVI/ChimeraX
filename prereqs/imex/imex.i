# ifdef _WIN32
#  ifdef LIBNAME_EXPORT
#   define LIBNAME_IMEX __declspec(dllexport)
#  else
#   define LIBNAME_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define LIBNAME_IMEX __attribute__((__visibility__("default")))
#  else
#   define LIBNAME_IMEX
#  endif
# endif
