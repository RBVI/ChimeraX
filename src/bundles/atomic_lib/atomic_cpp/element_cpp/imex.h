# ifdef _WIN32
#  ifdef ELEMENT_EXPORT
#   define ELEMENT_IMEX __declspec(dllexport)
#  else
#   define ELEMENT_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define ELEMENT_IMEX __attribute__((__visibility__("default")))
#  else
#   define ELEMENT_IMEX
#  endif
# endif
