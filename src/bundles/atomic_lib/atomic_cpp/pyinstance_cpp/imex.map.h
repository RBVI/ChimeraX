# ifdef _WIN32
#  ifdef PYINSTANCE_MAP_EXPORT
#   define PYINSTANCE_MAP_IMEX __declspec(dllexport)
#  else
#   define PYINSTANCE_MAP_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define PYINSTANCE_MAP_IMEX __attribute__((__visibility__("default")))
#  else
#   define PYINSTANCE_MAP_IMEX
#  endif
# endif
