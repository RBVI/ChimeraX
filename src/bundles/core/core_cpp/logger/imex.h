# ifdef _WIN32
#  ifdef LOGGER_EXPORT
#   define LOGGER_IMEX __declspec(dllexport)
#  else
#   define LOGGER_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define LOGGER_IMEX __attribute__((__visibility__("default")))
#  else
#   define LOGGER_IMEX
#  endif
# endif
