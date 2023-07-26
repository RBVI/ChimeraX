# ifdef _WIN32
#  ifdef ATOMSTRUCT_EXPORT
#   define ATOMSTRUCT_IMEX __declspec(dllexport)
#  else
#   define ATOMSTRUCT_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define ATOMSTRUCT_IMEX __attribute__((__visibility__("default")))
#  else
#   define ATOMSTRUCT_IMEX
#  endif
# endif
