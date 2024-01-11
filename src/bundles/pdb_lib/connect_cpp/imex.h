# ifdef _WIN32
#  ifdef PDB_CONNECT_EXPORT
#   define PDB_CONNECT_IMEX __declspec(dllexport)
#  else
#   define PDB_CONNECT_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define PDB_CONNECT_IMEX __attribute__((__visibility__("default")))
#  else
#   define PDB_CONNECT_IMEX
#  endif
# endif
