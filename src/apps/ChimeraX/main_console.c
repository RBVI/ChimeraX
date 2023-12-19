#include <stdio.h>
#include <Python.h>
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#include <locale.h>

extern int app_main(int argc, wchar_t** wargv);

int
main(int argc, char** argv)
{
	size_t len;
	wchar_t** wargv = (wchar_t**) malloc(argc * sizeof (wchar_t*));
	if (wargv == NULL) {
		fprintf(stderr, "out of memory\n");
		return 120;
	}

	setlocale(LC_ALL, "");
	for (int i = 0; i < argc; ++i) {
		wargv[i] = Py_DecodeLocale(argv[i], &len);
		if (wargv[i] == NULL) {
			fprintf(stderr, "out of memory\n");
			return 121;
		}
	}
	return app_main(argc, wargv);
}
