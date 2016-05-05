#include <stdio.h>
#include <Python.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#include <locale.h>

/*
 * Mimic:
 *
 *	python -I -X faulthandler -m ChimeraX_main.py app-args
 */

#define EXTRA 5	/* number of extra arguments */

static int
app_main(int argc, wchar_t** wargv)
{
	wchar_t** args = (wchar_t**) malloc((argc + EXTRA) * sizeof (wchar_t*));
	if (args == NULL) {
		fprintf(stderr, "out of memory\n");
		return 123;
	}
	args[0] = wargv[0];
	args[1] = L"-I";
	args[2] = L"-X";
	args[3] = L"faulthandler";
	args[4] = L"-m";
	args[EXTRA] = L"ChimeraX_main";
	for (int i = 1; i < argc; ++i)
		args[i + EXTRA] = wargv[i];

	int result = Py_Main(argc + EXTRA, args);
	return result;
}

#ifdef _WIN32
int WINAPI
wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine,
	int nCmdShow)
{
	return app_main(__argc, __wargv);
}

#else
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
#endif
