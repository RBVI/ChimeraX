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
 *	python -I [-X faulthandler] -m ChimeraX_main.py app-args
 */

static wchar_t* extra[] = {
	L"-I",
#ifndef _WIN32
	L"-X",
	L"faulthandler",
#endif
	L"-m",
	L"ChimeraX_main",
};
static const int ec = sizeof(extra) / sizeof (extra[0]);

static int
app_main(int argc, wchar_t** wargv)
{
	wchar_t** args = (wchar_t**) malloc((argc + ec + 1) * sizeof (wchar_t*));
	if (args == NULL) {
		fprintf(stderr, "out of memory\n");
		return 123;
	}
	args[0] = wargv[0];
	for (int i = 0; i < ec; ++i)
		args[i + 1] = extra[i];
	for (int i = 1; i < argc; ++i)
		args[i + ec] = wargv[i];
	args[argc + ec] = NULL;

	int result = Py_Main(argc + ec, args);
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
