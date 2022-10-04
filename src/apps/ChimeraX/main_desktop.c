#include <stdio.h>
#include <Python.h>
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#include <locale.h>

// To create a GUI subsystem executable on Windows, we have to
// define winMain() or wWinMain() instead of main()

extern int app_main(int argc, wchar_t** wargv);

int WINAPI
wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine,
	int nCmdShow)
{
	return app_main(__argc, __wargv);
}
