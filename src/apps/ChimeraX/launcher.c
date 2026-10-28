#include <stdio.h>
#include <Python.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#include <locale.h>

#if defined(__APPLE__) || defined(__linux__)
# define STARTUP_PREFETCH 1
# include <fcntl.h>
# include <limits.h>
# include <pthread.h>
# include <stdint.h>
# include <stdlib.h>
# include <string.h>
# include <sys/stat.h>
# include <unistd.h>
# ifdef __APPLE__
#  include <mach-o/dyld.h>
# endif
#endif

#ifdef STARTUP_PREFETCH
/*
 * A cold start reads the shared libraries and Python modules it needs one
 * page fault or one small file at a time.  ChimeraX records the files a GUI
 * startup used (chimerax.core.startup_prefetch) in the file named by
 * CHIMERAX_STARTUP_PREFETCH.  On later starts, ask the kernel to read them
 * ahead, from a background thread, before Python initializes, so the reads
 * overlap with startup instead of blocking it.  Windows has its own
 * application prefetcher.
 */
static char prefetch_list[PATH_MAX];

static void
prefetch_file(const char* path)
{
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return;
	struct stat st;
	if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
# ifdef __APPLE__
		struct radvisory ra;
		ra.ra_offset = 0;
		ra.ra_count = st.st_size > INT_MAX ? INT_MAX : (int) st.st_size;
		(void) fcntl(fd, F_RDADVISE, &ra);
# else
		(void) posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
# endif
	}
	close(fd);
}

static void*
prefetch_files(void* unused)
{
	FILE* f = fopen(prefetch_list, "r");
	if (f == NULL)
		return NULL;
	char line[PATH_MAX + 2];
	while (fgets(line, sizeof line, f) != NULL) {
		size_t len = strcspn(line, "\n");
		line[len] = '\0';
		if (line[0] == '/')
			prefetch_file(line);
	}
	fclose(f);
	return NULL;
}

static void
start_prefetch(void)
{
	if (getenv("CHIMERAX_NO_STARTUP_PREFETCH") != NULL)
		return;
	char exe[PATH_MAX], real_exe[PATH_MAX];
# ifdef __APPLE__
	uint32_t size = sizeof exe;
	if (_NSGetExecutablePath(exe, &size) != 0)
		return;
# else
	strcpy(exe, "/proc/self/exe");
# endif
	if (realpath(exe, real_exe) == NULL)
		return;

	const char* home = getenv("HOME");
	char cache_dir[PATH_MAX];
	int n;
# ifdef __APPLE__
	if (home == NULL || home[0] == '\0')
		return;
	n = snprintf(cache_dir, sizeof cache_dir, "%s/Library/Caches/ChimeraX", home);
# else
	const char* xdg = getenv("XDG_CACHE_HOME");
	if (xdg != NULL && xdg[0] == '/')
		n = snprintf(cache_dir, sizeof cache_dir, "%s/ChimeraX", xdg);
	else if (home != NULL && home[0] != '\0')
		n = snprintf(cache_dir, sizeof cache_dir, "%s/.cache/ChimeraX", home);
	else
		return;
# endif
	if (n < 0 || n >= (int) sizeof cache_dir)
		return;

	// One list per executable: FNV-1a hash of its real path.
	uint64_t hash = 0xcbf29ce484222325ULL;
	for (const unsigned char* c = (const unsigned char*) real_exe; *c; ++c) {
		hash ^= *c;
		hash *= 0x100000001b3ULL;
	}
	n = snprintf(prefetch_list, sizeof prefetch_list,
		"%s/startup-prefetch-%016llx", cache_dir, (unsigned long long) hash);
	if (n < 0 || n >= (int) sizeof prefetch_list)
		return;
	// Python records this startup's files here.
	setenv("CHIMERAX_STARTUP_PREFETCH", prefetch_list, 1);

	pthread_attr_t attr;
	pthread_t thread;
	if (pthread_attr_init(&attr) != 0)
		return;
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	(void) pthread_create(&thread, &attr, prefetch_files, NULL);
	pthread_attr_destroy(&attr);
}
#endif

/*
 * Mimic:
 *
 *	python -I [-X faulthandler] -m ChimeraX_main.py app-args
 */

static wchar_t* extra[] = {
	L"-I",
#if 0
	// TODO: Probably the right thing to do, but needs discussing
	L"-X",
	L"utf8",
#endif
#ifndef _WIN32
	// Works when started from cygwin shell, but not when
	// double-clicked upon, nor from command prompt shell
	L"-X",
	L"faulthandler",
#endif
	L"-m",
	L"chimerax.core",
};
static const int ec = sizeof(extra) / sizeof (extra[0]);

static wchar_t* debug_extra[] = {
	L"-X",
	L"dev",
/* Too much output from import timing for debugging.
	L"-X",
	L"importtime",
 */
};
static const int debug_ec = sizeof(debug_extra) / sizeof (debug_extra[0]);

/*
 * Make Nvidia Optimus GPU switching choose high performance graphics.
 * https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
 */
#ifdef _WIN32
_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
#endif

int
app_main(int argc, wchar_t** wargv)
{
#ifdef STARTUP_PREFETCH
	start_prefetch();
#endif
	int debug = 0;
	for (int i = 1; i < argc; ++i) {
		if (wcscmp(wargv[i], L"--debug") == 0) {
			debug = 1;
			break;
		}
	}
	int new_argc = argc + ec + debug * debug_ec;
	wchar_t** new_argv = (wchar_t**) malloc((new_argc + 1) * sizeof (wchar_t*));
	if (new_argv == NULL) {
		fprintf(stderr, "out of memory\n");
		return 123;
	}
	int ac = 0;
	new_argv[ac++] = wargv[0];
	if (debug) {
		for (int i = 0; i < debug_ec; ++i)
			new_argv[ac++] = debug_extra[i];
	}
	for (int i = 0; i < ec; ++i)
		new_argv[ac++] = extra[i];
	for (int i = 1; i < argc; ++i)
		new_argv[ac++] = wargv[i];
	assert(ac == new_argc);
	new_argv[ac] = NULL;

	int result = Py_Main(new_argc, new_argv);
	return result;
}
