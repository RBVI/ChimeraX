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
 * CHIMERAX_STARTUP_PREFETCH.  On later starts, read them ahead from
 * background threads before Python initializes, so the reads overlap with
 * startup instead of blocking it.  Windows has its own application
 * prefetcher.
 *
 * On Linux a line may end with a tab and the byte ranges startup touched,
 * "offset+length,...", and the files are read with several threads.  Asking
 * the kernel to read whole files ahead (posix_fadvise WILLNEED) from one
 * thread fell well behind startup: a cold start took 0.51 s with it, 0.39 s
 * reading the touched ranges with 4 threads, and 0.36 s warm.
 */
static char prefetch_list[PATH_MAX];

# ifdef __APPLE__
static void
prefetch_file(const char* path)
{
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return;
	struct stat st;
	if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
		struct radvisory ra;
		ra.ra_offset = 0;
		ra.ra_count = st.st_size > INT_MAX ? INT_MAX : (int) st.st_size;
		(void) fcntl(fd, F_RDADVISE, &ra);
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
# else
#  define PREFETCH_THREADS 4
#  define PREFETCH_BUFFER (1 << 20)

static char** prefetch_entries;
static size_t prefetch_count;
static size_t prefetch_next;

static void
read_range(int fd, char* buffer, off_t offset, off_t length)
{
	while (length > 0) {
		ssize_t n = pread(fd, buffer,
			length < PREFETCH_BUFFER ? (size_t) length : PREFETCH_BUFFER, offset);
		if (n <= 0)
			break;
		offset += n;
		length -= n;
	}
}

static void
prefetch_entry(char* entry, char* buffer)
{
	char* ranges = strchr(entry, '\t');
	if (ranges != NULL)
		*ranges++ = '\0';
	int fd = open(entry, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return;
	if (ranges == NULL) {
		struct stat st;
		if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode))
			read_range(fd, buffer, 0, st.st_size);
	} else {
		char* p = ranges;
		while (*p != '\0') {
			char* end;
			unsigned long long offset = strtoull(p, &end, 10);
			if (*end != '+')
				break;
			unsigned long long length = strtoull(end + 1, &end, 10);
			read_range(fd, buffer, (off_t) offset, (off_t) length);
			if (*end != ',')
				break;
			p = end + 1;
		}
	}
	close(fd);
}

static void*
prefetch_worker(void* unused)
{
	char* buffer = malloc(PREFETCH_BUFFER);
	if (buffer == NULL)
		return NULL;
	for (;;) {
		// Take entries in list order, which is roughly the order startup needs them.
		size_t i = __atomic_fetch_add(&prefetch_next, 1, __ATOMIC_RELAXED);
		if (i >= prefetch_count)
			break;
		prefetch_entry(prefetch_entries[i], buffer);
	}
	free(buffer);
	return NULL;
}

static void*
prefetch_files(void* unused)
{
	FILE* f = fopen(prefetch_list, "r");
	if (f == NULL)
		return NULL;
	struct stat st;
	char* text = NULL;
	if (fstat(fileno(f), &st) == 0 && st.st_size > 0
			&& (text = malloc((size_t) st.st_size + 1)) != NULL) {
		size_t n = fread(text, 1, (size_t) st.st_size, f);
		text[n] = '\0';
	}
	fclose(f);
	if (text == NULL)
		return NULL;

	size_t lines = 1;
	for (char* c = text; *c; ++c)
		if (*c == '\n')
			++lines;
	prefetch_entries = malloc(lines * sizeof (char*));
	if (prefetch_entries == NULL)
		return NULL;
	char* save = NULL;
	for (char* line = strtok_r(text, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save))
		if (line[0] == '/')
			prefetch_entries[prefetch_count++] = line;

	pthread_attr_t attr;
	if (pthread_attr_init(&attr) == 0) {
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		for (int i = 1; i < PREFETCH_THREADS; ++i) {
			pthread_t thread;
			(void) pthread_create(&thread, &attr, prefetch_worker, NULL);
		}
		pthread_attr_destroy(&attr);
	}
	return prefetch_worker(NULL);
}
# endif

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
