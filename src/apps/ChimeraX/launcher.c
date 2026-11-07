#include <stdio.h>
#include <Python.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#include <locale.h>
#ifdef __APPLE__
# include "startup_qt.h"
#endif

#if defined(_WIN32)
# define STARTUP_PREFETCH 1
# include <stdint.h>
# include <stdlib.h>
# include <string.h>
#elif defined(__APPLE__) || defined(__linux__)
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
 * startup instead of blocking it.
 *
 * On Linux a line may end with a tab and the byte ranges startup touched,
 * "offset+length,...", and the files are read with several threads.  Asking
 * the kernel to read whole files ahead (posix_fadvise WILLNEED) from one
 * thread fell well behind startup: a cold start took 0.51 s with it, 0.39 s
 * reading the touched ranges with 4 threads, and 0.36 s warm.
 *
 * Windows' own prefetcher (SysMain) did not make this unnecessary: with
 * only ChimeraX's files cold, reading the startup ranges from 4 threads
 * took a Windows 11 VM's cold start from 4.5 s to 3.7 s.  The list lives
 * in %LOCALAPPDATA%\UCSF\ChimeraX\Cache, and its lines are UTF-8.
 */
# ifdef _WIN32
#  define PREFETCH_THREADS 4
#  define PREFETCH_BUFFER (1 << 20)
#  define PREFETCH_PATH 32768

static wchar_t prefetch_list[PREFETCH_PATH];
static char** prefetch_entries;
static size_t prefetch_count;
static volatile LONG64 prefetch_next;

// PrefetchVirtualMemory (Windows 8): looked up at run time.
typedef struct {
	PVOID VirtualAddress;
	SIZE_T NumberOfBytes;
} prefetch_range;
typedef BOOL (WINAPI *prefetch_virtual_memory_fn)(HANDLE, ULONG_PTR, prefetch_range*, ULONG);
static prefetch_virtual_memory_fn prefetch_virtual_memory;

/*
 * A DLL is loaded through its executable image section, which Windows
 * caches apart from the file's data, so reading a DLL as data helps its
 * later load only partly: loading Qt6Core, Qt6Gui and Qt6Widgets with only
 * ChimeraX's files cold took 0.58 s, 0.31 s after reading them as data, and
 * 0.08 s after mapping them as images (SEC_IMAGE) and prefetching the view.
 * Mapping a view does not involve the loader, so nothing runs and no module
 * is registered; the cached image section is reused by the real load.
 * With image prefetching the VM's cold start went from 3.64 s without
 * prefetching to 2.90 s; reading DLLs as data had given 3.34 s.
 */
static int
prefetch_image(const wchar_t* path)
{
	HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_EXECUTE,
		FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file == INVALID_HANDLE_VALUE)
		return 0;
	HANDLE section = CreateFileMappingW(file, NULL, PAGE_READONLY | SEC_IMAGE, 0, 0, NULL);
	CloseHandle(file);
	if (section == NULL)
		return 0;
	unsigned char* view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(section);
	if (view == NULL)
		return 0;
	IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*) view;
	IMAGE_NT_HEADERS* nt = (IMAGE_NT_HEADERS*) (view + dos->e_lfanew);
	SIZE_T size = nt->OptionalHeader.SizeOfImage;
	if (prefetch_virtual_memory != NULL) {
		prefetch_range range = { view, size };
		(void) prefetch_virtual_memory(GetCurrentProcess(), 1, &range, 0);
	} else {
		// Fault the pages in one at a time.
		volatile unsigned char sink = 0;
		for (SIZE_T offset = 0; offset < size; offset += 4096)
			sink ^= view[offset];
		(void) sink;
	}
	UnmapViewOfFile(view);
	return 1;
}

static void
read_range(HANDLE file, char* buffer, unsigned long long offset, unsigned long long length)
{
	while (length > 0) {
		OVERLAPPED at = { 0 };
		at.Offset = (DWORD) offset;
		at.OffsetHigh = (DWORD) (offset >> 32);
		DWORD want = length < PREFETCH_BUFFER ? (DWORD) length : PREFETCH_BUFFER, got = 0;
		if (!ReadFile(file, buffer, want, &got, &at) || got == 0)
			break;
		offset += got;
		length -= got;
	}
}

static void
prefetch_entry(char* entry, char* buffer)
{
	char* ranges = strchr(entry, '\t');
	if (ranges != NULL)
		*ranges++ = '\0';
	wchar_t path[PREFETCH_PATH];
	if (MultiByteToWideChar(CP_UTF8, 0, entry, -1, path, PREFETCH_PATH) == 0)
		return;
	size_t length = wcslen(path);
	if (length > 4 && (_wcsicmp(path + length - 4, L".dll") == 0
			|| _wcsicmp(path + length - 4, L".pyd") == 0)
			&& prefetch_image(path))
		return;
	HANDLE file = CreateFileW(path, GENERIC_READ,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
		OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file == INVALID_HANDLE_VALUE)
		return;
	if (ranges == NULL) {
		LARGE_INTEGER size;
		if (GetFileSizeEx(file, &size))
			read_range(file, buffer, 0, (unsigned long long) size.QuadPart);
	} else {
		char* p = ranges;
		while (*p != '\0') {
			char* end;
			unsigned long long offset = _strtoui64(p, &end, 10);
			if (*end != '+')
				break;
			unsigned long long length = _strtoui64(end + 1, &end, 10);
			read_range(file, buffer, offset, length);
			if (*end != ',')
				break;
			p = end + 1;
		}
	}
	CloseHandle(file);
}

static DWORD WINAPI
prefetch_worker(LPVOID unused)
{
	// On a warm start the files are already cached, and reading them only
	// takes CPU from startup: 4 normal priority threads made a warm start
	// 67 ms slower in a 4-core VM.  Cold, the threads mostly wait on I/O,
	// which keeps its normal priority.
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	char* buffer = malloc(PREFETCH_BUFFER);
	if (buffer == NULL)
		return 0;
	for (;;) {
		// Take entries in list order, which is roughly the order startup needs them.
		LONG64 i = InterlockedIncrement64(&prefetch_next) - 1;
		if ((size_t) i >= prefetch_count)
			break;
		prefetch_entry(prefetch_entries[i], buffer);
	}
	free(buffer);
	return 0;
}

static DWORD WINAPI
prefetch_files(LPVOID unused)
{
	HANDLE f = CreateFileW(prefetch_list, GENERIC_READ,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
		OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (f == INVALID_HANDLE_VALUE)
		return 0;
	LARGE_INTEGER size;
	char* text = NULL;
	if (GetFileSizeEx(f, &size) && size.QuadPart > 0 && size.QuadPart < (1 << 26)
			&& (text = malloc((size_t) size.QuadPart + 1)) != NULL) {
		DWORD got = 0;
		if (!ReadFile(f, text, (DWORD) size.QuadPart, &got, NULL))
			got = 0;
		text[got] = '\0';
	}
	CloseHandle(f);
	if (text == NULL)
		return 0;

	size_t lines = 1;
	for (char* c = text; *c; ++c)
		if (*c == '\n')
			++lines;
	prefetch_entries = malloc(lines * sizeof (char*));
	if (prefetch_entries == NULL)
		return 0;
	char* save = NULL;
	for (char* line = strtok_s(text, "\n", &save); line != NULL; line = strtok_s(NULL, "\n", &save)) {
		size_t n = strlen(line);
		if (n > 0 && line[n - 1] == '\r')
			line[n - 1] = '\0';
		// Absolute paths only: "X:\..." or "\\server\...".
		if ((line[0] != '\0' && line[1] == ':') || (line[0] == '\\' && line[1] == '\\'))
			prefetch_entries[prefetch_count++] = line;
	}
	for (int i = 1; i < PREFETCH_THREADS; ++i) {
		HANDLE thread = CreateThread(NULL, 0, prefetch_worker, NULL, 0, NULL);
		if (thread != NULL)
			CloseHandle(thread);
	}
	return prefetch_worker(NULL);
}

static void
start_prefetch(void)
{
	if (GetEnvironmentVariableW(L"CHIMERAX_NO_STARTUP_PREFETCH", NULL, 0) > 0)
		return;
	static wchar_t exe[PREFETCH_PATH], local[PREFETCH_PATH];
	static char exe8[3 * PREFETCH_PATH];
	DWORD n = GetModuleFileNameW(NULL, exe, PREFETCH_PATH);
	if (n == 0 || n >= PREFETCH_PATH)
		return;
	n = GetEnvironmentVariableW(L"LOCALAPPDATA", local, PREFETCH_PATH);
	if (n == 0 || n >= PREFETCH_PATH)
		return;
	if (WideCharToMultiByte(CP_UTF8, 0, exe, -1, exe8, sizeof exe8, NULL, NULL) == 0)
		return;

	// One list per executable: FNV-1a hash of its path.
	uint64_t hash = 0xcbf29ce484222325ULL;
	for (const unsigned char* c = (const unsigned char*) exe8; *c; ++c) {
		hash ^= *c;
		hash *= 0x100000001b3ULL;
	}
	if (_snwprintf_s(prefetch_list, PREFETCH_PATH, _TRUNCATE,
			L"%ls\\UCSF\\ChimeraX\\Cache\\startup-prefetch-%016llx",
			local, (unsigned long long) hash) < 0)
		return;
	// Python records this startup's files here.  _wputenv_s updates the C
	// runtime's copy of the environment, which Python's os.environ reads.
	if (_wputenv_s(L"CHIMERAX_STARTUP_PREFETCH", prefetch_list) != 0)
		return;

	HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
	if (kernel != NULL)
		prefetch_virtual_memory = (prefetch_virtual_memory_fn)
			(void (*)(void)) GetProcAddress(kernel, "PrefetchVirtualMemory");

	HANDLE thread = CreateThread(NULL, 0, prefetch_files, NULL, 0, NULL);
	if (thread != NULL)
		CloseHandle(thread);
}
# else
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
# endif /* _WIN32 */
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
#ifdef __APPLE__
	start_qt_preload(argc, wargv);
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
