"""The native macOS preloader never turns headless invocations into GUI loads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS launcher optimization')
HEADER = Path(__file__).resolve().parents[1] / 'src/apps/ChimeraX/startup_qt.h'


@pytest.fixture(scope='module')
def preloader(tmp_path_factory):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('C compiler unavailable')
    directory = tmp_path_factory.mktemp('native-qt-preload')
    source = directory / 'check.c'
    source.write_text(r'''
#include <dlfcn.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
static int attempts, loads, flags;
static const char* mode;
static char loaded_path[PATH_MAX];
static int fake_executable(char* path, uint32_t* size) {
    if (strcmp(mode, "executable-failure") == 0) return 1;
    strcpy(path, "/private/Example.app/Contents/bin/ChimeraX");
    return 0;
}
static char* fake_realpath(const char* path, char* resolved) {
    if (strcmp(mode, "realpath-failure") == 0) return NULL;
    strcpy(resolved, path);
    return resolved;
}
static int fake_access(const char* path, int mode_bits) {
    return strcmp(mode, "missing-library") == 0 ? -1 : 0;
}
static void* fake_dlopen(const char* path, int open_flags) {
    loads++;
    flags = open_flags;
    strcpy(loaded_path, path);
    return NULL; // Failed preloading must also be harmless.
}
static int fake_create(pthread_t* thread, const pthread_attr_t* attr,
        void* (*start)(void*), void* arg) {
    attempts++;
    if (strcmp(mode, "thread-failure") == 0) return 11;
    start(arg);
    return 0;
}
#define _NSGetExecutablePath fake_executable
#define realpath fake_realpath
#define access fake_access
#define dlopen fake_dlopen
#define pthread_create fake_create
#include "startup_qt.h"
int main(int argc, char** argv) {
    mode = getenv("PRELOAD_CHECK_MODE");
    if (!mode) mode = "normal";
    wchar_t* wide[argc];
    for (int i = 0; i < argc; i++) {
        wide[i] = calloc(strlen(argv[i]) + 1, sizeof(wchar_t));
        mbstowcs(wide[i], argv[i], strlen(argv[i]) + 1);
    }
    start_qt_preload(argc, wide);
    printf("{\"attempts\":%d,\"loads\":%d,\"local_lazy\":%d,\"path\":\"%s\"}\n",
        attempts, loads, flags == (RTLD_LAZY | RTLD_LOCAL), loaded_path);
    for (int i = 0; i < argc; i++) free(wide[i]);
    return 0;
}
''')
    binary = directory / 'check'
    subprocess.run([compiler, '-std=c11', '-I', str(HEADER.parent),
        '-DPY_MAJOR_VERSION=' + str(sys.version_info.major),
        '-DPY_MINOR_VERSION=' + str(sys.version_info.minor),
        str(source), '-o', str(binary)], check=True, capture_output=True, text=True)
    def run(*args, mode='normal', disabled=False):
        env = dict(os.environ, PRELOAD_CHECK_MODE=mode)
        env.pop('CHIMERAX_NO_QT_PRELOAD', None)
        if disabled:
            env['CHIMERAX_NO_QT_PRELOAD'] = '1'
        result = subprocess.run([str(binary), *args], env=env,
            check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    return run


def test_gui_loads_only_the_private_framework_and_tolerates_dlopen_failure(preloader):
    result = preloader('--script', 'example.py')
    assert result['attempts'] == result['loads'] == 1
    assert result['local_lazy']
    assert result['path'] == ('/private/Example.app/Contents/bin/../lib/'
        f'python{sys.version_info.major}.{sys.version_info.minor}/site-packages/'
        'PySide6/Qt/lib/QtWidgets.framework/Versions/A/QtWidgets')


@pytest.mark.parametrize('option', ['--nogui', '--nog', '--offscreen', '--off', '--version',
    '--help', '--listioformats', '--disable-qt', '--disable', '--uninstall', '-m', '-c',
    '-I', '-u', '-V', '-cprint(1)'])
def test_headless_options_do_not_start_a_loader_thread(preloader, option):
    result = preloader(option)
    assert result['attempts'] == result['loads'] == 0


@pytest.mark.parametrize('mode', ['executable-failure', 'realpath-failure', 'missing-library'])
def test_unavailable_bundle_layout_keeps_the_normal_loading_path(preloader, mode):
    result = preloader(mode=mode)
    assert result['attempts'] == result['loads'] == 0


def test_thread_creation_failure_is_harmless(preloader):
    result = preloader(mode='thread-failure')
    assert result['attempts'] == 1 and result['loads'] == 0


def test_environment_override_disables_preloading(preloader):
    result = preloader(disabled=True)
    assert result['attempts'] == result['loads'] == 0
