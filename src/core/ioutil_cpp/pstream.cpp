// Copyright (c) 1989,2001 The Regents of the University of California.
// All rights reserved.
// 
// Redistribution and use in source and binary forms are permitted
// provided that the above copyright notice and this paragraph are
// duplicated in all such forms and that any documentation,
// distribution and/or use acknowledge that the software was developed
// by the Computer Graphics Laboratory, University of California,
// San Francisco.  The name of the University may not be used to
// endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
// WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY OF CALIFORNIA BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS SOFTWARE.
//
// $Id: pstream.cpp 36239 2012-04-26 00:09:34Z goddard $
#include "pstream.h"
#include <limits.h>
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# define STRICT
# include <windows.h>
# define PIPE_BUFSIZE 4096
# include <stdlib.h>
# include <vector>
# include <string.h>
#else
# include <unistd.h>
# include <sys/types.h>
# include <sys/wait.h>
// Need both <sys/errno.h> and <errno.h> to get EINTR on all systems
// (should only need <errno.h>.
# include <sys/errno.h>
# include <errno.h>
#endif

namespace ioutil {

raw_pipe2::raw_pipe2(): child(0)
{
#ifdef _WIN32
	fd[0] = fd[1] = NULL;
#else
	fd[0] = fd[1] = -1;
#endif
	bufsize[0] = bufsize[1] = 0;
}

bool
raw_pipe2::is_open() const
{
#ifdef _WIN32
	return fd[0] != NULL || fd[1] != NULL;
#else
	return fd[0] != -1 || fd[1] != -1;
#endif
}

// raw_pipe2::open:
//	Like popen but fills in an array of two file descriptors
//	with the stdin and stdout of the child process.

int
raw_pipe2::open(const char *cmd, int mode)
{
#ifdef _WIN32
	// expect that cmd is utf-8
	HANDLE in_pipe[2], out_pipe[2];	// From parent's POV
	HANDLE tmp;
	// From "help cmd" in section on filename completion:
	//static const char SpecialChars[] = "&()[]{}^=;!'+,`~";
	// From "help cmd" in section on quote processing for /C:
	static const char SpecialChars[] = "&<>()@^|";

	// Windows programs expand filename wildcard arguments directly,
	// i.e., without the use of a shell -- so we only need to figure
	// which shell to use if there are special shell characters in
	// the command.
	std::wstring shell;
	if (strpbrk(cmd, SpecialChars) != NULL) {
		const wchar_t *shell_env = _wgetenv(L"COMSPEC");
		if (shell_env)
			shell = shell_env;
		else {
			OSVERSIONINFO osvi;
			osvi.dwOSVersionInfoSize = sizeof osvi;
			if (GetVersionEx(&osvi))
				switch (osvi.dwPlatformId) {
				  case VER_PLATFORM_WIN32s:
					// Windows 3.1
					break;
				  case VER_PLATFORM_WIN32_WINDOWS:
					// Windows 95/98/Me
					shell = L"command.com";
					break;
				  default:
				  case VER_PLATFORM_WIN32_NT:
					// Windows NT3.51/NT4.0/2000/XP/.NET
					shell = L"cmd.exe";
					break;
				}
			if (shell.empty())
				return -1;
		}
		if (shell.find(L' ') != std::wstring::npos) {
#ifndef _MSC_VER
			shell = '"' + shell + '"';
#else
			std::wstring tmp(L"\"");
			tmp += shell;
			tmp += L'"';
			shell = tmp;
#endif
		}
	}

	// need non-const command string for CreateProcess to mangle
	std::wstring winCmd;
	size_t cmdlen = strlen(cmd);
	winCmd.resize(cmdlen + 1);
	MultiByteToWideChar(CP_UTF8, 0, cmd, cmdlen, &winCmd[0], cmdlen + 1);
	if (!shell.empty()) {
#ifndef _MSC_VER
		winCmd = shell + L" /c \"" + winCmd + L"\"";
#else
		std::wstring tmp(shell);
		tmp += L" /c \"";
		tmp += winCmd;
		tmp += L"\"";
		winCmd = tmp;
#endif
	}

	// create pipes to process
	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof sa;
	sa.lpSecurityDescriptor = NULL;
	sa.bInheritHandle = TRUE;
	if (!::CreatePipe(&in_pipe[0], &in_pipe[1], &sa, 0))
		return -1; // replace parent's handle with a non-inheritable one
	if (!::DuplicateHandle(::GetCurrentProcess(), 
			in_pipe[0], ::GetCurrentProcess(), &tmp, 0, FALSE,
			DUPLICATE_SAME_ACCESS)) {
		::CloseHandle(in_pipe[0]);
		::CloseHandle(in_pipe[1]);
		return -1;
	}
	::CloseHandle(in_pipe[0]);
	in_pipe[0] = tmp;

	if (!::CreatePipe(&out_pipe[0], &out_pipe[1], &sa, 0)) {
		::CloseHandle(in_pipe[0]);
		::CloseHandle(in_pipe[1]);
		return -1;
	}
	// replace parent's handle with a non-inheritable one
	if (!::DuplicateHandle(::GetCurrentProcess(), 
			out_pipe[1], ::GetCurrentProcess(), &tmp, 0, FALSE,
			DUPLICATE_SAME_ACCESS)) {
		::CloseHandle(in_pipe[0]);
		::CloseHandle(in_pipe[1]);
		::CloseHandle(out_pipe[0]);
		::CloseHandle(out_pipe[1]);
		return -1;
	}
	::CloseHandle(out_pipe[1]);
	out_pipe[1] = tmp;

	STARTUPINFOW sui;
	::ZeroMemory(&sui, sizeof sui);
	sui.cb = sizeof sui;
	sui.dwFlags = STARTF_USESTDHANDLES;
	sui.hStdInput = out_pipe[0];
	sui.hStdOutput = in_pipe[1];
	sui.hStdError = ::GetStdHandle(STD_ERROR_HANDLE);
	PROCESS_INFORMATION pi;
	::ZeroMemory(&pi, sizeof pi);
	BOOL created;
	if (shell.empty())
		created = ::CreateProcessW(NULL, &winCmd[0], NULL, NULL, TRUE,
						0, NULL, NULL, &sui, &pi);
	else
		created = ::CreateProcessW(shell.c_str(), &winCmd[0], NULL,
					NULL, TRUE, 0, NULL, NULL, &sui, &pi);
	if (!created) {
		::CloseHandle(in_pipe[0]);
		::CloseHandle(in_pipe[1]);
		::CloseHandle(out_pipe[0]);
		::CloseHandle(out_pipe[1]);
		return -1;
	}
	child = pi.hProcess;
	::CloseHandle(pi.hThread);

	::CloseHandle(in_pipe[1]);
	::CloseHandle(out_pipe[0]);
	if (mode & std::ios::in) {
		fd[0] = in_pipe[0];
		bufsize[0] = PIPE_BUFSIZE;
	} else {
		fd[0] = NULL;
		::CloseHandle(in_pipe[0]);
	}
	if (mode & std::ios::out) {
		fd[1] = out_pipe[1];
		bufsize[1] = PIPE_BUFSIZE;
	} else {
		fd[1] = NULL;
		::CloseHandle(out_pipe[1]);
	}
#else
	int in_pipe[2], out_pipe[2];	// From parent's POV

	const char *shell;
	if ((shell = ::getenv("SHELL")) == NULL)
		shell = "/bin/sh";

	if (::pipe(in_pipe) < 0)
		return -1;
	if (::pipe(out_pipe) < 0) {
		(void) ::close(in_pipe[0]);
		(void) ::close(in_pipe[1]);
		return -1;
	}
	if ((child = fork()) < 0) {
		(void) ::close(in_pipe[0]);
		(void) ::close(in_pipe[1]);
		(void) ::close(out_pipe[0]);
		(void) ::close(out_pipe[1]);
		return -1;
	}
	if (child == 0) {
		(void) ::dup2(out_pipe[0], 0);
		(void) ::dup2(in_pipe[1], 1);
		(void) ::close(in_pipe[0]);
		(void) ::close(in_pipe[1]);
		(void) ::close(out_pipe[0]);
		(void) ::close(out_pipe[1]);
		(void) ::execl(shell, shell, "-c", cmd, (char *) NULL);
		::_exit(1);
	}
	(void) ::close(in_pipe[1]);
	(void) ::close(out_pipe[0]);
	if (mode & std::ios::in) {
		fd[0] = in_pipe[0];
		bufsize[0] = ::fpathconf(fd[0], _PC_PIPE_BUF);
	} else {
		fd[0] = -1;
		::close(in_pipe[0]);
	}
	if (mode & std::ios::out) {
		fd[1] = out_pipe[1];
		bufsize[1] = ::fpathconf(fd[1], _PC_PIPE_BUF);
	} else {
		fd[1] = -1;
		::close(out_pipe[1]);
	}
#endif
	return 0;
}

// raw_pipe2::close:
//	Close both streams.

int
raw_pipe2::close(int mode)
{
	if (!child)
		return 0;
#ifdef _WIN32
	if ((mode & std::ios::in) && fd[0] != NULL) {
		::CloseHandle(fd[0]);
		fd[0] = NULL;
	}
	if ((mode & std::ios::out) && fd[1] != NULL) {
		::CloseHandle(fd[1]);
		fd[1] = NULL;
	}
	if (is_open())
		return 0;
	DWORD timeout = useBlockingWait ? INFINITE : 0;
	DWORD result = ::WaitForSingleObject(child, timeout);
	DWORD status = 0;
	BOOL valid = FALSE;
	if (result != WAIT_FAILED)
		valid  = ::GetExitCodeProcess(child, &status);
	::CloseHandle(child);
	child = NULL;
	if (valid && status != STILL_ACTIVE && status > 0)
		return -1;
#else
	if ((mode & std::ios::in) && fd[0] != -1) {
		(void) ::close(fd[0]);
		fd[0] = -1;
	}
	if ((mode & std::ios::out) && fd[1] != -1) {
		(void) ::close(fd[1]);
		fd[1] = -1;
	}
	if (is_open())
		return 0;
	// We wait for our child to exit.
	int flags = useBlockingWait ? 0 : WNOHANG;
	int status = 0;
	while (::waitpid(child, &status, flags) < 0) {
		if (errno == EINTR)
			continue;
		break;
	}
	child = 0;
	if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
		return -1;
#endif
	return 0;
}

int
raw_pipe2::read(void *buf, int count)
{
#ifdef _WIN32
	DWORD num;
	if (!::ReadFile(fd[0], buf, count, &num, NULL))
		return -1;
	return num;
#else
	return ::read(fd[0], buf, count);
#endif
}

int
raw_pipe2::write(const void *buf, int count)
{
#ifdef _WIN32
	DWORD num;
	if (!::WriteFile(fd[1], buf, count, &num, NULL))
		return -1;
	return num;
#else
	return ::write(fd[1], buf, count);
#endif
}

} // namespace ioutil
