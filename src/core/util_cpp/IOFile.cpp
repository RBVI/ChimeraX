// vi: set expandtab ts=4 sw=4:
#include "IOFile.h"

#include <stdlib.h>
#include <sys/types.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# define STRICT
# include <Windows.h>
# include <direct.h>
# include <sys/stat.h>
#else
# include <unistd.h>
# include <pwd.h>
# include <sys/stat.h>
#endif

#ifndef S_ISDIR
# define S_ISDIR(x) (((x) & S_IFMT) == S_IFDIR)
#endif

namespace util {

bool
path_exists(const std::string &path, bool asFile)
{
#ifdef _WIN32
#ifdef _USE_32BIT_TIME_T
    struct _stat32 statbuf;
#else
    struct _stat64i32 statbuf;
#endif
    // path is UTF-8, stat needs mbcs encoding, wstat uses UTF-16
    std::vector<wchar_t> wpath(path.length() + 1);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.length(),
                    &wpath[0], path.length() + 1);
    if (_wstat(&wpath[0], &statbuf) < 0)
        return false;
    static const int READABLE = S_IREAD;
#else
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) < 0)
        return false;
    static const int READABLE = S_IRUSR|S_IRGRP|S_IROTH;
#endif
    bool isDir = S_ISDIR(statbuf.st_mode);
    return (statbuf.st_mode & READABLE) != 0 && (asFile ? !isDir : isDir);
}

InputFile::InputFile(const std::string &filename): ifs_(NULL)
{
#ifndef _WIN32
    ifs_ = new std::ifstream(filename.c_str());
#else
    // filename is UTF-8, want UTF-16
    std::vector<wchar_t> wpath(filename.length() + 1);
    MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), filename.length(),
                    &wpath[0], filename.length() + 1);
    ifs_ = new std::ifstream(&wpath[0]);
#endif
}

InputFile::~InputFile()
{
    if (ifs_ == NULL)
        return;
    ifs_->close();
    delete ifs_;
}

std::ifstream &InputFile::ifstream() const
{
    return *ifs_;
}

OutputFile::OutputFile(const std::string &filename): ofs_(NULL)
{
#ifndef _WIN32
    ofs_ = new std::ofstream(filename.c_str());
#else
    // filename is UTF-8, want UTF-16
    std::vector<wchar_t> wpath(filename.length() + 1);
    MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), filename.length(),
                    &wpath[0], filename.length() + 1);
    ofs_ = new std::ofstream(&wpath[0]);
#endif
}

OutputFile::~OutputFile()
{
    if (ofs_ == NULL)
        return;
    ofs_->close();
    delete ofs_;
}

std::ofstream &OutputFile::ofstream() const
{
    return *ofs_;
}

}  // namespace util
