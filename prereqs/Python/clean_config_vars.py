import os
import sys


def clean_makefile():
    import sysconfig

    build_path = sys.argv[1]
    makefile = sysconfig.get_makefile_filename()
    save_name = makefile + ".save"
    os.rename(makefile, save_name)
    with open(save_name) as input, open(makefile, "w") as output:
        for line in input:
            output.write(clean_make_line(line, build_path))


_arg_map = {
    "CC": ["-I", "-L", "-fdebug-prefix-map="],
    "RUNSHARED": ["DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH"],
    "BASECFLAGS": ["-F"],
    "CONFINCLUDEPY": [""],
    "CONFINCLUDEDIR": [""],
    "INCLUDEPY": [""],
    "INCLUDEDIR": [""],
}

# python-build-standalone records absolute paths to the hermetic toolchain it
# was built with (e.g. .../tools/llvm/bin/llvm-ar).  Those paths do not exist
# on the build or run machine, so any sysconfig variable naming a build tool is
# rewritten to the system equivalent, resolved from PATH.
_tool_vars = (
    "AR", "RANLIB", "NM", "STRIP", "READELF",
    "CC", "CXX", "CPP", "CXXCPP",
    "LDSHARED", "LDCXXSHARED", "BLDSHARED", "LINKCC", "MAINCC",
)

# LLVM-prefixed tools are not generally on PATH; map them to the standard names.
_tool_rename = {
    "llvm-ar": "ar",
    "llvm-ranlib": "ranlib",
    "llvm-nm": "nm",
    "llvm-strip": "strip",
    "llvm-readelf": "readelf",
}


def clean_tool(value):
    parts = []
    for token in value.split():
        # An absolute path to a build tool: keep only the tool name so it is
        # found on PATH.  Skip script paths (e.g. a build-python launcher).
        if token.startswith("/") and "/bin/" in token and not token.endswith(".py"):
            base = os.path.basename(token)
            parts.append(_tool_rename.get(base, base))
        else:
            parts.append(token)
    return " ".join(parts)


def clean_make_line(line, build_path):
    key, value = line.split("=", 1)
    try:
        flag_prefixes = _arg_map[key]
    except KeyError:
        return line
    else:
        return key + "=" + clean(value.strip(), flag_prefixes) + "\n"


def clean(value, flag_prefixes):
    return " ".join([p for p in value.split() if keep(p, flag_prefixes)])


def keep(part, flag_prefixes):
    for prefix in flag_prefixes:
        if part.startswith(prefix):
            return False
    return True


def clean_sysconfigdata():
    import sysconfig, os.path, pprint

    build_path = sys.argv[1]
    libdir = os.path.dirname(sysconfig.__file__)
    configdata = sysconfig._get_sysconfigdata_name()
    configpath = os.path.join(libdir, configdata + ".py")
    mod = __import__(configdata, globals(), locals(), ["build_time_vars"], 0)
    print("mod", configdata, mod)
    clean_vars = {}
    for key, value in mod.build_time_vars.items():
        clean_vars[key] = clean_data_value(key, value, build_path)
    # python-build-standalone records an empty LIBDIR, and it is built with
    # --enable-shared, so distutils appends LIBDIR to the link library path when
    # building extensions.  An empty value yields a bare "-L" that swallows the
    # following argument and breaks linking, so point LIBDIR at this
    # interpreter's real lib directory (where libpython lives).
    if os.name == "posix":
        clean_vars["LIBDIR"] = os.path.join(sys.base_prefix, "lib")
    print("path", configpath)
    with open(configpath, "w") as output:
        print(
            "# system configuration generated and used by the sysconfig module",
            file=output,
        )
        print("# cleaned as part of ChimeraX build", file=output)
        print("build_time_vars = ", file=output, end="")
        pprint.pprint(clean_vars, stream=output)


def clean_data_value(key, value, build_path):
    if key in _tool_vars and isinstance(value, str):
        value = clean_tool(value)
    try:
        flag_prefixes = _arg_map[key]
    except KeyError:
        return value
    else:
        return clean(value, flag_prefixes)


if __name__ == "__main__":
    # clean_makefile()
    clean_sysconfigdata()
