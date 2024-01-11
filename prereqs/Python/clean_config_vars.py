def clean_makefile():
    import sys, os, sysconfig
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

def clean_make_line(line, build_path):
    key, value = line.split('=', 1)
    try:
        flag_prefixes = _arg_map[key]
    except KeyError:
        return line
    else:
        return key + '=' + clean(value.strip(), flag_prefixes) + '\n'


def clean(value, flag_prefixes):
    return ' '.join([p for p in value.split() if keep(p, flag_prefixes)])


def keep(part, flag_prefixes):
    for prefix in flag_prefixes:
        if part.startswith(prefix):
            return False
    return True


def clean_sysconfigdata():
    import sys, sysconfig, os.path, pprint
    build_path = sys.argv[1]
    libdir = os.path.dirname(sysconfig.__file__)
    configdata = sysconfig._get_sysconfigdata_name()
    configpath = os.path.join(libdir, configdata + ".py")
    mod = __import__(configdata, globals(), locals(), ["build_time_vars"], 0)
    print("mod", configdata, mod)
    clean_vars = {}
    for key, value in mod.build_time_vars.items():
        clean_vars[key] = clean_data_value(key, value, build_path)
    print("path", configpath)
    with open(configpath, "w") as output:
        print("# system configuration generated and used by"
              " the sysconfig module", file=output)
        print("# cleaned as part of ChimeraX build", file=output)
        print("build_time_vars = ", file=output, end='')
        pprint.pprint(clean_vars, stream=output)


def clean_data_value(key, value, build_path):
    try:
        flag_prefixes = _arg_map[key]
    except KeyError:
        return value
    else:
        return clean(value, flag_prefixes)


if __name__ == "__main__":
    # clean_makefile()
    clean_sysconfigdata()
