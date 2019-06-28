def clean_makefile():
    import sys, os, sysconfig
    build_path = sys.argv[1]
    makefile = sysconfig.get_makefile_filename()
    save_name = makefile + ".save"
    os.rename(makefile, save_name)
    with open(save_name) as input, open(makefile, "w") as output:
        for line in input:
            output.write(clean(line, build_path))


_arg_map = {
    "CC=": ["-I", "-L", "-fdebug-prefix-map="],
    "RUNSHARED=": ["DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH"],
    "BASECFLAGS=": ["-F"],
}

def clean(line, build_path):
    for prefix, flag_prefixes in _arg_map.items():
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            parts = [p for p in value.split() if keep(p, flag_prefixes)]
            line = prefix + ' '.join(parts) + '\n'
            break
    return line


def keep(part, flag_prefixes):
    for prefix in flag_prefixes:
        if part.startswith(prefix):
            return False
    return True


if __name__ == "__main__":
    clean_makefile()
