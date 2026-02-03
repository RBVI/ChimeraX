def install_kalign():
    import platform
    abi = (platform.system(), platform.machine())
    binaries = {
        ('Darwin', 'arm64'): ('kalign_darwin_arm64', 'kalign'),
        ('Darwin', 'x86_64'): ('kalign_darwin_x86_64', 'kalign'),
        ('Windows', 'x86_64'): ('kalign_windows_x86_64', 'kalign.exe'),
        ('Linux', 'x86_64'): ('kalign_linux_x86_64', 'kalign'),
        ('Linux', 'aarch64'): ('kalign_linux_aarch64', 'kalign')}

    if abi not in binaries:
        raise RuntimeError(f'OpenFold requires kalign software which is not available for platform {abi[0]} {abi[1]}')

    filename, exe_name = binaries[abi]
    
    url = f'https://github.com/RBVI/openfold-3/raw/refs/heads/main/kalign/{filename}'

    import sys
    from os.path import join, dirname
    save_path = join(dirname(sys.executable), exe_name)

    try:
        from urllib import request
        request.urlretrieve(url, save_path)
    except Exception as e:
        raise RuntimeError(f'Could not download kalign: {e}')

    import os
    os.chmod(save_path, 0o755)

install_kalign()
