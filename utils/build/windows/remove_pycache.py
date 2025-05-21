import sys, os, os.path, shutil
cache = "__pycache__"
for d in sys.argv[1:]:
    for root, dirs, files in os.walk(d):
        print(root)
        if cache in dirs:
            print("rm", os.path.join(root, cache))
            shutil.rmtree(os.path.join(root, cache), ignore_errors=True)
            dirs.remove(cache)
import time
time.sleep(15)
