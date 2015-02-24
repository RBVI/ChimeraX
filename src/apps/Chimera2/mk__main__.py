# vi: set expandtab shiftwidth=4 softtabstop=4:

import sys

if len(sys.argv) != 2:
    print("usage: %s python-file" % sys.argv[0], file=sys.stderr)
    raise SystemExit(2)

filename = sys.argv[1]
if not filename.endswith('.py'):
    print('only works with Python source files', file=sys.stderr)
    raise SystemExit(1)

module_name = filename[0:-3]
main = open('__main__.py', 'w', encoding='utf-8')
print('from %s import *  # noqa\n' % module_name, file=main)

with open(sys.argv[1], 'rU', encoding='utf-8') as f:
    in_main = False
    first_main = True
    indent = ''
    line_num = 0
    for line in f.readlines():
        line_num += 1
        if not in_main:
            tokens = line.split()
            if tokens[0:3] == ['if', '__name__', '==']:
                in_main = True
            continue
        if first_main:
            indent = line[0: len(line) - len(line.lstrip())]
            first_main = False
        if line[0:len(indent)] != indent:
            print('inconsistent indenting on line %d' % line_num,
                  file=sys.stderr)
            raise SystemExit(1)

        print(line[len(indent):], end='', file=main)

main.close()
