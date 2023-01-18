# Convert input fasta file with multi-line sequences to single-line sequences

from sys import stdin, stdout
first = True
while True:
    line = stdin.readline()
    if not line:
        break
    elif line.startswith('>'):
        if not first:
            stdout.write('\n')
        stdout.write(line)
        first = False
    else:
        stdout.write(line.strip())

stdout.flush()
