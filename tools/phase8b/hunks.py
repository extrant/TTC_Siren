#!/usr/bin/env python
"""Builds an intermediate version of a file that contains only some of the changes between HEAD and the working tree.

Used to cut one working tree into commits that each hold ONE logical change (a helper of the work process, not part of the product):

    python tools/phase8b/hunks.py list  <file>                     numbered changes with a one-line summary
    python tools/phase8b/hunks.py build <file> <regex> [<regex>..] writes HEAD + the changes whose text matches ANY regex into the working tree
    python tools/phase8b/hunks.py final <file> <saved-copy>        restores the saved final version

A "change" is one block of lines that differs (difflib opcodes over lines). With every change selected the result equals the working tree.
"""
import difflib
import re
import subprocess
import sys


def head_text(path):
    return subprocess.run(['git', 'show', 'HEAD:' + path], capture_output=True, check=True).stdout.decode('utf-8').replace('\r\n', '\n')


def changes(path, work_lines):
    old = head_text(path).splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, old, work_lines, autojunk=False)
    return old, sm.get_opcodes()


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == 'final':
        src = sys.argv[3]
        open(path, 'w', encoding='utf-8', newline='').write(open(src, encoding='utf-8', newline='').read())
        return
    work = open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n').splitlines(keepends=True)
    old, ops = changes(path, work)
    diffs = [(i, op) for i, op in enumerate(ops) if op[0] != 'equal']
    if cmd == 'list':
        for n, (_, (tag, i1, i2, j1, j2)) in enumerate(diffs):
            new_text = ''.join(work[j1:j2])
            old_text = ''.join(old[i1:i2])
            first = (new_text or old_text).strip().splitlines()[0][:110] if (new_text or old_text).strip() else '(blank lines)'
            print('%3d %-7s -%d +%d  %s' % (n, tag, i2 - i1, j2 - j1, first))
        return
    if cmd == 'build':
        patterns = [re.compile(p, re.S) for p in sys.argv[3:]]
        out = []
        chosen = 0
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                out.extend(old[i1:i2])
                continue
            text = ''.join(old[i1:i2]) + '\n' + ''.join(work[j1:j2])
            if any(p.search(text) for p in patterns):
                out.extend(work[j1:j2])
                chosen += 1
            else:
                out.extend(old[i1:i2])
        open(path, 'w', encoding='utf-8', newline='').write(''.join(out))
        print('%s: %d of %d changes applied' % (path, chosen, len(diffs)))


if __name__ == '__main__':
    main()
