#!/usr/bin/env python3
"""Pretty-MD helper. Emits PLAN.md-style boxed/coloured output."""
import sys

ESC  = '\x1b'
BR   = f'{ESC}[1;37m'   # white  — borders / titles
HD   = f'{ESC}[1;33m'   # yellow — view hint
SC   = f'{ESC}[1;35m'   # magenta — section labels
CY   = f'{ESC}[1;36m'   # cyan — column headers / inline emphasis
GR   = f'{ESC}[1;32m'   # green — done / success
YE   = f'{ESC}[1;33m'   # yellow — caution
RE   = f'{ESC}[1;31m'   # red — danger / warning
W    = f'{ESC}[1;37m'   # bold white inline
RS   = f'{ESC}[0m'      # reset
BAR  = '═'
SBR  = '─'

W_OUTER = 122   # outer box / divider width

def box(title, width=W_OUTER):
    """Title comes in already spaced (e.g. 'N T T   /   M I 3 0 0 A').
    Centering uses visible width (vwidth) so em-dashes / wide chars stay
    aligned, consistent with tablize()."""
    bar = BAR * (width - 2)
    inner = width - 2
    tw = vwidth(title)
    pad_l = max(0, (inner - tw) // 2)
    pad_r = max(0, inner - pad_l - tw)
    out  = [f'{BR}╔{bar}╗']
    out += [f'║{" "*pad_l}{title}{" "*pad_r}║']
    out += [f'╚{bar}╝{RS}']
    return '\n'.join(out)

def hint(name):
    return f'{HD}  View with:  cat {name}   or   less -R {name}{RS}'

def divider(width=W_OUTER):
    return f'{BR}{BAR*width}{RS}'

def section(label, suffix=''):
    """Magenta section label, 2-space indent. Suffix (if any) is part of
    the same coloured run + reset, matching the original STATUS.md style
    (e.g. 'OPEN TASKS - 6900 XT  [do these first]' all magenta)."""
    body = f'  {label}'
    if suffix:
        body += f'  {suffix}'
    return f'{SC}{body}{RS}'

def cyan(text):  return f'{CY}{text}{RS}'
def green(text): return f'{GR}{text}{RS}'
def red(text):   return f'{RE}{text}{RS}'
def yel(text):   return f'{YE}{text}{RS}'
def bold(text):  return f'{W}{text}{RS}'

import re, unicodedata
_ANSI = re.compile(r'\x1b\[[0-9;]*m')

def vwidth(s):
    """Visible monospace width: strip ANSI then sum East-Asian wide=2, else 1.
    Handles em-dash, en-dash, Greek letters, etc. as 1 char each."""
    s = _ANSI.sub('', s)
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            w += 2
        else:
            w += 1
    return w

def tablize(headers, rows, indent='  ', highlight_header=True):
    """Render an aligned, ANSI-coloured table.
    headers: list[str] — plain text (will be wrapped in cyan)
    rows:    list[list[str]] — plain text or pre-coloured cells (any ANSI)
    Returns a single string; visible alignment is correct even with ANSI codes."""
    ncol = len(headers)
    # column visible width = max of header and all rows
    col_w = []
    for c in range(ncol):
        cells = [headers[c]] + [r[c] if c < len(r) else '' for r in rows]
        col_w.append(max(vwidth(s) for s in cells))
    def pad(s, w):
        return s + ' ' * (w - vwidth(s))
    def line(cells, fill_char='│', edge_l='│', edge_r='│'):
        out = [edge_l]
        for c, cell in enumerate(cells):
            out.append(' ' + pad(cell, col_w[c]) + ' ' + fill_char)
        # replace last fill_char with edge_r
        out[-1] = ' ' + pad(cells[-1], col_w[ncol-1]) + ' ' + edge_r
        return ''.join(out)
    def rule(left, mid, right):
        bars = ['─' * (col_w[c] + 2) for c in range(ncol)]
        return left + mid.join(bars) + right
    out = []
    out.append(indent + rule('┌', '┬', '┐'))
    hdr = [cyan(h) for h in headers] if highlight_header else list(headers)
    out.append(indent + line(hdr))
    out.append(indent + rule('├', '┼', '┤'))
    for r in rows:
        # pad short rows
        r = list(r) + [''] * (ncol - len(r))
        out.append(indent + line(r))
    out.append(indent + rule('└', '┴', '┘'))
    return '\n'.join(out)

def write(path, blocks):
    """blocks: list of strings; joined with a blank line between each.
    Always UTF-8 with '\\n' newlines so box-drawing / ESC bytes are written
    identically regardless of system locale."""
    body = '\n\n'.join(blocks).rstrip() + '\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(body)
    print(f'wrote {path} ({len(body.encode("utf-8"))} bytes, '
          f'{body.count(chr(10))} lines)')

if __name__ == '__main__':
    # Smoke test
    print(box('Smoke Test'))
    print()
    print(hint('foo.md'))
    print()
    print(divider())
    print(section('1. EXAMPLE', '(suffix)'))
    print()
    print(f'  {cyan("cyan")} {green("green")} {red("red")} {yel("yellow")} {bold("bold")}')
