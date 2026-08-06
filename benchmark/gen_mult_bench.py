#!/usr/bin/env python3
"""Multiplier (shift-and-add): straight-line adders with GROWING widths.
Static (exec=1 each), sequential critical path; the pass spends the budget on the
widest adders (largest depth_saved). Shows width-driven allocation (no E[k])."""
widths = [4, 6, 8, 10, 12, 14]
lines = ['module {', '  func.func @multiplier(%r: !quantum.reg) -> !quantum.bit {',
         '    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit',
         '    %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit']
prev0, prev1 = '%q0', '%q1'
for i, w in enumerate(widths):
    lines.append(f'    %m{i}:2 = quantum.custom "Adder"() {prev0}, {prev1} '
                 f'{{width = {w} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit')
    prev0, prev1 = f'%m{i}#0', f'%m{i}#1'
lines += [f'    return {prev0} : !quantum.bit', '  }', '}']
print('\n'.join(lines))
