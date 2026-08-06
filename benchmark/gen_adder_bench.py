#!/usr/bin/env python3
"""Generate benchmark MLIR: subset-sum adder tree in a BBHT while loop.

Abstract adders = quantum.custom "Adder" with {width, strategy="unspecified"}.
Structure: A0 setup (outside loop, exec=1); inside a measurement-driven scf.while
(estimated_iterations = E[k], the profiled trip count): 4 leaves -> 2 branches ->
root, threaded so the pass's critical-path level analysis sees the tree.

Emits MLIR to stdout. Usage: gen_adder_bench.py <E[k]>
"""
import sys

EK = int(sys.argv[1]) if len(sys.argv) > 1 else 7

# widths per level
W_SETUP, W_LEAF, W_BRANCH, W_ROOT = 16, 8, 12, 16

print(f'''module {{
  func.func @subset_sum(%arg0: !quantum.reg) -> !quantum.bit {{
    %f = arith.constant false
    // 12 qubits: 0..7 data, 8..11 scratch
    %q0 = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %arg0[1] : !quantum.reg -> !quantum.bit
    // A0 setup adder (outside loop, exec=1)
    %s:2 = quantum.custom "Adder"() %q0, %q1 {{width = {W_SETUP} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
    // BBHT measurement-driven loop; estimated_iterations = profiled E[k]
    %r:2 = scf.while (%a0 = %s#0, %a1 = %s#1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit) {{
      %m, %mq = quantum.measure %a0 : i1, !quantum.bit
      scf.condition(%m) %mq, %a1 : !quantum.bit, !quantum.bit
    }} do {{
    ^bb0(%b0: !quantum.bit, %b1: !quantum.bit):
      // level-1 leaves (width {W_LEAF}), mutually independent -> parallel level
      %l1:2 = quantum.custom "Adder"() %b0, %b1 {{width = {W_LEAF} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      %l2:2 = quantum.custom "Adder"() %b0, %b1 {{width = {W_LEAF} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      %l3:2 = quantum.custom "Adder"() %b0, %b1 {{width = {W_LEAF} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      %l4:2 = quantum.custom "Adder"() %b0, %b1 {{width = {W_LEAF} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      // level-2 branches (width {W_BRANCH}); each consumes two leaf outputs
      %br1:2 = quantum.custom "Adder"() %l1#0, %l2#0 {{width = {W_BRANCH} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      %br2:2 = quantum.custom "Adder"() %l3#0, %l4#0 {{width = {W_BRANCH} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      // level-3 root (width {W_ROOT})
      %rt:2 = quantum.custom "Adder"() %br1#0, %br2#0 {{width = {W_ROOT} : i64, strategy = "unspecified"}} : !quantum.bit, !quantum.bit
      scf.yield %rt#0, %rt#1 : !quantum.bit, !quantum.bit
    }} attributes {{estimated_iterations = {EK} : i64}}
    return %r#0 : !quantum.bit
  }}
}}''')
