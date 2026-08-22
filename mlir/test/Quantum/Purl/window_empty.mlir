// RUN: quantum-opt --purl="analyze-only=true calib=%S/backend_tiny.json" --verify-diagnostics %s
//
// With a tiny T2 the coherence ceiling C_max falls below the variance floor
// C_min: the window is empty, a compile-time diagnostic, and no transform.

func.func @carry_tinyT2() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 1) : !quantum.reg
  %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
  // expected-error@+1 {{empty cut-period window}}
  %res:2 = scf.while (%f = %true, %q = %q0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%f) %f, %q : i1, !quantum.bit
  } do {
  ^bb0(%fl: i1, %qq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit
    %q1 = quantum.custom "Hadamard"() %qq : !quantum.bit
    %areg1 = quantum.insert %areg[ 0], %a2 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %q1 : i1, !quantum.bit
  }
  %rr = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
