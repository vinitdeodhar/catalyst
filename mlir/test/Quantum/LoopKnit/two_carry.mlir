// RUN: quantum-opt --loop-knit="analyze-only=true" --verify-diagnostics %s
//
// Two live qubits cross the carry un-measured -> multi-wire cut, unsupported.

func.func @two_carry() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 2) : !quantum.reg
  %p0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
  %q0 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
  // expected-error@+1 {{multi-wire cut unsupported}}
  %res:3 = scf.while (%f = %true, %p = %p0, %q = %q0)
      : (i1, !quantum.bit, !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
    scf.condition(%f) %f, %p, %q : i1, !quantum.bit, !quantum.bit
  } do {
  ^bb0(%fl: i1, %pp: !quantum.bit, %qq: !quantum.bit):
    %p1 = quantum.custom "Hadamard"() %pp : !quantum.bit
    %q1 = quantum.custom "T"() %qq : !quantum.bit
    scf.yield %fl, %p1, %q1 : i1, !quantum.bit, !quantum.bit
  }
  %r0 = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  %r1 = quantum.insert %r0[ 1], %res#2 : !quantum.reg, !quantum.bit
  quantum.dealloc %r1 : !quantum.reg
  return %res#0 : i1
}
