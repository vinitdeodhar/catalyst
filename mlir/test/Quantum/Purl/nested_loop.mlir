// RUN: quantum-opt --purl="analyze-only=true" --verify-diagnostics %s
//
// A nested dynamic while inside the body has no static B -> the pass must
// require the inner loop to be bounded first.

func.func @nested() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 1) : !quantum.reg
  %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
  // expected-error@+1 {{nested dynamic loop: bound inner loop first}}
  %res:2 = scf.while (%f = %true, %q = %q0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%f) %f, %q : i1, !quantum.bit
  } do {
  ^bb0(%fl: i1, %qq: !quantum.bit):
    %q1 = quantum.custom "Hadamard"() %qq : !quantum.bit
    %inner:2 = scf.while (%g = %fl, %r = %q1) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
      scf.condition(%g) %g, %r : i1, !quantum.bit
    } do {
    ^bb0(%gg: i1, %rr: !quantum.bit):
      %h = quantum.custom "Hadamard"() %rr : !quantum.bit
      %m, %rm = quantum.measure %h : i1, !quantum.bit
      scf.yield %m, %rm : i1, !quantum.bit
    }
    scf.yield %inner#0, %inner#1 : i1, !quantum.bit
  }
  %rr = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
