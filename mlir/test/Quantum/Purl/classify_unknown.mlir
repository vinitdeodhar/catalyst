// RUN: quantum-opt --purl="analyze-only=true" --verify-diagnostics %s | FileCheck %s
//
// The carried qubit is measured but NOT provably reset -> class "unknown"; the
// pass emits a remark and does not fire.

// CHECK: purl.class = "unknown"
// CHECK-NOT: purl.carry_slot

func.func @unknown_loop() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 1) : !quantum.reg
  %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
  // expected-remark@+1 {{purl: unknown class; pass does not fire}}
  %res:2 = scf.while (%f = %true, %q = %q0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%f) %f, %q : i1, !quantum.bit
  } do {
  ^bb0(%fl: i1, %qq: !quantum.bit):
    %h = quantum.custom "Hadamard"() %qq : !quantum.bit
    %m, %qm = quantum.measure %h : i1, !quantum.bit
    scf.yield %m, %qm : i1, !quantum.bit
  }
  %rr = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
