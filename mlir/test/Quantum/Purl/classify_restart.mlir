// RUN: quantum-opt --purl="analyze-only=true" %s | FileCheck %s
//
// Coin-flip: the carried qubit is measured and PauliX-reset every iteration
// (measure + outcome-conditioned X). All quantum slots RESET -> class "restart";
// the pass does not modify the loop.

// CHECK: purl.class = "restart"
// CHECK-NOT: purl.carry_slot
// CHECK-NOT: purl.C

func.func @coin_flip() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 1) : !quantum.reg
  %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
  %res:2 = scf.while (%f = %true, %q = %q0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%f) %f, %q : i1, !quantum.bit
  } do {
  ^bb0(%fl: i1, %qq: !quantum.bit):
    %h = quantum.custom "Hadamard"() %qq : !quantum.bit
    %m, %qm = quantum.measure %h : i1, !quantum.bit
    %qr = scf.if %m -> (!quantum.bit) {
      %x = quantum.custom "PauliX"() %qm : !quantum.bit
      scf.yield %x : !quantum.bit
    } else {
      scf.yield %qm : !quantum.bit
    }
    scf.yield %m, %qr : i1, !quantum.bit
  }
  %rr = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
