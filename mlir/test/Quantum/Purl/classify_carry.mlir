// RUN: quantum-opt --purl="analyze-only=true" %s | FileCheck %s
//
// RUS-shaped register carry: slot 0 is the data/target qubit (threaded through
// the loop un-measured -> CARRY); slot 1 is an ancilla measured + reset every
// iteration (RESET). Loop class = "carry", carry slot = 0.

// CHECK-DAG: purl.class = "carry"
// CHECK-DAG: purl.carry_slot = 0 : i64

func.func @rus_reg() -> i1 {
  %true = arith.constant true
  %reg = quantum.alloc( 2) : !quantum.reg
  %res:2 = scf.while (%f = %true, %r = %reg) : (i1, !quantum.reg) -> (i1, !quantum.reg) {
    scf.condition(%f) %f, %r : i1, !quantum.reg
  } do {
  ^bb0(%fl: i1, %rr: !quantum.reg):
    // ancilla slot 1: entangle, measure, reset (RESET)
    %a = quantum.extract %rr[ 1] : !quantum.reg -> !quantum.bit
    %ah = quantum.custom "Hadamard"() %a : !quantum.bit
    %m, %am = quantum.measure %ah : i1, !quantum.bit
    %ar = scf.if %m -> (!quantum.bit) {
      %ax = quantum.custom "PauliX"() %am : !quantum.bit
      scf.yield %ax : !quantum.bit
    } else {
      scf.yield %am : !quantum.bit
    }
    // data slot 0: gate only, never measured (CARRY)
    %d = quantum.extract %rr[ 0] : !quantum.reg -> !quantum.bit
    %dh = quantum.custom "Hadamard"() %d : !quantum.bit
    %r1 = quantum.insert %rr[ 0], %dh : !quantum.reg, !quantum.bit
    %r2 = quantum.insert %r1[ 1], %ar : !quantum.reg, !quantum.bit
    scf.yield %m, %r2 : i1, !quantum.reg
  }
  %d = quantum.extract %res#1[ 0] : !quantum.reg -> !quantum.bit
  %mz, %dm = quantum.measure %d : i1, !quantum.bit
  %rr = quantum.insert %res#1[ 0], %dm : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %mz : i1
}
