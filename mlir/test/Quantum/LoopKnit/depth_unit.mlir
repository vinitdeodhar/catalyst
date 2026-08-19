// RUN: quantum-opt --loop-knit="analyze-only=true calib=unit" %s | FileCheck %s
//
// Hand-computable body depth (unit weights: every gate = 1, readout = 1,
// tau = 1 for the measurement-conditioned scf.if). Carried qubit q:
//   ancilla path:  H (d=1) -> measure (d=2)  [supplies the loop condition]
//   carried path:  H (d=1) -> scf.if{ T (d=2) } -> +tau -> d=3
// B = max yielded-qubit depth = 3.

// CHECK-DAG: knit.class = "carry"
// CHECK-DAG: knit.carry_slot = 0 : i64
// CHECK-DAG: knit.body_layers = 3.000000e+00

func.func @depth_body() -> i1 {
  %true = arith.constant true
  %dreg = quantum.alloc( 1) : !quantum.reg
  %d0 = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  %res:2 = scf.while (%f = %true, %q = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%f) %f, %q : i1, !quantum.bit
  } do {
  ^bb0(%fl: i1, %qq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit
    %q1 = quantum.custom "Hadamard"() %qq : !quantum.bit
    %q2 = scf.if %m -> (!quantum.bit) {
      %t = quantum.custom "T"() %q1 : !quantum.bit
      scf.yield %t : !quantum.bit
    } else {
      scf.yield %q1 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %a2 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %q2 : i1, !quantum.bit
  }
  %rr = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
