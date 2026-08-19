// RUN: quantum-opt --loop-knit="calib=unit p=0.625 C=4" %s | FileCheck %s
//
// Full rewrite (Part 3.4) of the canonical carry-type RUS program (bare-qubit
// carry, expval output -- the writeup Section 2 input; its output matches
// rus_marked_suspend.mlir up to SSA names). Fires with C=4 for p=5/8.

// the external sampling hook is declared at module scope
// CHECK: func.func private @knit_sample_term() -> (i64, i1)
// the extended 4-element carry (i1 fail, !quantum.bit, i32 counter, f64 weight)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// counter + periodic-cut guard
// CHECK: arith.addi
// CHECK: arith.remsi
// CHECK: arith.cmpi eq
// CHECK: arith.andi
// the guarded inline cut: sample, basis change, measure, reset/prep, weight
// CHECK: scf.if
// CHECK: call @knit_sample_term
// CHECK: quantum.measure
// CHECK: arith.select
// CHECK: arith.mulf
// the loop carries the knit.applied tag
// CHECK: knit.applied
// legalized output: weighted Z sample, still returns f64
// CHECK: quantum.measure
// CHECK: arith.mulf
// CHECK: return {{.*}} : f64

func.func @rus_program() -> f64 {
  %true = arith.constant true
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  %d0 = quantum.custom "Hadamard"() %draw : !quantum.bit
  %res:2 = scf.while (%fail = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%fail) %fail, %d : i1, !quantum.bit
  } do {
  ^bb0(%f: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %a2, %dq1 = quantum.custom "CNOT"() %a1, %dq : !quantum.bit, !quantum.bit
    %a3 = quantum.custom "Hadamard"() %a2 : !quantum.bit
    %m, %a4 = quantum.measure %a3 : i1, !quantum.bit
    %dq2 = scf.if %m -> (!quantum.bit) {
      %dz = quantum.custom "PauliZ"() %dq1 : !quantum.bit
      scf.yield %dz : !quantum.bit
    } else {
      scf.yield %dq1 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %a4 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq2 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
