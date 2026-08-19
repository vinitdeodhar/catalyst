// RUN: quantum-opt --loop-knit="calib=%S/backend_leaky.json p=0.625 shots=20000" %s | FileCheck %s
//
// Profitability model (3.5) on a failure-identity RUS (held magic state) under a
// leaky calibration: the tier-1 REFRESH strategy wins. Rewrite = reset +
// re-prepare |psi0>, NO weight in the carry, and the expval SURVIVES (refresh
// delivers a genuine quantum state, so no output legalization).

// CHECK-NOT: func.func private @knit_sample_term
// carry extended with a counter only (no f64 weight)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// cut = measure + re-prepare the known |psi0> = H T H T H |0>
// CHECK: quantum.measure
// CHECK: quantum.custom "Hadamard"
// CHECK: quantum.custom "T"
// CHECK: knit.strategy = "refresh"
// output is NOT legalized: the expval survives
// CHECK: quantum.namedobs
// CHECK: quantum.expval
// CHECK: return {{.*}} : f64

func.func @held_memory() -> f64 {
  %true = arith.constant true
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  %p1 = quantum.custom "Hadamard"() %draw : !quantum.bit
  %p2 = quantum.custom "T"() %p1 : !quantum.bit
  %p3 = quantum.custom "Hadamard"() %p2 : !quantum.bit
  %p4 = quantum.custom "T"() %p3 : !quantum.bit
  %d0 = quantum.custom "Hadamard"() %p4 : !quantum.bit
  %res:2 = scf.while (%fail = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%fail) %fail, %d : i1, !quantum.bit
  } do {
  ^bb0(%f: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit
    %ar = scf.if %m -> (!quantum.bit) {
      %ax = quantum.custom "PauliX"() %a2 : !quantum.bit
      scf.yield %ax : !quantum.bit
    } else {
      scf.yield %a2 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %ar : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
