// RUN: quantum-opt --loop-knit="calib=unit p=0.625 C=3" %s | FileCheck %s
//
// Held-memory carry loop: the carried wire holds |psi0> = H T H T H |0> while a
// coin ancilla retries; the body NEVER touches the carried wire, so its failure
// action is provably the identity. The pass proves it (knit.known_state =
// "identity") and lowers to the cheap deterministic gamma=1 cut: measure +
// re-prepare |psi0>, with NO @knit_sample_term, NO f64 weight in the carry.

// CHECK-NOT: func.func private @knit_sample_term
// carry is (i1, !quantum.bit, i32) -- counter only, no f64 weight
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// deterministic cut: measure then re-prepare |psi0> (H T H T H)
// CHECK: scf.if
// CHECK: quantum.measure
// CHECK: quantum.custom "Hadamard"
// CHECK: quantum.custom "T"
// CHECK-NOT: call @knit_sample_term
// CHECK: knit.cut = "deterministic"
// CHECK: knit.known_state = "identity"
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
    // coin ancilla only; the carried wire dq is untouched (held)
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
