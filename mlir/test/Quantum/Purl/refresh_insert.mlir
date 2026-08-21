// RUN: quantum-opt --purl="calib=unit p=0.625 C=3" %s | FileCheck %s
//
// Held-memory carry loop: the carried wire holds |psi0> = H T H T H |0> while a
// coin ancilla retries; the body NEVER touches the carried wire, so its failure
// action is provably the identity. --purl proves it (purl.known_state="identity")
// and emits a REFRESH purl.qcut inside the periodic cut guard -- NOT an inline
// expansion. The carry gains an i32 counter only (no f64 weight), there is no
// sample fn, and the quantum.expval output survives.

// CHECK-NOT: @purl_sample_term
// carry is (i1, !quantum.bit, i32) -- counter only, no f64 weight
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// CHECK: scf.if
// the abstract cut op, refresh strategy, with its |psi0> prep region (H T H T H)
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy refresh>
// CHECK: quantum.custom "Hadamard"
// CHECK: quantum.custom "T"
// CHECK: purl.yield
// while-op attributes print alphabetically: known_state before strategy
// CHECK: purl.known_state = "identity"
// CHECK: purl.strategy = "refresh"
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
