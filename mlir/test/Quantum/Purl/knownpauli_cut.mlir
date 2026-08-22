// RUN: quantum-opt --purl="calib=unit p=0.625 C=3" %s | FileCheck %s
//
// The body applies a fixed PauliX to the carried wire every iteration, so its
// failure action is a KNOWN Pauli (X). The state at a cut is X^counter|psi0>,
// which the pass reconstructs deterministically (re-prep |psi0> + counter-parity
// X), and it corrects the final Z-readout by the trip-count parity (X
// anticommutes with Z). Still gamma = 1: no @purl_sample_term, no f64 weight.

// CHECK-NOT: func.func private @purl_sample_term
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// deterministic cut with a counter-parity PauliX re-prep correction
// CHECK: quantum.measure
// CHECK: quantum.custom "PauliX"
// the loop carries the deterministic-cut tags
// CHECK: purl.cut = "deterministic"
// CHECK: purl.known_state = "pauli"
// output: expval INTACT (refresh, not legalized), corrected by trip-count parity
// CHECK: arith.andi
// CHECK: quantum.namedobs
// CHECK: quantum.expval
// CHECK: return {{.*}} : f64

func.func @knownpauli() -> f64 {
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
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit
    // fixed Pauli byproduct on the carried wire every attempt
    %dqx = quantum.custom "PauliX"() %dq : !quantum.bit
    %areg1 = quantum.insert %areg[ 0], %a2 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dqx : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
