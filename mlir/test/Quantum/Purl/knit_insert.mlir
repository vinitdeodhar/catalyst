// RUN: quantum-opt --purl="calib=unit p=0.625 C=4" %s | FileCheck %s
//
// KNIT insertion (spec 3.7). The carried wire takes a Clifford-but-not-Pauli
// action (Hadamard) each iteration, so the known-state proof fails and --purl
// falls back to the gamma=4 quasi cut. It emits an ABSTRACT purl.qcut with
// strategy=knit threading an f64 weight -- NOT the inline expansion, and with
// NO @purl_sample_term (that is the lowering's job, --purl-lower-qcut). The carry
// gains both an i32 counter and an f64 weight, and the quantum.expval output is
// legalized to a weighted Z sample.

// CHECK-NOT: @purl_sample_term
// carry: counter (i32) + quasi-probability weight (f64)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// CHECK: scf.if
// the abstract knit cut op, threading the f64 weight in -> out
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy knit>
// CHECK: purl.yield
// CHECK: (!quantum.bit, f64) -> (!quantum.bit, f64)
// while-op attributes (alphabetical): cut, known_state, strategy
// CHECK: purl.cut = "quasiprobability"
// CHECK: purl.known_state = "none"
// CHECK: purl.strategy = "knit"
// then (post-loop) expval legalized to a weighted Z sample: weight * (-1)^z
// CHECK: arith.select
// CHECK: arith.mulf
// CHECK: return {{.*}} : f64

func.func @unprovable() -> f64 {
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
    // Clifford-but-not-Pauli action on the carried wire -> unprovable -> knit
    %dqh = quantum.custom "Hadamard"() %dq : !quantum.bit
    %areg1 = quantum.insert %areg[ 0], %a2 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dqh : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
