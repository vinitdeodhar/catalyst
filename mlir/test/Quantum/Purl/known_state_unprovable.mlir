// RUN: quantum-opt --purl="calib=unit p=0.625 C=4" --purl-lower-qcut %s | FileCheck %s
//
// The body applies a Hadamard to the carried wire every iteration. H is Clifford
// but NOT a Pauli (it maps X->Z), so the carried state is not a known fixed
// state: the pass cannot prove it and must fall back to the general
// quasi-probability gamma = 4 cut (with @purl_sample_term and an f64 weight).

// CHECK: func.func private @purl_sample_term
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// CHECK: call @purl_sample_term
// CHECK: purl.cut = "quasiprobability"
// CHECK: purl.known_state = "none"

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
    // Clifford-but-not-Pauli action on the carried wire -> unprovable
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
