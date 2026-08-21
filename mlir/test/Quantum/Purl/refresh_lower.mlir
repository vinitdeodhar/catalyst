// RUN: quantum-opt --purl="calib=unit p=0.625 C=3" --purl-lower-qcut %s | FileCheck %s
//
// Two-phase pipeline: --purl inserts a REFRESH purl.qcut, then --purl-lower-qcut
// mechanically expands it. After lowering there is NO purl.qcut left; the cut has
// become measure (end segment) + conditional-X reset + the inlined |psi0> prep
// (H T H T H). No sample fn (refresh is weight-free). expval survives.

// CHECK-NOT: purl.qcut
// CHECK-NOT: @purl_sample_term
// CHECK: scf.if
// expanded: measure, reset, re-prepare |psi0>
// CHECK: quantum.measure
// CHECK: quantum.custom "PauliX"
// CHECK: quantum.custom "Hadamard"
// CHECK: quantum.custom "T"
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
