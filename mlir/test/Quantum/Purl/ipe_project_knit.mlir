// RUN: quantum-opt --purl="calib=unit p=0.45" %s | FileCheck %s
//
// ipe_project, fast config (spec 6.3). Same controlled-RZ projection body as the
// faithful config, but the coarser stop (higher p) and a generous coherence budget
// open a non-empty knit window. The carried state is still UNKNOWN (refresh remains
// unsound), yet KNIT is valid -- its gamma=4 identity decomposition is unbiased for
// ANY carried state -- so the pass emits the knit rewrite: an i32 cut counter and an
// f64 quasi-probability WEIGHT threaded through the carry, a `purl.qcut` (strategy
// knit) every C rounds, and the expval legalized to the weighted sample.

// CHECK-LABEL: func.func @ipe_project
// CHECK: quantum.custom "RY"
// the carry gains an i32 cut counter AND an f64 quasi-probability weight
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// the non-Clifford controlled-RZ projection body survives
// CHECK: quantum.custom "RZ"({{.*}}) %{{.*}} ctrls
// CHECK: quantum.measure
// every C rounds: a quasi-probability knit cut threading the weight (NOT a refresh)
// CHECK: scf.if
// CHECK: purl.qcut
// CHECK-SAME: axis = #purl<pauli Z>
// CHECK-SAME: strategy = #purl<strategy knit>
// CHECK-NOT: strategy = #purl<strategy refresh>
// unknown carried state -> knit, quasiprobability cut
// CHECK: purl.cut = "quasiprobability"
// CHECK-SAME: purl.known_state = "none"
// CHECK-SAME: purl.strategy = "knit"
// expval legalized to the weighted sample: weight * (+1/-1)
// CHECK: arith.mulf

func.func @ipe_project() -> f64 {
  %true = arith.constant true
  %theta = arith.constant 0.8975979010256552 : f64      // 2*pi/7
  %alpha2 = arith.constant 0.7853981633974483 : f64     // 2*alpha = pi/4
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  // input superposition cos(alpha)|0> + sin(alpha)|1> = Ry(2 alpha)|0> (non-eigenstate)
  %d0 = quantum.custom "RY"(%alpha2) %draw : !quantum.bit
  %res:2 = scf.while (%cont = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%cont) %cont, %d : i1, !quantum.bit
  } do {
  ^bb0(%c: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    // controlled-U = controlled-RZ(theta): NON-Clifford on the carried wire d
    %dq1, %a2 = quantum.custom "RZ"(%theta) %dq ctrls(%a1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %a3 = quantum.custom "S"() %a2 : !quantum.bit         // feedback
    %a4 = quantum.custom "Hadamard"() %a3 : !quantum.bit
    %m, %a5 = quantum.measure %a4 : i1, !quantum.bit      // phase bit / herald
    %areg1 = quantum.insert %areg[ 0], %a5 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq1 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
