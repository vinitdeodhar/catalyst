// rus_data (spec 6.5): the knit rewrite on the Paetznick-Svore V3 RUS body. At
// p=5/8 on a unit-coherence calibration the window is non-empty (C_min=3), so the
// KNIT arm fires: the carry gains an i32 cut counter + f64 quasi-probability weight,
// a purl.qcut (strategy knit) every C failing iterations, and the expval is legalized
// to the weighted sample. The carried data state is UNKNOWN (non-Clifford), so knit
// -- unbiased for ANY state -- is the only valid cut; refresh never appears.
// RUN: quantum-opt --purl="calib=unit p=0.625" %s | FileCheck %s

// deterministic-cut sample-term function is NOT emitted (that is refresh); this is knit
// CHECK-NOT: strategy = #purl<strategy refresh>
// CHECK-LABEL: func.func @rus_data
// held non-Clifford data prep survives before the loop
// CHECK: quantum.custom "RZ"
// carry gains the i32 counter + f64 weight
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// the data-controlled non-Clifford gadget survives in the body
// CHECK: quantum.custom "RZ"({{.*}}) %{{.*}} ctrls
// CHECK: quantum.measure
// every C failing iterations: a quasi-probability knit cut threading the weight
// CHECK: scf.if
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy knit>
// unknown carried state -> knit, quasiprobability cut
// CHECK: purl.cut = "quasiprobability"
// CHECK-SAME: purl.known_state = "none"
// CHECK-SAME: purl.strategy = "knit"
// expval legalized to the weighted sample
// CHECK: arith.mulf

func.func @rus_data() -> f64 {
  %true = arith.constant true
  %ry = arith.constant 0.4 : f64
  %rz = arith.constant 0.7 : f64
  %theta = arith.constant 1.1071487177940904 : f64   // atan(2): V3 axial angle
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  %pa = quantum.custom "RY"(%ry) %draw : !quantum.bit
  %d0 = quantum.custom "RZ"(%rz) %pa : !quantum.bit
  %res:2 = scf.while (%cont = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%cont) %cont, %d : i1, !quantum.bit
  } do {
  ^bb0(%c: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %a2, %dq1 = quantum.custom "RZ"(%theta) %a1 ctrls(%dq) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %m, %a3 = quantum.measure %a2 : i1, !quantum.bit
    %areg1 = quantum.insert %areg[ 0], %a3 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq1 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
