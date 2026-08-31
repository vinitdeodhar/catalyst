// Migrate partner selection (spec 13.1): among edges incident to the carried qubit,
// the pass fixes the partner as the edge minimizing 3*(gate_2q_err + leak_2q). Here
// qubit 0 is incident to edge (0,1) [cheap leak 1e-4] and edge (0,2) [expensive leak
// 2e-2]; the cheaper edge must win, so the pair is [0, 1].
// RUN: quantum-opt --purl="calib=%S/migrate_calib.json p=0.45 shots=200000 carry-qubit=0 analyze-only=true" %s | FileCheck %s

// CHECK: purl.pair = array<i64: 0, 1>
// CHECK-SAME: purl.strategy = "migrate"

func.func @rus_data() -> f64 {
  %true = arith.constant true
  %ry = arith.constant 0.4 : f64
  %rz = arith.constant 0.7 : f64
  %theta = arith.constant 1.1071487177940904 : f64
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
