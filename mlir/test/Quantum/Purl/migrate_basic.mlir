// Migrate strategy (spec 13): an unknown-state carry loop, on a coupling-map calib
// where a cheap pair edge makes swap-based carrier replacement cost-positive, receives
// the MIGRATE rewrite: a partner qubit threaded through the carry, a three-CNOT SWAP
// in the cut guard, a reset of the swapped-out (abandoned) qubit, and the partner
// register cleaned up after the loop. gamma=1: no weight, expval intact.
// RUN: quantum-opt --purl="calib=%S/migrate_calib.json p=0.45 shots=200000 carry-qubit=0" %s | FileCheck %s

// a fresh partner qubit is allocated before the loop
// CHECK: quantum.alloc
// CHECK: quantum.alloc
// the carry gains an i32 counter AND a partner qubit (no f64 weight -> unweighted)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, !quantum.bit) -> (i1, !quantum.bit, i32, !quantum.bit)
// the migrate guard fires the SWAP: three CNOTs on the pair
// CHECK: scf.if
// CHECK: quantum.custom "CNOT"
// CHECK: quantum.custom "CNOT"
// CHECK: quantum.custom "CNOT"
// the swapped-out (abandoned) qubit is reset (measure + conditional PauliX)
// CHECK: quantum.measure
// CHECK: quantum.custom "PauliX"
// decision-audit attributes: migrate / swap / the ping-pong pair
// CHECK: purl.cut = "swap"
// CHECK-SAME: purl.pair = array<i64: 0, 1>
// CHECK-SAME: purl.strategy = "migrate"
// NO quasi-probability weight fold and NO refresh sample-term function
// CHECK-NOT: arith.mulf
// CHECK-NOT: strategy = #purl<strategy knit>
// the partner register is cleaned up after the loop
// CHECK: quantum.dealloc

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
