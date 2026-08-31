// qwalk (spec 13 migrate benchmark): a FAT-TAILED, 2q-heavy, non-Clifford carry.
// The held data d is coupled each step by the net-identity sandwich CNOT(a,d), T(a),
// CNOT(a,d) x3 (6 two-qubit gates on d). The CNOT pair cancels on d, but the middle
// T is NON-Clifford, so the known-state proof returns UNKNOWN -> refresh unsound. On
// a coupling-map calib with a cheap partner edge, the cost model selects MIGRATE
// (gamma=1, no variance floor) and rewrites with a partner qubit + three-CNOT SWAP.
// RUN: quantum-opt --purl="calib=%S/migrate_calib.json p=0.5 shots=200000 carry-qubit=0" %s | FileCheck %s

// CHECK-LABEL: func.func @qwalk
// held non-Clifford reference prep survives before the loop
// CHECK: quantum.custom "RZ"
// carry gains an i32 counter + a partner qubit (migrate: unweighted, no f64 weight)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, !quantum.bit) -> (i1, !quantum.bit, i32, !quantum.bit)
// the net-identity non-Clifford coupling survives in the body
// CHECK: quantum.custom "T"
// the migrate SWAP (three CNOTs on the pair) + reset of the abandoned qubit
// CHECK: scf.if
// CHECK: quantum.custom "CNOT"
// CHECK: quantum.custom "PauliX"
// decision-audit attributes: migrate / swap / the ping-pong pair; never refresh/knit
// CHECK: purl.cut = "swap"
// CHECK-SAME: purl.known_state = "none"
// CHECK-SAME: purl.pair = array<i64: 0, 1>
// CHECK-SAME: purl.strategy = "migrate"
// CHECK-NOT: arith.mulf

func.func @qwalk() -> f64 {
  %true = arith.constant true
  %ry = arith.constant 0.4 : f64
  %rz = arith.constant 0.7 : f64
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
    // block 1: CNOT(a,d), T(a), CNOT(a,d)  (net identity on d, non-Clifford)
    %a2, %d1 = quantum.custom "CNOT"() %a1, %dq : !quantum.bit, !quantum.bit
    %a3 = quantum.custom "T"() %a2 : !quantum.bit
    %a4, %d2 = quantum.custom "CNOT"() %a3, %d1 : !quantum.bit, !quantum.bit
    // block 2
    %a5, %d3 = quantum.custom "CNOT"() %a4, %d2 : !quantum.bit, !quantum.bit
    %a6 = quantum.custom "T"() %a5 : !quantum.bit
    %a7, %d4 = quantum.custom "CNOT"() %a6, %d3 : !quantum.bit, !quantum.bit
    // block 3
    %a8, %d5 = quantum.custom "CNOT"() %a7, %d4 : !quantum.bit, !quantum.bit
    %a9 = quantum.custom "T"() %a8 : !quantum.bit
    %a10, %d6 = quantum.custom "CNOT"() %a9, %d5 : !quantum.bit, !quantum.bit
    %m, %a11 = quantum.measure %a10 : i1, !quantum.bit
    %areg1 = quantum.insert %areg[ 0], %a11 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %d6 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
