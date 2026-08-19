// RUN: quantum-opt --loop-knit="calib=unit p=0.625 C=3" %s | FileCheck %s
//
// RUS-synthesis shape: the carried wire is ENTANGLED with a fresh ancilla
// (CNOT), the ancilla is measured, and the measurement-induced byproduct is
// undone by an outcome-conditioned PauliX. The stabilizer/Pauli-frame prover
// tracks this through the entangling gate + measurement + correction and proves
// the net failure action is the identity -> deterministic gamma=1 cut.

// CHECK-NOT: func.func private @knit_sample_term
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// CHECK: knit.cut = "deterministic"
// CHECK: knit.known_state = "identity"
// CHECK: return {{.*}} : f64

func.func @entangling_coin() -> f64 {
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
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit          // ancilla |+>
    %a2, %dq1 = quantum.custom "CNOT"() %a1, %dq : !quantum.bit, !quantum.bit  // entangle
    %m, %a3 = quantum.measure %a2 : i1, !quantum.bit              // measure coin (Z)
    %dq2 = scf.if %m -> (!quantum.bit) {                          // undo byproduct
      %dx = quantum.custom "PauliX"() %dq1 : !quantum.bit
      scf.yield %dx : !quantum.bit
    } else {
      scf.yield %dq1 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %a3 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq2 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
