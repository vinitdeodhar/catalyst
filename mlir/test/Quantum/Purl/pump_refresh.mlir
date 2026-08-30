// RUN: quantum-opt --purl="calib=unit p=0.1 C=2" %s | FileCheck %s
//
// pump -- entanglement-pumping proxy (spec 6.1). The carried wire d holds the
// magic state |psi0> = H T H T H |0> across a low-p heralded loop. Each iteration
// runs the entangling sandwich CNOT(a->d), S(a), CNOT(a->d) THREE times (6 two-
// qubit gates on d) against a fresh ancilla a, then measures a (the herald). Both
// CNOTs share the control a and S touches only a, so the two flips on d cancel:
// the net action on d is the IDENTITY (no outcome-conditioned correction needed).
// The stabilizer prover certifies this -> deterministic gamma=1 REFRESH cut, and
// the 6 two-qubit gates make per-2q leakage (spec 4.1) the dominant held error.

// deterministic (gamma=1) refresh cut -> NO quasiprobability sample-term function
// CHECK-NOT: func.func private @purl_sample_term
// CHECK-LABEL: func.func @pump
// held |psi0> = H T H T H |0> prepared before the loop
// CHECK: quantum.custom "T"
// the carry is extended with an i32 cut counter
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32) -> (i1, !quantum.bit, i32)
// the CNOT-S-CNOT sandwich (6 two-qubit gates on d) survives in the body
// CHECK: quantum.custom "CNOT"
// CHECK: quantum.custom "S"
// CHECK: quantum.measure
// every C failing iterations: a deterministic refresh cut re-preparing the known state
// CHECK: scf.if
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy refresh>
// CHECK: quantum.custom "T"
// CHECK: purl.yield
// proven identity -> deterministic gamma=1 cut, refresh strategy
// CHECK: purl.cut = "deterministic"
// CHECK-SAME: purl.known_state = "identity"
// CHECK-SAME: purl.strategy = "refresh"

func.func @pump() -> f64 {
  %true = arith.constant true
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  // held input |psi0> = H T H T H |0>
  %p0 = quantum.custom "Hadamard"() %draw : !quantum.bit
  %p1 = quantum.custom "T"() %p0 : !quantum.bit
  %p2 = quantum.custom "Hadamard"() %p1 : !quantum.bit
  %p3 = quantum.custom "T"() %p2 : !quantum.bit
  %d0 = quantum.custom "Hadamard"() %p3 : !quantum.bit
  %res:2 = scf.while (%fail = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%fail) %fail, %d : i1, !quantum.bit
  } do {
  ^bb0(%f: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit               // coin prep
    // block 1: CNOT(a->d), S(a), CNOT(a->d)
    %a2, %d1 = quantum.custom "CNOT"() %a1, %dq : !quantum.bit, !quantum.bit
    %a3 = quantum.custom "S"() %a2 : !quantum.bit
    %a4, %d2 = quantum.custom "CNOT"() %a3, %d1 : !quantum.bit, !quantum.bit
    // block 2
    %a5, %d3 = quantum.custom "CNOT"() %a4, %d2 : !quantum.bit, !quantum.bit
    %a6 = quantum.custom "S"() %a5 : !quantum.bit
    %a7, %d4 = quantum.custom "CNOT"() %a6, %d3 : !quantum.bit, !quantum.bit
    // block 3
    %a8, %d5 = quantum.custom "CNOT"() %a7, %d4 : !quantum.bit, !quantum.bit
    %a9 = quantum.custom "S"() %a8 : !quantum.bit
    %a10, %d6 = quantum.custom "CNOT"() %a9, %d5 : !quantum.bit, !quantum.bit
    %m, %a11 = quantum.measure %a10 : i1, !quantum.bit                 // herald
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
