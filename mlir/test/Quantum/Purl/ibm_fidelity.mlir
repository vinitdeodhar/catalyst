// RUN: quantum-opt --purl="calib=%S/ibm_eagle_r3.json p=0.1 shots=200000 p-leak=1e-3 carry-qubit=0 depth=10 analyze-only=true" %s | FileCheck %s
//
// Real IBM Eagle r3 per-qubit hardware data (T1/T2/gate/readout for the carried
// qubit + median 2q error) drives a TIME-BASED fidelity prediction. The carried
// wire receives a CNOT each iteration and idles through the body; at the mean
// (unbounded) trip count of a low-p heralded loop it decoheres substantially,
// while a refresh cut bounds the held depth. Leakage is the separate p-leak knob
// (published IBM estimate), not part of the device dataset.

// body depth is a real duration (seconds), computed from IBM gate/readout times
// CHECK: purl.body_seconds
// per-depth prediction requested via depth=10
// CHECK: purl.fidelity_at_depth
// mean delivered fidelity at the bounded vs unbounded depth
// CHECK: purl.predicted_fidelity = {bounded = {{[0-9.eE+-]+}} : f64, unbounded = {{[0-9.eE+-]+}} : f64}
// the entangling coin is proven identity -> refresh (gamma = 1) is selected
// CHECK: purl.strategy = "refresh"

func.func @rus_ibm() -> f64 {
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
    %a2, %dq1 = quantum.custom "CNOT"() %a1, %dq : !quantum.bit, !quantum.bit
    %m, %a3 = quantum.measure %a2 : i1, !quantum.bit
    %dq2 = scf.if %m -> (!quantum.bit) {
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
