// Classification + provability (unit calib, analyze-only: attributes, no rewrite):
// RUN: quantum-opt --purl="calib=unit p=0.12 analyze-only=true" %s | FileCheck %s
// Inadmissibility on the deployed calibration: the knit window is empty at this p
// (refresh is impossible on an unknown state), a compile-time diagnostic:
// RUN: quantum-opt --purl="calib=%S/ibm_eagle_r3.json p=0.12 carry-qubit=0" --verify-diagnostics %s
//
// ipe_project, faithful config (spec 6.3). The carried wire d starts in a
// SUPERPOSITION of the eigenstates of U = Rz(theta) and each round applies a
// controlled-U (controlled-RZ, NON-Clifford) whose ancilla measurement partially
// projects d. The carried state is outcome-dependent, so the known-state proof
// must return UNKNOWN -- there is no fixed state to re-prepare, and refresh is
// unsound. The pass classifies + proves (attributes below) but never refreshes;
// on the IBM calibration the knit window is also empty, reported as a diagnostic.

// CHECK: purl.class = "carry"
// CHECK: purl.known_state = "none"
// never a refresh: the carried state is not provably fixed
// CHECK-NOT: strategy = #purl<strategy refresh>
// CHECK-NOT: purl.qcut

func.func @ipe_project() -> f64 {
  %true = arith.constant true
  %theta = arith.constant 0.8975979010256552 : f64      // 2*pi/7
  %alpha2 = arith.constant 0.7853981633974483 : f64     // 2*alpha = pi/4
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  // input superposition cos(alpha)|0> + sin(alpha)|1> = Ry(2 alpha)|0> (non-eigenstate)
  %d0 = quantum.custom "RY"(%alpha2) %draw : !quantum.bit
  // expected-error@+1 {{empty cut-period window}}
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
