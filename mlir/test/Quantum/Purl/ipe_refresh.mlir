// RUN: quantum-opt --purl="calib=unit p=0.12 C=3" %s | FileCheck %s
//
// Adaptive Iterative Phase Estimation shape: the carried wire (0) holds the
// EIGENSTATE |psi> = |+> of U across a measurement-conditioned loop that extracts
// phase bits until confident. The controlled-U^k is net-identity on the eigenstate
// (phase kickback onto the ancilla), so wire 0 is a held CARRY; wire 1 is the IPE
// ancilla. --purl proves the held eigenstate (known_state = identity, prep = a
// single Hadamard for |+>) and emits a REFRESH purl.qcut on the register slot.

// CHECK: scf.while ({{.*}}) : (tensor<i1>, !quantum.reg, i32) -> (!quantum.reg, i32)
// CHECK: scf.if
// CHECK: quantum.extract %{{[0-9]+}}[ 0]
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy refresh>
// the prep region re-prepares |+> = H|0>
// CHECK: quantum.custom "Hadamard"
// CHECK: purl.yield
// CHECK: quantum.insert %{{[0-9]+}}[ 0]
// CHECK: purl.known_state = "identity"
// CHECK-SAME: purl.strategy = "refresh"

func.func public @ipe() -> tensor<f64> {
  %c = stablehlo.constant dense<true> : tensor<i1>
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %plus = quantum.custom "Hadamard"() %1 : !quantum.bit          // held eigenstate |+>
  %2 = quantum.insert %0[ 0], %plus : !quantum.reg, !quantum.bit
  %3 = scf.while (%arg0 = %c, %arg1 = %2) : (tensor<i1>, !quantum.reg) -> !quantum.reg {
    %e = tensor.extract %arg0[] : tensor<i1>
    scf.condition(%e) %arg1 : !quantum.reg
  } do {
  ^bb0(%arg0: !quantum.reg):
    %a0 = quantum.extract %arg0[ 1] : !quantum.reg -> !quantum.bit  // IPE ancilla
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit               // phase bit
    %fe = tensor.from_elements %m : tensor<i1>
    %ins = quantum.insert %arg0[ 1], %a2 : !quantum.reg, !quantum.bit
    scf.yield %fe, %ins : tensor<i1>, !quantum.reg
  }
  %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
  %5 = quantum.namedobs %4[ PauliX] : !quantum.obs                 // eigenstate axis
  %6 = quantum.expval %5 : f64
  %fe2 = tensor.from_elements %6 : tensor<f64>
  %7 = quantum.insert %3[ 0], %4 : !quantum.reg, !quantum.bit
  quantum.dealloc %7 : !quantum.reg
  return %fe2 : tensor<f64>
}
