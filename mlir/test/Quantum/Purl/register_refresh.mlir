// RUN: quantum-opt --purl="calib=unit p=0.625 C=3" %s | FileCheck %s
//
// Real Catalyst-emitted IR shape (spec 3.1 / 9.1): the carried qubit is threaded
// through a !quantum.reg carry, not a bare !quantum.bit. Wire 0 holds |psi0> =
// H T H T H |0> and is NEVER extracted in the body (held identity); wire 1 is the
// coin. --purl anchors on the observable (post-loop extract[0] -> namedobs),
// classifies the held register slot as a CARRY, proves identity, and emits a
// REFRESH purl.qcut by extracting the held wire from the register, cutting it, and
// re-inserting -- all while threading the !quantum.reg carry.

// carry extended with an i32 counter, register still threaded
// CHECK: scf.while ({{.*}}) : (tensor<i1>, !quantum.reg, i32) -> (!quantum.reg, i32)
// the register cut: extract held wire -> qcut{refresh} -> re-insert
// CHECK: scf.if
// CHECK: quantum.extract %{{[0-9]+}}[ 0]
// CHECK: purl.qcut
// CHECK-SAME: strategy = #purl<strategy refresh>
// CHECK: quantum.custom "Hadamard"
// CHECK: quantum.custom "T"
// CHECK: purl.yield
// CHECK: quantum.insert %{{[0-9]+}}[ 0]
// analysis attributes land on the rewritten loop (printed alphabetically)
// CHECK: purl.known_state = "identity"
// CHECK-SAME: purl.strategy = "refresh"

func.func public @rus_0() -> tensor<f64> {
  %c = stablehlo.constant dense<true> : tensor<i1>
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
  %out_qubits_0 = quantum.custom "T"() %out_qubits : !quantum.bit
  %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits_0 : !quantum.bit
  %out_qubits_2 = quantum.custom "T"() %out_qubits_1 : !quantum.bit
  %out_qubits_3 = quantum.custom "Hadamard"() %out_qubits_2 : !quantum.bit
  %2 = quantum.insert %0[ 0], %out_qubits_3 : !quantum.reg, !quantum.bit
  %3 = scf.while (%arg0 = %c, %arg1 = %2) : (tensor<i1>, !quantum.reg) -> !quantum.reg {
    %extracted = tensor.extract %arg0[] : tensor<i1>
    scf.condition(%extracted) %arg1 : !quantum.reg
  } do {
  ^bb0(%arg0: !quantum.reg):
    %8 = quantum.extract %arg0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %8 : !quantum.bit
    %mres, %out_qubit = quantum.measure %out_qubits_4 : i1, !quantum.bit
    %from_elements_5 = tensor.from_elements %mres : tensor<i1>
    %9 = quantum.insert %arg0[ 1], %out_qubit : !quantum.reg, !quantum.bit
    scf.yield %from_elements_5, %9 : tensor<i1>, !quantum.reg
  }
  %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
  %5 = quantum.namedobs %4[ PauliZ] : !quantum.obs
  %6 = quantum.expval %5 : f64
  %from_elements = tensor.from_elements %6 : tensor<f64>
  %7 = quantum.insert %3[ 0], %4 : !quantum.reg, !quantum.bit
  quantum.dealloc %7 : !quantum.reg
  return %from_elements : tensor<f64>
}
