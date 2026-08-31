// rus_data (spec 6.5): Paetznick-Svore RUS synthesis of the non-Clifford V3 gate
// applied to arbitrary program DATA. The carried wire `d` is prepared through
// non-Clifford gates (Ry, Rz), and each iteration runs a data-controlled non-Clifford
// rotation on the ancilla (the V3 RUS gadget). The non-Clifford preparation must
// defeat the tableau, so the known-state proof returns UNKNOWN -- refresh is unsound;
// knit is the only valid cut. (Best-effort circuit reconstruction, spec 6.5 opt 2.)
//
// Analysis only: classify carry + prove unknown + emit no refresh.
// RUN: quantum-opt --purl="calib=unit p=0.625 analyze-only=true" %s | FileCheck %s

// CHECK: purl.class = "carry"
// CHECK: purl.known_state = "none"
// never a refresh: a non-Clifford data state cannot be certified / re-prepared
// CHECK-NOT: strategy = #purl<strategy refresh>
// CHECK-NOT: purl.qcut

func.func @rus_data() -> f64 {
  %true = arith.constant true
  %ry = arith.constant 0.4 : f64
  %rz = arith.constant 0.7 : f64
  %theta = arith.constant 1.1071487177940904 : f64   // atan(2): V3 axial angle
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  // |psi> = Rz(0.7) Ry(0.4) |0> -- a generic NON-stabilizer data state
  %pa = quantum.custom "RY"(%ry) %draw : !quantum.bit
  %d0 = quantum.custom "RZ"(%rz) %pa : !quantum.bit
  %res:2 = scf.while (%cont = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%cont) %cont, %d : i1, !quantum.bit
  } do {
  ^bb0(%c: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    // data-controlled non-Clifford rotation on the ancilla (V3 RUS gadget stand-in)
    %a2, %dq1 = quantum.custom "RZ"(%theta) %a1 ctrls(%dq) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %m, %a3 = quantum.measure %a2 : i1, !quantum.bit       // 0 = success, 1 = failure
    %areg1 = quantum.insert %areg[ 0], %a3 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq1 : i1, !quantum.bit                  // continue while failure (m=1)
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
