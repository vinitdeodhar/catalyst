// RUN: quantum-opt --purl="calib=%S/backend_leak_nt.json p=0.1 shots=200000 margin=1.0 force-knit=true" --purl-lower-qcut %s | FileCheck %s
//
// The carried wire is entangled and measured in the X basis (heralding), so the
// state is NOT provably known -> REFRESH is inapplicable. Under a heavy-tailed
// (low p) leakage-dominated calibration the tier-3 quasi-probability KNIT strategy
// (gamma = 4) applies. Since spec 13, the cost model selects MIGRATE for unknown
// states, so knit is the comparison arm reached via force-knit=true (spec 13.7/13.8);
// this flat calib has no coupling map, so migrate would be unavailable anyway.

// CHECK: func.func private @purl_sample_term() -> (i64, i1)
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit, i32, f64) -> (i1, !quantum.bit, i32, f64)
// CHECK: call @purl_sample_term
// CHECK: purl.known_state = "none"
// CHECK: purl.strategy = "knit"

func.func @rus_program() -> f64 {
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
    %a3 = quantum.custom "Hadamard"() %a2 : !quantum.bit
    %m, %a4 = quantum.measure %a3 : i1, !quantum.bit
    %dq2 = scf.if %m -> (!quantum.bit) {
      %dz = quantum.custom "PauliZ"() %dq1 : !quantum.bit
      scf.yield %dz : !quantum.bit
    } else {
      scf.yield %dq1 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %a4 : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq2 : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
