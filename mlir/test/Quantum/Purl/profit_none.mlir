// RUN: quantum-opt --purl="calib=%S/backend_clean.json p=0.625 shots=20000 margin=2.0" --verify-diagnostics %s | FileCheck %s
//
// Profitability model (3.5) under a CLEAN calibration: no strategy beats leaving
// the loop alone by the margin, so `purl.strategy = "none"` and the loop is left
// unchanged (a remark reports the decision). This is the compile-time cost model
// declining to fire.

// the loop is untouched: 2-element carry, expval intact, no cut machinery
// CHECK-NOT: func.func private @purl_sample_term
// CHECK: scf.while ({{.*}}) : (i1, !quantum.bit) -> (i1, !quantum.bit)
// CHECK: purl.strategy = "none"
// CHECK: quantum.expval

func.func @held_memory() -> f64 {
  %true = arith.constant true
  %dreg = quantum.alloc( 1) : !quantum.reg
  %draw = quantum.extract %dreg[ 0] : !quantum.reg -> !quantum.bit
  %d0 = quantum.custom "Hadamard"() %draw : !quantum.bit
  // expected-remark@+1 {{purl: not profitable; loop left unchanged}}
  %res:2 = scf.while (%fail = %true, %d = %d0) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    scf.condition(%fail) %fail, %d : i1, !quantum.bit
  } do {
  ^bb0(%f: i1, %dq: !quantum.bit):
    %areg = quantum.alloc( 1) : !quantum.reg
    %a0 = quantum.extract %areg[ 0] : !quantum.reg -> !quantum.bit
    %a1 = quantum.custom "Hadamard"() %a0 : !quantum.bit
    %m, %a2 = quantum.measure %a1 : i1, !quantum.bit
    %ar = scf.if %m -> (!quantum.bit) {
      %ax = quantum.custom "PauliX"() %a2 : !quantum.bit
      scf.yield %ax : !quantum.bit
    } else {
      scf.yield %a2 : !quantum.bit
    }
    %areg1 = quantum.insert %areg[ 0], %ar : !quantum.reg, !quantum.bit
    quantum.dealloc %areg1 : !quantum.reg
    scf.yield %m, %dq : i1, !quantum.bit
  }
  %obs = quantum.namedobs %res#1[ PauliZ] : !quantum.obs
  %e = quantum.expval %obs : f64
  %dreg1 = quantum.insert %dreg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %dreg1 : !quantum.reg
  return %e : f64
}
