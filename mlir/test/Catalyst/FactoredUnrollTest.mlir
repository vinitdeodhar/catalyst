// RUN: quantum-opt --factored-unroll="unroll-factor=2" --split-input-file %s | FileCheck %s
// RUN: quantum-opt --factored-unroll="unroll-factor=1" --split-input-file %s | FileCheck %s --check-prefix=NOOP
// RUN: quantum-opt --factored-unroll="qubit-budget=5" --split-input-file %s | FileCheck %s --check-prefix=BUDGET

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: scf.for strip-mine. Trip count 4, unroll-factor=2 (divides evenly, no
//         epilogue) ⇒ the body is duplicated: two alloc_qb / CNOT / dealloc_qb per
//         iteration, each clone allocating a fresh, disjoint ancilla.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @strip_for
// CHECK: scf.for
// The body is duplicated: two disjoint ancilla allocs and two CNOTs.
// CHECK-DAG: quantum.alloc_qb
// CHECK-DAG: quantum.alloc_qb
// CHECK-DAG: quantum.custom "CNOT"
// CHECK-DAG: quantum.custom "CNOT"
func.func @strip_for(%q0: !quantum.bit) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %q0) -> (!quantum.bit) {
        %anc = quantum.alloc_qb : !quantum.bit
        %g:2 = quantum.custom "CNOT"() %a, %anc : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %g#1 : !quantum.bit
        scf.yield %g#0 : !quantum.bit
    }
    return %r : !quantum.bit
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: scf.while speculative batch (RUS). unroll-factor=2 ⇒ the trial (before
//         region) is cloned once on a disjoint ancilla; the two success flags are
//         OR-ed into the loop condition, and a priority scf.if selects the winning
//         candidate while deallocating the loser.
//
//         Under unroll-factor=1 the pass is a no-op: a single trial, no OR, no
//         selection (NOOP prefix).
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @batch_while
// CHECK: scf.while
// Two cloned trials on disjoint ancillas:
// CHECK-DAG: quantum.alloc_qb
// CHECK-DAG: quantum.alloc_qb
// CHECK-DAG: quantum.measure
// CHECK-DAG: quantum.measure
// batch_flag = OR of the two success flags:
// CHECK: arith.ori
// priority winner-select that deallocs the losing candidate:
// CHECK: scf.if
// CHECK: quantum.dealloc_qb
// CHECK: scf.condition

// NOOP-LABEL: func.func @batch_while
// NOOP: scf.while
// NOOP-COUNT-1: quantum.alloc_qb
// NOOP-NOT: arith.ori
// NOOP-NOT: scf.if

// Estimator-driven F: budget=5, W_target=1 (the qubit arg), W_trial=1 (one
// alloc_qb per trial) ⇒ F = (5-1)/1 = 4 speculative trials.
// BUDGET-LABEL: func.func @batch_while
// BUDGET-DAG: quantum.alloc_qb
// BUDGET-DAG: quantum.alloc_qb
// BUDGET-DAG: quantum.alloc_qb
// BUDGET-DAG: quantum.alloc_qb
// BUDGET: arith.ori
// BUDGET: scf.if
// BUDGET: scf.condition
func.func @batch_while(%q0: !quantum.bit) -> !quantum.bit {
    %res = scf.while (%arg0 = %q0) : (!quantum.bit) -> !quantum.bit {
        // trial body: allocate an ancilla, prepare it, measure the success flag.
        %anc = quantum.alloc_qb : !quantum.bit
        %h = quantum.custom "Hadamard"() %anc : !quantum.bit
        %flag, %out = quantum.measure %h : i1, !quantum.bit
        scf.condition(%flag) %out : !quantum.bit
    } do {
    ^bb0(%body: !quantum.bit):
        quantum.dealloc_qb %body : !quantum.bit
        scf.yield %q0 : !quantum.bit
    }
    return %res : !quantum.bit
}
