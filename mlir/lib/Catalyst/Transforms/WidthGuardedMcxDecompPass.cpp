// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// WidthGuardedMcxDecompPass
// ─────────────────────────
// Selects the decomposition of a multi-controlled X gate
// (`quantum.custom "PauliX"` with N control qubits) using the peak-qubit-width
// reported by ResourceAnalysis, evaluated against a device `qubit-budget`.
//
// The current program width is taken from the estimator (ResourceAnalysis's
// value-semantics liveness -> ResourceResult::numQubits()); it is NOT a
// hardcoded formula.  Emitting the V-chain adds the ladder's clean ancillas, so
// the post-transform width is
//
//     currentWidth (from estimator) + nAncillas (ladder requirement)
//
// and the V-chain is emitted iff that fits `qubit-budget`; otherwise the
// ancilla-free native op is left in place.
//
// The V-chain allocates N-2 clean ancillas (`quantum.alloc_qb`), computes an
// AND-ladder into them, flips the target with the final ancilla, then uncomputes
// the ladder so every ancilla returns to |0> before `quantum.dealloc_qb`.  The
// transformation is extensionally equal to the original MCX.
//
// Standard clean-ancilla V-chain (controls c[0..N-1], target t, ancillas
// a[0..N-3]):
//
//   Toffoli(c[0],  c[1],   a[0])
//   Toffoli(c[j+1], a[j-1], a[j])     for j = 1 .. N-3      (compute)
//   Toffoli(c[N-1], a[N-3], t)                              (flip target)
//   Toffoli(c[j+1], a[j-1], a[j])     for j = N-3 .. 1      (uncompute)
//   Toffoli(c[0],  c[1],   a[0])
//
// Total: 2N-3 Toffolis, N-2 ancillas.

#define DEBUG_TYPE "width-guarded-mcx-decomp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace catalyst {

#define GEN_PASS_DECL_WIDTHGUARDEDMCXDECOMPPASS
#define GEN_PASS_DEF_WIDTHGUARDEDMCXDECOMPPASS
#include "Catalyst/Transforms/Passes.h.inc"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// True iff `op` is a multi-controlled X we handle: gate "PauliX", a single
// target, N>=3 positive controls, not adjoint.
static bool isMultiControlledX(CustomOp op, int64_t &nControls)
{
    if (op.getGateName() != "PauliX")
        return false;
    if (op.getInQubits().size() != 1)
        return false;
    if (op.getAdjoint())
        return false;
    ValueRange ctrls = op.getInCtrlQubits();
    if (ctrls.size() < 3)
        return false;
    // Require all control values to be constant `true` (positive controls); the
    // V-chain ladder below assumes positive controls.
    for (Value cv : op.getInCtrlValues()) {
        APInt val;
        if (!matchPattern(cv, m_ConstantInt(&val)) || val.isZero())
            return false;
    }
    nControls = static_cast<int64_t>(ctrls.size());
    return true;
}

// Emit a zero-parameter, no-control Toffoli on {a,b,c}; return its 3 out qubits.
static SmallVector<Value, 3> toffoli(IRRewriter &rewriter, Location loc, Value a,
                                     Value b, Value c)
{
    SmallVector<Value> qubits{a, b, c};
    auto op = CustomOp::create(rewriter, loc, "Toffoli", ValueRange(qubits));
    return SmallVector<Value, 3>(op.getOutQubits().begin(), op.getOutQubits().end());
}

// Rewrite one multi-controlled X into its V-chain decomposition in place.
static void emitVChain(CustomOp mcx, int64_t N, IRRewriter &rewriter)
{
    rewriter.setInsertionPoint(mcx);
    Location loc = mcx.getLoc();

    // Working copies of the SSA values for controls and target.
    SmallVector<Value> c(mcx.getInCtrlQubits().begin(), mcx.getInCtrlQubits().end());
    Value t = mcx.getInQubits()[0];

    // Allocate N-2 clean ancillas.
    int64_t nAnc = N - 2;
    SmallVector<Value> a(nAnc);
    for (int64_t i = 0; i < nAnc; ++i) {
        a[i] = AllocQubitOp::create(rewriter, loc).getOutQubit();
    }

    auto tof = [&](Value &x, Value &y, Value &z) {
        auto r = toffoli(rewriter, loc, x, y, z);
        x = r[0];
        y = r[1];
        z = r[2];
    };

    // ── compute ladder ──
    tof(c[0], c[1], a[0]);                        // a[0] = c0 & c1
    for (int64_t j = 1; j <= N - 3; ++j) {
        tof(c[j + 1], a[j - 1], a[j]);            // a[j] = a[j-1] & c[j+1]
    }
    // ── flip target ──
    tof(c[N - 1], a[nAnc - 1], t);                // t ^= a[N-3] & c[N-1]
    // ── uncompute ladder ──
    for (int64_t j = N - 3; j >= 1; --j) {
        tof(c[j + 1], a[j - 1], a[j]);
    }
    tof(c[0], c[1], a[0]);

    // Ancillas are back to |0>; release them.
    for (int64_t i = 0; i < nAnc; ++i) {
        DeallocQubitOp::create(rewriter, loc, a[i]);
    }

    // Rewire users of the MCX results to the ladder's final SSA values.
    rewriter.replaceAllUsesWith(mcx.getOutQubits()[0], t);
    for (int64_t i = 0; i < N; ++i) {
        rewriter.replaceAllUsesWith(mcx.getOutCtrlQubits()[i], c[i]);
    }
    rewriter.eraseOp(mcx);
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct WidthGuardedMcxDecompPass
    : public impl::WidthGuardedMcxDecompPassBase<WidthGuardedMcxDecompPass> {
    using WidthGuardedMcxDecompPassBase::WidthGuardedMcxDecompPassBase;

    void runOnOperation() final
    {
        auto module = cast<ModuleOp>(getOperation());
        auto &analysis = getAnalysis<ResourceAnalysis>();

        // Current peak qubit width of the program, from the estimator's
        // value-semantics liveness (ResourceResult::numQubits()).  Taken as the
        // max over all analysed functions so the register-allocating entry (or
        // the widest loop body) dominates.  No width formula is assumed here.
        int64_t currentWidth = 0;
        for (const auto &entry : analysis.getResults()) {
            currentWidth = std::max(currentWidth, entry.second.numQubits());
        }
        if (currentWidth == 0) {
            // Estimator reported nothing; make no (unfounded) decision.
            markAllAnalysesPreserved();
            return;
        }

        // Collect MCX ops whose V-chain form fits the budget.  The only added
        // width is the ladder's clean-ancilla requirement (nAnc = N-2), which is
        // exactly the number of quantum.alloc_qb the emitter will issue.
        SmallVector<std::pair<CustomOp, int64_t>> targets;
        module.walk([&](CustomOp op) {
            int64_t N = 0;
            if (isMultiControlledX(op, N)) {
                int64_t nAnc = N - 2;
                int64_t postWidth = currentWidth + nAnc;
                LLVM_DEBUG(llvm::dbgs()
                           << "[" << DEBUG_TYPE << "] MCX N=" << N
                           << " currentWidth=" << currentWidth
                           << " +ancillas=" << nAnc << " -> postWidth=" << postWidth
                           << " budget=" << qubitBudget << "\n");
                if (postWidth <= qubitBudget) {
                    targets.push_back({op, N});
                }
            }
        });

        if (targets.empty()) {
            markAllAnalysesPreserved();
            return;
        }

        IRRewriter rewriter(&getContext());
        for (auto &[mcx, N] : targets) {
            emitVChain(mcx, N, rewriter);
        }
    }
};

} // namespace catalyst
