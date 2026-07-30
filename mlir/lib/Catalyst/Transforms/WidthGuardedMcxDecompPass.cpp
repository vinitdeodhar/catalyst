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
// Cost-gated, runtime-symbolic decomposition dispatch for a multi-controlled X
// gate inside a for-loop with a RUNTIME trip count.
//
// Workflow:
//   1. ResourceAnalysis classifies the `scf.for` as dynamic (runtime bound ⇒ the
//      total cost is symbolic in the bound) and gives the per-iteration body.
//   2. This pass synthesizes two versions of the loop body's MCX — ancilla-free
//      (native op, O(c²) gates, 0 ancillas) and V-chain (2c-3 Toffolis on c-2
//      clean ancillas) — and guards them with a SYMBOLIC cost expression:
//
//        saving(N) = (g_af(c) - g_vc(c)) · trip(N)          // symbolic in N
//        scf.if saving(N) > cost-budget { V-chain loop } else { ancilla-free loop }
//
//   trip(N) = ⌈(ub-lb)/step⌉ is materialized in IR from the loop operands, so the
//   guard is evaluated at runtime once N is concrete.  The guard is a
//   qubits-vs-gates trade: spend the V-chain's c-2 ancillas only when the loop is
//   long enough that the accumulated gate saving justifies them.  Both branches
//   are extensionally equal ⇒ no herald.
//
//   Static (compile-time-constant) loops are skipped: their decision is not
//   symbolic, so a runtime guard is not warranted.
//
// V-chain ladder (controls c[0..N-1], target t, ancillas a[0..N-3]):
//   Toffoli(c0,c1,a0); Toffoli(c[j+1],a[j-1],a[j]) j=1..N-3; Toffoli(c[N-1],a[N-3],t);
//   uncompute in reverse.  Total 2N-3 Toffolis, N-2 ancillas.

#define DEBUG_TYPE "width-guarded-mcx-decomp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// True iff `op` is a multi-controlled X we handle: gate "PauliX", single target,
// N>=3 positive (constant-true) controls, not adjoint.  Sets nControls.
static bool isMultiControlledX(CustomOp op, int64_t &nControls)
{
    if (op.getGateName() != "PauliX" || op.getInQubits().size() != 1 || op.getAdjoint())
        return false;
    ValueRange ctrls = op.getInCtrlQubits();
    if (ctrls.size() < 3)
        return false;
    for (Value cv : op.getInCtrlValues()) {
        APInt val;
        if (!matchPattern(cv, m_ConstantInt(&val)) || val.isZero())
            return false;
    }
    nControls = static_cast<int64_t>(ctrls.size());
    return true;
}

// First multi-controlled X in a region (the loop body's MCX), or null.
static CustomOp findMcx(Region &region, int64_t &nControls)
{
    CustomOp found;
    region.walk([&](CustomOp op) {
        int64_t n = 0;
        if (!found && isMultiControlledX(op, n)) {
            found = op;
            nControls = n;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return found;
}

// Rewrite `mcx` into its V-chain Toffoli ladder in place (self-contained builder).
static void emitVChain(CustomOp mcx, int64_t N)
{
    OpBuilder builder(mcx); // insertion point immediately before mcx
    Location loc = mcx.getLoc();

    SmallVector<Value> c(mcx.getInCtrlQubits().begin(), mcx.getInCtrlQubits().end());
    Value t = mcx.getInQubits()[0];

    int64_t nAnc = N - 2;
    SmallVector<Value> a(nAnc);
    for (int64_t i = 0; i < nAnc; ++i)
        a[i] = AllocQubitOp::create(builder, loc).getOutQubit();

    auto tof = [&](Value &x, Value &y, Value &z) {
        SmallVector<Value> q{x, y, z};
        auto op = CustomOp::create(builder, loc, "Toffoli", ValueRange(q));
        x = op.getOutQubits()[0];
        y = op.getOutQubits()[1];
        z = op.getOutQubits()[2];
    };

    tof(c[0], c[1], a[0]);                       // compute a[0] = c0 & c1
    for (int64_t j = 1; j <= N - 3; ++j)
        tof(c[j + 1], a[j - 1], a[j]);           // a[j] = a[j-1] & c[j+1]
    tof(c[N - 1], a[nAnc - 1], t);               // flip target
    for (int64_t j = N - 3; j >= 1; --j)
        tof(c[j + 1], a[j - 1], a[j]);           // uncompute
    tof(c[0], c[1], a[0]);

    for (int64_t i = 0; i < nAnc; ++i)
        DeallocQubitOp::create(builder, loc, a[i]);

    mcx.getOutQubits()[0].replaceAllUsesWith(t);
    for (int64_t i = 0; i < N; ++i)
        mcx.getOutCtrlQubits()[i].replaceAllUsesWith(c[i]);
    mcx->erase();
}

// Per-iteration two-qubit-gate cost of each decomposition of a c-control X.
//   ancilla-free : 24c² - 116c + 156   (validated fully-decomposed 2q count)
//   V-chain      : 6·(2c-3) = 12c - 18  (2c-3 Toffolis, Toffoli = 6 two-qubit)
static int64_t gateSavingPerIter(int64_t c)
{
    int64_t gAf = 24 * c * c - 116 * c + 156;
    int64_t gVc = 12 * c - 18;
    return gAf - gVc;
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

        // Collect dynamic-bound for-loops whose body holds a multi-controlled X.
        SmallVector<std::pair<scf::ForOp, int64_t>> targets;
        module.walk([&](scf::ForOp forOp) {
            bool isDynamic = false;
            const ResourceResult *body = analysis.getForLoopBody(forOp, isDynamic);
            if (!body || !isDynamic) // only symbolic-bound loops get a runtime guard
                return;
            int64_t c = 0;
            if (!findMcx(forOp.getBodyRegion(), c))
                return;
            if (gateSavingPerIter(c) <= 0)
                return;
            targets.push_back({forOp, c});
        });

        if (targets.empty()) {
            markAllAnalysesPreserved();
            return;
        }

        IRRewriter rewriter(&getContext());
        for (auto &[forOp, c] : targets) {
            emitGuardedDispatch(forOp, c, rewriter);
        }
    }

    // Replace `forOp` with an scf.if on the symbolic cost, dispatching to a
    // V-chain clone (then) or the ancilla-free clone (else).
    void emitGuardedDispatch(scf::ForOp forOp, int64_t c, IRRewriter &rewriter)
    {
        rewriter.setInsertionPoint(forOp);
        Location loc = forOp.getLoc();
        int64_t delta = gateSavingPerIter(c);

        // trip(N) = ceildiv(ub - lb, step)
        Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound(), step = forOp.getStep();
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value diff = arith::SubIOp::create(rewriter, loc, ub, lb);
        Value stepm1 = arith::SubIOp::create(rewriter, loc, step, one);
        Value num = arith::AddIOp::create(rewriter, loc, diff, stepm1);
        Value trip = arith::DivUIOp::create(rewriter, loc, num, step);

        // saving(N) = delta * trip(N) ; fire iff saving > cost-budget
        Value deltaC = arith::ConstantIndexOp::create(rewriter, loc, delta);
        Value save = arith::MulIOp::create(rewriter, loc, deltaC, trip);
        Value budget = arith::ConstantIndexOp::create(rewriter, loc, costBudget);
        Value fire =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, save, budget);

        Operation *rawFor = forOp.getOperation();
        int64_t N = c;
        auto ifOp = scf::IfOp::create(
            rewriter, loc, fire,
            [&](OpBuilder &b, Location l) { // then: V-chain version
                Operation *cl = b.clone(*rawFor);
                int64_t cc = 0;
                CustomOp cmcx = findMcx(cast<scf::ForOp>(cl).getBodyRegion(), cc);
                emitVChain(cmcx, N);
                b.setInsertionPointAfter(cl);
                scf::YieldOp::create(b, l, cl->getResults());
            },
            [&](OpBuilder &b, Location l) { // else: ancilla-free (native) version
                Operation *cl = b.clone(*rawFor);
                scf::YieldOp::create(b, l, cl->getResults());
            });

        rewriter.replaceOp(forOp, ifOp.getResults());
    }
};

} // namespace catalyst
