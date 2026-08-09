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

// FactoredUnrollPass
// ──────────────────
// Estimator-guided factored unrolling that trades qubit width for circuit depth /
// feedback latency, for BOTH loop kinds:
//
//   scf.for  (static/bounded): classic strip-mine — unroll the body by factor F
//     (reuses mlir::loopUnrollByFactor). Each cloned iteration's `alloc_qb` is a
//     fresh SSA value ⇒ a distinct ancilla, so the F copies run on disjoint qubits
//     and expose parallelism.
//
//   scf.while (measurement-driven, RUS): speculative batch — clone the trial in the
//     `before` region F times on disjoint ancillas, OR the F success flags, and
//     select the winning candidate (nested scf.if that deallocs the F-1 losers).
//     Terminates the loop if ANY trial succeeds, cutting expected feedback rounds
//     from 1/p toward 1/(1-(1-p)^F). Trials are independent on disjoint qubits, so
//     discarding losers is clean (product state).
//
// F is chosen by the resource estimator to fit the hardware:
//     F = floor((qubit-budget - W_target) / W_trial)     [when qubit-budget > 0]
// or given explicitly via `unroll-factor` (for tests). W_target / W_trial come from
// getAnalysis<ResourceAnalysis>() (§4.1 peak width). F <= 1 ⇒ no-op.

#define DEBUG_TYPE "factored-unroll"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace catalyst {

#define GEN_PASS_DECL_FACTOREDUNROLLPASS
#define GEN_PASS_DEF_FACTOREDUNROLLPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Dealloc every QubitType value in `vals` (the losing trials' candidates).
static void deallocQubits(OpBuilder &b, Location loc, ArrayRef<Value> vals)
{
    for (Value v : vals)
        if (isa<QubitType>(v.getType()))
            DeallocQubitOp::create(b, loc, v);
}

// Build a nested scf.if selecting the winning candidate by flag priority; each
// branch deallocs the other clones' qubit candidates. Returns the selected args.
static SmallVector<Value> buildWinnerSelect(OpBuilder &b, Location loc,
                                            ArrayRef<Value> flags,
                                            ArrayRef<SmallVector<Value>> cands,
                                            unsigned i)
{
    TypeRange resTypes = ValueRange(cands[0]).getTypes();
    // Base case: last clone — no flag test, it is the fallthrough winner.
    if (i + 1 == flags.size()) {
        for (unsigned j = 0; j < i; ++j)
            deallocQubits(b, loc, cands[j]);
        return SmallVector<Value>(cands[i]);
    }
    auto ifOp = scf::IfOp::create(b, loc, resTypes, flags[i],
                                  /*addThenBlock=*/true, /*addElseBlock=*/true);
    // then: flag_i succeeded → dealloc the other clones' qubits, yield cands[i].
    {
        OpBuilder tb = OpBuilder::atBlockBegin(ifOp.thenBlock());
        for (unsigned j = 0; j < flags.size(); ++j)
            if (j != i)
                deallocQubits(tb, loc, cands[j]);
        scf::YieldOp::create(tb, loc, cands[i]);
    }
    // else: recurse on the remaining clones.
    {
        OpBuilder eb = OpBuilder::atBlockBegin(ifOp.elseBlock());
        SmallVector<Value> w = buildWinnerSelect(eb, loc, flags, cands, i + 1);
        scf::YieldOp::create(eb, loc, w);
    }
    return SmallVector<Value>(ifOp.getResults());
}

} // namespace

struct FactoredUnrollPass : public impl::FactoredUnrollPassBase<FactoredUnrollPass> {
    using FactoredUnrollPassBase::FactoredUnrollPassBase;

    // F from options: explicit unroll-factor, else budget / trial-width, else 1.
    int64_t resolveFactor(scf::WhileOp trialLoop)
    {
        if (unrollFactor > 0)
            return unrollFactor;
        if (qubitBudget > 0) {
            auto &analysis = getAnalysis<ResourceAnalysis>();
            // W_trial = peak alloc-qubits in one trial body; W_target = carried qubits.
            int64_t wTrial = 1, wTarget = 0;
            if (const ResourceResult *r = analysis.getResult(
                    trialLoop->getParentOfType<func::FuncOp>().getName())) {
                wTarget = r->numArgQubits;
                wTrial = std::max<int64_t>(1, r->numAllocQubits);
            }
            return std::max<int64_t>(1, (qubitBudget - wTarget) / wTrial);
        }
        return 1;
    }

    void runOnOperation() final
    {
        auto module = cast<ModuleOp>(getOperation());

        // ── scf.for: strip-mine (reuse the MLIR utility) ──
        int64_t forF = unrollFactor > 0 ? unrollFactor : (qubitBudget > 0 ? qubitBudget : 1);
        if (forF > 1) {
            SmallVector<scf::ForOp> fors;
            module.walk([&](scf::ForOp op) { fors.push_back(op); });
            for (scf::ForOp f : fors)
                (void)loopUnrollByFactor(f, static_cast<uint64_t>(forF));
        }

        // ── scf.while: speculative batch ──
        SmallVector<scf::WhileOp> whiles;
        module.walk([&](scf::WhileOp op) { whiles.push_back(op); });
        for (scf::WhileOp w : whiles) {
            int64_t F = resolveFactor(w);
            if (F > 1)
                batchWhile(w, F);
        }
    }

    void batchWhile(scf::WhileOp w, int64_t F)
    {
        Block *before = w.getBeforeBody();
        auto cond = w.getConditionOp();
        Value flag0 = cond.getCondition();
        SmallVector<Value> args0(cond.getArgs().begin(), cond.getArgs().end());

        OpBuilder b(cond);
        Location loc = w.getLoc();

        SmallVector<Value> flags{flag0};
        SmallVector<SmallVector<Value>> cands{args0};

        // Snapshot the trial ops before cloning: cloning inserts into `before`
        // (in front of the terminator), so iterating the live block while cloning
        // would re-clone the fresh clones and never terminate.
        SmallVector<Operation *> trialOps;
        for (Operation &op : before->without_terminator())
            trialOps.push_back(&op);

        // Clone the trial (all before-body ops except the condition) F-1 times.
        // Clones read the same block args (independent trials from the same init);
        // each cloned alloc_qb is a fresh, disjoint ancilla.
        for (int64_t c = 1; c < F; ++c) {
            IRMapping map;
            for (Operation *op : trialOps)
                b.clone(*op, map);
            flags.push_back(map.lookupOrDefault(flag0));
            SmallVector<Value> ar;
            for (Value a : args0)
                ar.push_back(map.lookupOrDefault(a));
            cands.push_back(ar);
        }

        // batch_flag = OR of the F flags.
        Value batchFlag = flags[0];
        for (int64_t i = 1; i < F; ++i)
            batchFlag = arith::OrIOp::create(b, loc, batchFlag, flags[i]);

        // winner = priority select among candidates (deallocs losers).
        SmallVector<Value> winner = buildWinnerSelect(b, loc, flags, cands, 0);

        // Replace the condition terminator.
        scf::ConditionOp::create(b, loc, batchFlag, winner);
        cond.erase();
    }
};

} // namespace catalyst
