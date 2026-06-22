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

// WhileLoopPeelingPass
// ────────────────────
// Peels k straight-line iterations out of scf.while ops that carry an
// `estimated_iterations = k` IntegerAttr (set by the frontend or by the
// profile-guided refinement script).
//
// For each such while op the transformation produces:
//
//   // Peeled iteration 1
//   %cond0 = <inline condition(%inits)>
//   %r = scf.if %cond0 -> resTypes {
//     %newVals0 = <inline body(%pass0)>
//     // Peeled iteration 2
//     %cond1 = <inline condition(%newVals0)>
//     %r1 = scf.if %cond1 -> resTypes {
//       %newVals1 = <inline body(%pass1)>
//       // … k-th peel …
//       // Residual while (no estimated_iterations attr)
//       %r2 = scf.while (%newValsK) : … { same regions }
//       scf.yield %r2
//     } else {
//       scf.yield %pass1   // loop terminated before k-th iteration
//     }
//     scf.yield %r1
//   } else {
//     scf.yield %pass0     // loop terminated before 1st iteration
//   }
//
// The inlined condition and body come from clones of the scf.while's
// `before` and `after` (do) regions respectively, with block arguments
// remapped to the current carried values.
//
// The residual while is a clone of the original op (with new inits and
// without the `estimated_iterations` attr) so the same before/after
// regions keep running for any iterations beyond k.

#define DEBUG_TYPE "while-loop-peeling"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_WHILELOOPPEELINGPASS
#define GEN_PASS_DEF_WHILELOOPPEELINGPASS
#include "Catalyst/Transforms/Passes.h.inc"

// ── helpers ──────────────────────────────────────────────────────────────────

// Clone ops from `block` (without terminator) at the builder position,
// mapping `block`'s arguments to `argVals`. Returns the (unmapped) terminator.
static Operation *cloneBlockBody(OpBuilder &b, Block &block, ValueRange argVals, IRMapping &map)
{
    for (auto [bArg, v] : llvm::zip(block.getArguments(), argVals))
        map.map(bArg, v);
    for (Operation &op : block.without_terminator())
        b.clone(op, map);
    return block.getTerminator();
}

// Inline the `before` region given loop-carried `inits`.
// Populates `map`, returns {condition, pass-through-values}.
static std::pair<Value, SmallVector<Value>> inlineBefore(OpBuilder &b, Region &before,
                                                          ValueRange inits, IRMapping &map)
{
    auto *term = cloneBlockBody(b, before.front(), inits, map);
    auto condOp = cast<scf::ConditionOp>(term);
    Value cond = map.lookupOrDefault(condOp.getCondition());
    SmallVector<Value> pass;
    for (Value v : condOp.getArgs())
        pass.push_back(map.lookupOrDefault(v));
    return {cond, pass};
}

// Inline the `after` (do) region given `passVals` (from scf.condition).
// Populates `map`, returns the values yielded back to the before region.
static SmallVector<Value> inlineAfter(OpBuilder &b, Region &after, ValueRange passVals,
                                       IRMapping &map)
{
    auto *term = cloneBlockBody(b, after.front(), passVals, map);
    auto yieldOp = cast<scf::YieldOp>(term);
    SmallVector<Value> newInits;
    for (Value v : yieldOp.getResults())
        newInits.push_back(map.lookupOrDefault(v));
    return newInits;
}

// Create a clone of `orig` with `newInits` replacing the original init
// operands. The `estimated_iterations` attr is stripped from the clone.
static scf::WhileOp cloneResidual(OpBuilder &b, scf::WhileOp orig, ValueRange newInits)
{
    // Clone without any SSA remapping so that values captured from outer scopes
    // (e.g. constants used both as inits AND inside loop regions) are not
    // accidentally remapped when we later update the init operands.
    IRMapping emptyMap;
    auto *cloned = b.clone(*orig.getOperation(), emptyMap);
    cloned->removeAttr("estimated_iterations");
    // Now update only the init operands (indices 0..N-1 of the while op).
    for (auto [i, v] : llvm::enumerate(newInits))
        cloned->setOperand(static_cast<unsigned>(i), v);
    return cast<scf::WhileOp>(cloned);
}

// ── recursive peeling ─────────────────────────────────────────────────────────

// Emit the peeled representation of `orig` at the current builder position,
// treating `curInits` as the loop-carried values for this level.
// Returns the values that replace the while op's results.
static SmallVector<Value> buildPeeled(OpBuilder &b, Location loc, scf::WhileOp orig,
                                       ValueRange curInits, int64_t peelLeft)
{
    if (peelLeft <= 0) {
        // Base case: emit the residual while loop with new inits.
        auto residual = cloneResidual(b, orig, curInits);
        LLVM_DEBUG(llvm::dbgs() << "while-loop-peeling: emitted residual while\n");
        return SmallVector<Value>(residual.getResults());
    }

    LLVM_DEBUG(llvm::dbgs() << "while-loop-peeling: peeling iteration " << peelLeft << "\n");

    // Inline the condition check.
    IRMapping beforeMap;
    auto [cond, passVals] = inlineBefore(b, orig.getBefore(), curInits, beforeMap);

    // scf.if: then = run body + recurse; else = loop-didn't-run result.
    auto ifOp = scf::IfOp::create(b, loc, orig.getResultTypes(), cond, /*withElseRegion=*/true);

    // ── then branch ───────────────────────────────────────────────────────
    {
        OpBuilder tb = OpBuilder::atBlockEnd(ifOp.thenBlock());
        IRMapping afterMap;
        SmallVector<Value> newInits = inlineAfter(tb, orig.getAfter(), passVals, afterMap);
        SmallVector<Value> inner = buildPeeled(tb, loc, orig, newInits, peelLeft - 1);
        scf::YieldOp::create(tb, loc, inner);
    }

    // ── else branch ───────────────────────────────────────────────────────
    {
        OpBuilder eb = OpBuilder::atBlockEnd(ifOp.elseBlock());
        scf::YieldOp::create(eb, loc, passVals);
    }

    return SmallVector<Value>(ifOp.getResults());
}

// ── pass ─────────────────────────────────────────────────────────────────────

struct WhileLoopPeelingPass : public impl::WhileLoopPeelingPassBase<WhileLoopPeelingPass> {
    using WhileLoopPeelingPassBase::WhileLoopPeelingPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        // Collect all while ops that should be peeled (safe to modify after walk).
        SmallVector<scf::WhileOp> targets;
        func.walk([&](scf::WhileOp op) {
            if (peelFactor >= 0) {
                // CLI override: peel all while ops by `peelFactor` iterations.
                if (peelFactor > 0)
                    targets.push_back(op);
            }
            else if (op->hasAttrOfType<IntegerAttr>("estimated_iterations")) {
                targets.push_back(op);
            }
        });

        for (auto whileOp : targets) {
            int64_t k = peelFactor >= 0
                            ? peelFactor
                            : whileOp->getAttrOfType<IntegerAttr>("estimated_iterations")
                                  .getValue()
                                  .getSExtValue();
            if (k <= 0)
                continue;

            LLVM_DEBUG(llvm::dbgs()
                       << "while-loop-peeling: peeling " << k << " iterations of " << whileOp
                       << "\n");

            builder.setInsertionPoint(whileOp);
            SmallVector<Value> results =
                buildPeeled(builder, whileOp.getLoc(), whileOp, whileOp.getInits(), k);

            whileOp.replaceAllUsesWith(results);
            whileOp.erase();
        }
    }
};

} // namespace catalyst
