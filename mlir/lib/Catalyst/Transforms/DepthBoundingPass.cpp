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

// DepthBoundingPass
// ─────────────────
// For each scf.while carrying a `max_iterations = N` IntegerAttr, adds an i64
// counter to the loop-carried values and ANDs the original loop condition with
// `counter < N`, guaranteeing termination within N iterations.
//
// Transformation sketch for a while op with carry types (T0, T1, ...):
//
//   Original:
//     %res:N = scf.while (%c0 = %v0, ...) : (T0, ...) -> (T0, ...) {
//       ^before(%a0: T0, ...):
//         %cond = <condition(%a0, ...)>
//         scf.condition(%cond) %a0, ...
//     } do {
//       ^after(%b0: T0, ...):
//         %new = <body(%b0, ...)>
//         scf.yield %new, ...
//     }
//
//   After depth-bounding with max_iterations = N:
//     %zero = arith.constant 0 : i64
//     %res:N+1 = scf.while (%c0 = %v0, ..., %ctr = %zero)
//                           : (T0, ..., i64) -> (T0, ..., i64) {
//       ^before(%a0: T0, ..., %cnt: i64):
//         %cond = <condition(%a0, ...)>
//         %lim  = arith.constant N : i64
//         %ok   = arith.cmpi ult, %cnt, %lim : i64
//         %both = arith.andi %cond, %ok : i1
//         scf.condition(%both) %a0, ..., %cnt
//     } do {
//       ^after(%b0: T0, ..., %cnt: i64):
//         %new = <body(%b0, ...)>
//         %one  = arith.constant 1 : i64
//         %ncnt = arith.addi %cnt, %one : i64
//         scf.yield %new, ..., %ncnt
//     }
//
// The N-th result (%res#N, the final counter) is never used externally;
// only %res#0 .. %res#N-1 are mapped to the original results.

#define DEBUG_TYPE "depth-bounding"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

#define GEN_PASS_DECL_DEPTHBOUNDINGPASS
#define GEN_PASS_DEF_DEPTHBOUNDINGPASS
#include "Catalyst/Transforms/Passes.h.inc"

static void rewriteWithBound(OpBuilder &b, scf::WhileOp orig, int64_t maxIter)
{
    Location loc = orig.getLoc();
    Type i64Type = b.getI64Type();

    // ── Build new init list: original inits + counter = 0 ─────────────────
    b.setInsertionPoint(orig);
    Value zero = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
    SmallVector<Value> newInits(orig.getInits());
    newInits.push_back(zero);

    // ── Build new result type list: original result types + i64 ───────────
    SmallVector<Type> newResTypes(orig.getResultTypes());
    newResTypes.push_back(i64Type);

    // Capture from orig before lambdas run (lambdas run synchronously inside
    // scf::WhileOp::create so the pointer stays valid throughout).
    scf::WhileOp *origPtr = &orig;
    int64_t limit = maxIter;

    // ── Create new scf.while with builder lambdas ─────────────────────────
    auto newWhile = scf::WhileOp::create(
        b, loc, TypeRange(newResTypes), ValueRange(newInits),

        // ── Before region ───────────────────────────────────────────────
        [&](OpBuilder &nb, Location nloc, ValueRange beforeArgs) {
            scf::WhileOp &o = *origPtr;
            size_t nOrig = o.getBeforeArguments().size();

            // Map original before-block args to new args [0 .. nOrig-1].
            IRMapping map;
            for (size_t i = 0; i < nOrig; ++i)
                map.map(o.getBeforeArguments()[i], beforeArgs[i]);
            Value counterArg = beforeArgs.back();

            // Clone original before body (without terminator).
            for (Operation &op : o.getBeforeBody()->without_terminator())
                nb.clone(op, map);

            // Retrieve original condition value and pass-through args.
            scf::ConditionOp condOp = o.getConditionOp();
            Value mappedCond = map.lookupOrDefault(condOp.getCondition());

            // Build: not_exceeded = counter < limit
            Value lim = arith::ConstantOp::create(nb, nloc, nb.getI64IntegerAttr(limit));
            Value notExc = arith::CmpIOp::create(nb, nloc, arith::CmpIPredicate::ult,
                                                  counterArg, lim);
            Value combined = arith::AndIOp::create(nb, nloc, mappedCond, notExc);

            // scf.condition(combined, mapped_pass_args..., counter)
            SmallVector<Value> passArgs;
            for (Value v : condOp.getArgs())
                passArgs.push_back(map.lookupOrDefault(v));
            passArgs.push_back(counterArg);
            scf::ConditionOp::create(nb, nloc, combined, passArgs);
        },

        // ── After region ────────────────────────────────────────────────
        [&](OpBuilder &ab, Location aloc, ValueRange afterArgs) {
            scf::WhileOp &o = *origPtr;
            size_t nOrig = o.getAfterArguments().size();

            // Map original after-block args to new args [0 .. nOrig-1].
            IRMapping map;
            for (size_t i = 0; i < nOrig; ++i)
                map.map(o.getAfterArguments()[i], afterArgs[i]);
            Value counterArg = afterArgs.back();

            // Clone original after body (without terminator).
            for (Operation &op : o.getAfterBody()->without_terminator())
                ab.clone(op, map);

            // Collect original yield values (mapped to new values).
            scf::YieldOp yieldOp = o.getYieldOp();
            SmallVector<Value> yieldVals;
            for (Value v : yieldOp.getResults())
                yieldVals.push_back(map.lookupOrDefault(v));

            // Increment counter and yield it.
            Value one = arith::ConstantOp::create(ab, aloc, ab.getI64IntegerAttr(1));
            yieldVals.push_back(arith::AddIOp::create(ab, aloc, counterArg, one));
            scf::YieldOp::create(ab, aloc, yieldVals);
        });

    LLVM_DEBUG(llvm::dbgs() << "depth-bounding: bounded " << orig << " to " << maxIter
                             << " iterations\n");

    // ── Replace original results and erase ────────────────────────────────
    // zip stops at the shorter range (orig has N results, newWhile has N+1).
    for (auto [oldR, newR] : llvm::zip(orig.getResults(), newWhile.getResults()))
        oldR.replaceAllUsesWith(newR);
    orig.erase();
}

// ── Pass ─────────────────────────────────────────────────────────────────────

struct DepthBoundingPass : public impl::DepthBoundingPassBase<DepthBoundingPass> {
    using DepthBoundingPassBase::DepthBoundingPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        SmallVector<scf::WhileOp> targets;
        func.walk([&](scf::WhileOp op) {
            if (op->hasAttrOfType<IntegerAttr>("max_iterations"))
                targets.push_back(op);
        });

        for (auto whileOp : targets) {
            int64_t maxIter =
                whileOp->getAttrOfType<IntegerAttr>("max_iterations").getValue().getSExtValue();
            if (maxIter <= 0)
                continue;
            rewriteWithBound(builder, whileOp, maxIter);
        }
    }
};

} // namespace catalyst
