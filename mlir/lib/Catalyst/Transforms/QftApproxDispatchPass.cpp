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

// QftApproxDispatchPass
// ─────────────────────
// Approximate-QFT (AQFT) aperture, applied to a QFT whose register width is a
// RUNTIME value.  The compiled QFT is a nested loop:
//
//   scf.for %i = 0 to %n {                 // outer: Hadamard on qubit i
//     quantum.custom "Hadamard" ...
//     scf.for %j = %i+1 to %n {            // inner: controlled-phase cascade
//       quantum.custom "ControlledPhaseShift"(...) qubit[j], qubit[i]
//     }
//   }
//
// The controlled-phase between qubits i and j has angle pi/2^(j-i); rotations
// with (j-i) > b are below the noise floor and dropped (Coppersmith AQFT).  The
// pass rewrites the inner upper bound
//
//     n   ->   min(n, i + b + 1)      with  b = ceil(log2 n) + 2   (Barenco-optimal)
//
// so distances > b are skipped.  Because b(n) = ceil(log2 n)+2 >= n-1 for small
// n, the bound equals n there (exact QFT) and only cuts once n is large enough
// that dropping tail rotations reduces net infidelity -- the aperture is
// self-dispatching in the runtime width n.
//
// Depends on ResourceAnalysis: only fires on a *dynamic* (runtime-bound) inner
// loop whose body is a single ControlledPhaseShift -- the symbolic-cost QFT
// cascade the estimator classifies.  Static loops are left alone (their choice
// folds at compile time).

#define DEBUG_TYPE "qft-approx-dispatch"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace catalyst {

#define GEN_PASS_DECL_QFTAPPROXDISPATCHPASS
#define GEN_PASS_DEF_QFTAPPROXDISPATCHPASS
#include "Catalyst/Transforms/Passes.h.inc"

// Count the ControlledPhaseShift ops directly in a region (not nested loops).
static bool bodyIsSingleCPhase(Region &region)
{
    int64_t nCphase = 0;
    for (Operation &op : region.front()) {
        if (auto cust = dyn_cast<CustomOp>(&op)) {
            if (cust.getGateName() == "ControlledPhaseShift")
                nCphase++;
        }
    }
    return nCphase == 1;
}

// b(n) = ceil(log2 n) + 2, built in IR from index value n via a comparison chain
// ceil(log2 n) = sum_{k>=0} (n > 2^k).  maxBits covers n up to 2^maxBits.
static Value emitCeilLog2Plus2(OpBuilder &b, Location loc, Value n, int maxBits = 7)
{
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value oneIdx = arith::ConstantIndexOp::create(b, loc, 1);
    Value acc = arith::ConstantIndexOp::create(b, loc, 2); // the "+2"
    for (int k = 0; k < maxBits; ++k) {
        Value pow = arith::ConstantIndexOp::create(b, loc, (int64_t)1 << k);
        Value gt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, n, pow);
        // i1 -> {0,1} via select (index_cast would sign-extend true to -1).
        Value inc = arith::SelectOp::create(b, loc, gt, oneIdx, zero);
        acc = arith::AddIOp::create(b, loc, acc, inc);
    }
    return acc; // ceil(log2 n) + 2
}

struct QftApproxDispatchPass
    : public impl::QftApproxDispatchPassBase<QftApproxDispatchPass> {
    using QftApproxDispatchPassBase::QftApproxDispatchPassBase;

    void runOnOperation() final
    {
        auto module = cast<ModuleOp>(getOperation());
        auto &analysis = getAnalysis<ResourceAnalysis>();

        // Find inner loops that are a dynamic single-ControlledPhaseShift cascade.
        SmallVector<scf::ForOp> targets;
        module.walk([&](scf::ForOp inner) {
            bool isDynamic = false;
            const ResourceResult *body = analysis.getForLoopBody(inner, isDynamic);
            if (!body || !isDynamic)
                return;
            if (!bodyIsSingleCPhase(inner.getBodyRegion()))
                return;
            // The inner loop must be nested in an outer scf.for (the QFT outer sweep).
            if (!inner->getParentOfType<scf::ForOp>())
                return;
            targets.push_back(inner);
        });

        if (targets.empty()) {
            markAllAnalysesPreserved();
            return;
        }

        for (scf::ForOp inner : targets) {
            auto outer = inner->getParentOfType<scf::ForOp>();
            Value n = outer.getUpperBound();      // outer runs 0..n  => register width
            Value i = outer.getInductionVar();    // current qubit index
            Value innerUb = inner.getUpperBound();

            OpBuilder b(inner);                    // insert just before the inner loop
            Location loc = inner.getLoc();
            Value bn = emitCeilLog2Plus2(b, loc, n);                       // b = ceil(log2 n)+2
            Value one = arith::ConstantIndexOp::create(b, loc, 1);
            Value bp1 = arith::AddIOp::create(b, loc, bn, one);            // b + 1
            Value lim = arith::AddIOp::create(b, loc, i, bp1);            // i + b + 1
            Value newUb = arith::MinSIOp::create(b, loc, innerUb, lim);   // min(n, i+b+1)
            inner.getUpperBoundMutable().assign(newUb);
        }
    }
};

} // namespace catalyst
