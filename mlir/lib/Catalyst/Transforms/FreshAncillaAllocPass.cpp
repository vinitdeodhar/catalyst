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

// FreshAncillaAllocPass
// ─────────────────────
// Compile-time, resource-estimator-gated space-vs-fidelity choice for a
// measure-and-reset (MCMR) ancilla in a STATIC loop -- e.g. syndrome extraction.
//
//   MCMR (space-efficient): one ancilla index `c`, reused every round via a
//   measure + conditional-reset. The data qubits idle through the reset/feedback
//   latency on every one of the N rounds -> N * T_fb of extra decoherence.
//
//   Fresh-ancilla (fidelity-efficient): give round i its own ancilla `c + i`
//   and drop the reset. No reset/feedback latency; costs N-1 extra qubits.
//
// This is a *feasibility* choice, not a runtime crossover: fresh ancillas are
// strictly higher fidelity (no reset latency) whenever they fit, so the decision
// is made at compile time from the estimator's static trip count N and a
// `qubit-budget`:
//
//     fresh-ancilla  iff  c + N <= qubit-budget      (else keep MCMR)
//
// The pass fires only on STATIC loops (runtime-bound loops are not a
// compile-time feasibility decision); ResourceAnalysis supplies the
// static-vs-dynamic classification.

#define DEBUG_TYPE "fresh-ancilla-alloc"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace catalyst {

#define GEN_PASS_DECL_FRESHANCILLAALLOCPASS
#define GEN_PASS_DEF_FRESHANCILLAALLOCPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// A matched MCMR loop: the reset scf.if, the ancilla index c, the top-level
// ancilla extract and insert (idx_attr == c), and the reg the reset passes through.
struct McmrMatch {
    scf::IfOp resetIf;
    int64_t ancIdx;
    ExtractOp ancExtract;
    InsertOp ancInsert;
    Value passThroughReg; // else-branch yield of the reset if
};

// Detect the reset scf.if inside `body`: then-region = extract[c] + PauliX + insert[c].
// Returns the ancilla index via `c` and the if op; false if not found.
static bool findResetIf(Block &body, scf::IfOp &resetIf, int64_t &c)
{
    for (Operation &op : body) {
        auto ifOp = dyn_cast<scf::IfOp>(&op);
        if (!ifOp || ifOp.getElseRegion().empty())
            continue;
        // then must contain a PauliX and an extract with a constant idx.
        ExtractOp ex;
        bool hasPauliX = false;
        for (Operation &inner : ifOp.thenBlock()->getOperations()) {
            if (auto e = dyn_cast<ExtractOp>(&inner))
                ex = e;
            if (auto cu = dyn_cast<CustomOp>(&inner))
                if (cu.getGateName() == "PauliX")
                    hasPauliX = true;
        }
        if (hasPauliX && ex && ex.getIdxAttrAttr()) {
            resetIf = ifOp;
            c = ex.getIdxAttrAttr().getInt();
            return true;
        }
    }
    return false;
}

} // namespace

struct FreshAncillaAllocPass
    : public impl::FreshAncillaAllocPassBase<FreshAncillaAllocPass> {
    using FreshAncillaAllocPassBase::FreshAncillaAllocPassBase;

    void runOnOperation() final
    {
        auto module = cast<ModuleOp>(getOperation());
        auto &analysis = getAnalysis<ResourceAnalysis>();

        SmallVector<McmrMatch> matches;
        module.walk([&](scf::ForOp forOp) {
            // Static loops only: a runtime-bound loop is not a compile-time
            // feasibility decision.
            bool isDynamic = false;
            const ResourceResult *body = analysis.getForLoopBody(forOp, isDynamic);
            if (!body || isDynamic)
                return;

            // Static trip count N.
            auto lb = getConstantIntValue(forOp.getLowerBound());
            auto ub = getConstantIntValue(forOp.getUpperBound());
            auto st = getConstantIntValue(forOp.getStep());
            if (!lb || !ub || !st || *st <= 0)
                return;
            int64_t N = (*ub - *lb + *st - 1) / *st;

            scf::IfOp resetIf;
            int64_t c = 0;
            if (!findResetIf(*forOp.getBody(), resetIf, c))
                return;

            // Budget: fresh ancillas occupy c .. c+N-1; must fit the budget.
            if (c + N > qubitBudget) {
                LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] c+N=" << (c + N)
                                        << " > budget=" << qubitBudget << " -> keep MCMR\n");
                return;
            }

            // Find the top-level ancilla extract & insert (idx_attr == c), i.e.
            // the reused-ancilla ops outside the reset if.
            ExtractOp ancEx;
            InsertOp ancIns;
            for (Operation &op : *forOp.getBody()) {
                if (auto e = dyn_cast<ExtractOp>(&op))
                    if (e.getIdxAttrAttr() && e.getIdxAttrAttr().getInt() == c)
                        ancEx = e;
                if (auto i = dyn_cast<InsertOp>(&op))
                    if (i.getIdxAttrAttr() && i.getIdxAttrAttr().getInt() == c)
                        ancIns = i;
            }
            if (!ancEx || !ancIns)
                return;

            Value passReg = cast<scf::YieldOp>(resetIf.elseBlock()->getTerminator()).getOperand(0);
            matches.push_back({resetIf, c, ancEx, ancIns, passReg});
        });

        if (matches.empty()) {
            markAllAnalysesPreserved();
            return;
        }

        for (auto &m : matches) {
            auto forOp = m.resetIf->getParentOfType<scf::ForOp>();
            Value iv = forOp.getInductionVar();
            OpBuilder b(forOp.getBody(), forOp.getBody()->begin());
            Location loc = forOp.getLoc();

            // Per-iteration ancilla index: c + iv  (as i64 for extract/insert).
            Value ivI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), iv);
            Value cVal = arith::ConstantIntOp::create(b, loc, m.ancIdx, 64);
            Value aidx = arith::AddIOp::create(b, loc, cVal, ivI64);

            auto qubitTy = m.ancExtract.getQubit().getType();
            auto qregTy = m.ancInsert.getType();

            // Rebuild ancilla extract with dynamic index.
            OpBuilder be(m.ancExtract);
            auto newEx = ExtractOp::create(be, m.ancExtract.getLoc(), qubitTy,
                                           m.ancExtract.getQreg(), aidx, IntegerAttr());
            m.ancExtract.getQubit().replaceAllUsesWith(newEx.getQubit());
            m.ancExtract.erase();

            // Rebuild ancilla insert with dynamic index.
            OpBuilder bi(m.ancInsert);
            auto newIns = InsertOp::create(bi, m.ancInsert.getLoc(), qregTy,
                                           m.ancInsert.getInQreg(), aidx, IntegerAttr(),
                                           m.ancInsert.getQubit());
            m.ancInsert.getResult().replaceAllUsesWith(newIns.getResult());
            m.ancInsert.erase();

            // Drop the reset: replace its result with the pass-through reg.
            m.resetIf.getResult(0).replaceAllUsesWith(m.passThroughReg);
            m.resetIf.erase();
        }
    }
};

} // namespace catalyst
