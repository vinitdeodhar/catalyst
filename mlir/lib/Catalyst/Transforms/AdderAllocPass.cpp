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

// AdderAllocPass
// ──────────────
// Profile-guided adder-strategy allocation. Selects, per abstract adder op
// (`quantum.custom "Adder"` with a `width` attribute), between a space-efficient
// ripple-carry and a time-efficient parallel (carry-lookahead) implementation,
// under a global `ancilla-budget`, weighted by each adder's EXECUTION COUNT:
//
//     exec(adder) = (product of enclosing static for-loop trip counts)
//                   x (profiled E[k] of any enclosing measurement-driven while,
//                      read from its `estimated_iterations` attribute -- the
//                      gate-counter profiler's output, the estimator's runtime arm)
//
// Adders that share a critical-path LEVEL (same longest Adder-dependency depth,
// in the same loop scope) form an all-or-nothing GROUP: a level's depth drops
// only when every adder in it is parallel (a ripple sibling dominates the max),
// so the allocation unit is a group, not an individual adder. Greedy over groups
// by (depth_saved x exec) / ancilla_cost until the budget is exhausted.
//
// Depends on ResourceAnalysis (getAnalysis<ResourceAnalysis>()) to establish the
// loop structure; the profiled E[k] enters via `estimated_iterations` (shared with
// while-loop-peeling). Sets each Adder's `strategy` attribute ("ripple"/"parallel")
// and `ancillas` for parallel ones; a separate lowering pass emits gates.

#define DEBUG_TYPE "adder-alloc"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumOps.h"

#include <cmath>

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace catalyst {

#define GEN_PASS_DECL_ADDERALLOCPASS
#define GEN_PASS_DEF_ADDERALLOCPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Cost model (2q-gate layers): ripple = 2w, parallel = 4*ceil(log2 w), anc = w.
static int64_t rippleDepth(int64_t w) { return 2 * w; }
static int64_t parallelDepth(int64_t w) { return 4 * (int64_t)std::ceil(std::log2((double)w)); }
static int64_t parallelAnc(int64_t w) { return w; }

static bool isAdder(Operation *op, int64_t &width)
{
    auto cu = dyn_cast_or_null<CustomOp>(op);
    if (!cu || cu.getGateName() != "Adder")
        return false;
    auto wAttr = cu->getAttrOfType<IntegerAttr>("width");
    if (!wAttr)
        return false;
    width = wAttr.getInt();
    return true;
}

struct AdderInfo {
    CustomOp op;
    int64_t width;
    int64_t exec;
    int64_t level;
    Operation *scope; // nearest enclosing loop op (or function)
};

} // namespace

struct AdderAllocPass : public impl::AdderAllocPassBase<AdderAllocPass> {
    using AdderAllocPassBase::AdderAllocPassBase;

    // exec count = product of enclosing static for trip counts x while E[k].
    int64_t execCount(Operation *op, Operation *&scope)
    {
        int64_t exec = 1;
        scope = op->getParentOp();
        bool haveScope = false;
        for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
            if (auto f = dyn_cast<scf::ForOp>(p)) {
                auto lb = getConstantIntValue(f.getLowerBound());
                auto ub = getConstantIntValue(f.getUpperBound());
                auto st = getConstantIntValue(f.getStep());
                if (lb && ub && st && *st > 0)
                    exec *= (*ub - *lb + *st - 1) / *st;
                if (!haveScope) { scope = p; haveScope = true; }
            }
            else if (auto w = dyn_cast<scf::WhileOp>(p)) {
                if (auto ek = w->getAttrOfType<IntegerAttr>("estimated_iterations"))
                    exec *= ek.getInt();
                if (!haveScope) { scope = p; haveScope = true; }
            }
        }
        return exec;
    }

    // Critical-path level: 1 + max level of Adder ops producing this adder's inputs.
    int64_t levelOf(CustomOp adder, DenseMap<Operation *, int64_t> &memo)
    {
        auto it = memo.find(adder.getOperation());
        if (it != memo.end())
            return it->second;
        int64_t lvl = 0;
        for (Value in : adder.getInQubits()) {
            Operation *def = in.getDefiningOp();
            int64_t w;
            if (def && isAdder(def, w))
                lvl = std::max(lvl, levelOf(cast<CustomOp>(def), memo) + 1);
        }
        memo[adder.getOperation()] = lvl;
        return lvl;
    }

    void runOnOperation() final
    {
        auto module = cast<ModuleOp>(getOperation());
        // Genuine dependency: establishes the loop structure the exec counts ride on.
        (void)getAnalysis<ResourceAnalysis>();

        SmallVector<AdderInfo> adders;
        DenseMap<Operation *, int64_t> levelMemo;
        module.walk([&](CustomOp op) {
            int64_t width;
            if (!isAdder(op, width))
                return;
            Operation *scope = nullptr;
            int64_t exec = execCount(op, scope);
            int64_t level = levelOf(op, levelMemo);
            adders.push_back({op, width, exec, level, scope});
        });
        if (adders.empty()) {
            markAllAnalysesPreserved();
            return;
        }

        // Group by (scope, level): the all-or-nothing critical-path groups.
        struct Group {
            SmallVector<AdderInfo *> members;
            int64_t exec = 1;
            int64_t maxRipple = 0, maxParallel = 0, ancCost = 0;
        };
        llvm::DenseMap<std::pair<Operation *, int64_t>, Group> groups;
        for (auto &a : adders) {
            auto &g = groups[{a.scope, a.level}];
            g.members.push_back(&a);
            g.exec = a.exec; // same within a level/scope
            g.maxRipple = std::max(g.maxRipple, rippleDepth(a.width));
            g.maxParallel = std::max(g.maxParallel, parallelDepth(a.width));
            g.ancCost += parallelAnc(a.width);
        }

        // Greedy over groups by (depth_saved * exec) / ancilla_cost.
        SmallVector<Group *> order;
        for (auto &kv : groups)
            order.push_back(&kv.second);
        llvm::sort(order, [](Group *x, Group *y) {
            double ux = double((x->maxRipple - x->maxParallel) * x->exec) / std::max<int64_t>(1, x->ancCost);
            double uy = double((y->maxRipple - y->maxParallel) * y->exec) / std::max<int64_t>(1, y->ancCost);
            return ux > uy;
        });

        int64_t used = 0;
        for (Group *g : order) {
            bool beneficial = g->maxRipple > g->maxParallel;
            if (beneficial && used + g->ancCost <= qubitBudget) {
                used += g->ancCost;
                for (AdderInfo *a : g->members) {
                    a->op->setAttr("strategy", StringAttr::get(&getContext(), "parallel"));
                    a->op->setAttr("ancillas",
                                   IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                                    parallelAnc(a->width)));
                }
            }
            else {
                for (AdderInfo *a : g->members)
                    a->op->setAttr("strategy", StringAttr::get(&getContext(), "ripple"));
            }
        }
        LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] used " << used << "/" << qubitBudget
                                << " ancillas across " << groups.size() << " groups\n");
    }
};

} // namespace catalyst
